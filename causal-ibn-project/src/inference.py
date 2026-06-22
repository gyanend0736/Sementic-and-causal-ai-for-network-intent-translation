"""
inference.py — Counterfactual causal safety check.

Two counterfactual questions are asked per metric:

  Q1 — Intent effect (what happens ON the target node):
       "If srv_gaming adds +10 Mbps, does its own latency breach SLA?"
       treatment: srv_gaming_bandwidth_used_mbps
       → catches self-congestion from the intent itself

  Q2 — Upstream effect (what the switch is doing TO the target node):
       "Given sw1 is currently flooding at 141 Mbps,
        what is srv_gaming_latency_ms right now vs if sw1 were normal?"
       treatment: sw1_bandwidth_used_mbps  (the upstream switch)
       → catches the cross-node congestion path that makes gaming lag
         even when the gaming server's own bandwidth looks fine

This is the correct framing. Gaming at 10 Mbps on a 100 Mbps link
looks self-safe, but if sw1 is at 99% buffer capacity the inter-node
causal edge sw1_bw → srv_gaming_latency causes the breach.
The system must detect that and tell the resolver to throttle sw1's
other tenants, not gaming itself.
"""

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm
from dowhy.gcm import counterfactual_samples
from dowhy.gcm.util.general import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(BASE_DIR)
TELEMETRY_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw_telemetry.csv')

TELEMETRY_WINDOW = 50
SPIKE_PERCENTILE = 95
N_CF_SAMPLES     = 30      # repeated live rows for counterfactual distribution

set_random_seed(42)


# ---------------------------------------------------------------------------
# 1. Build NetworkX DAG from topology
# ---------------------------------------------------------------------------

def _build_nx_graph(network_state):
    METRICS = [
        "active_flows", "bandwidth_used_mbps", "buffer_occupancy",
        "latency_ms", "packet_loss_percent", "jitter_ms",
        "cpu_utilization_percent",
    ]
    INTRA_EDGES = [
        ("active_flows",            "bandwidth_used_mbps"),
        ("bandwidth_used_mbps",     "buffer_occupancy"),
        ("buffer_occupancy",        "latency_ms"),
        ("buffer_occupancy",        "packet_loss_percent"),
        ("buffer_occupancy",        "jitter_ms"),
        ("cpu_utilization_percent", "latency_ms"),
        ("cpu_utilization_percent", "packet_loss_percent"),
        ("cpu_utilization_percent", "jitter_ms"),
    ]

    G = nx.DiGraph()
    for node in network_state["topology"].get("nodes", []):
        nid = node["id"]
        for metric in METRICS:
            G.add_node(f"{nid}_{metric}")
        for src_m, tgt_m in INTRA_EDGES:
            G.add_edge(f"{nid}_{src_m}", f"{nid}_{tgt_m}")

    for link in network_state["topology"].get("links", []):
        src, tgt = link["source"], link["target"]
        G.add_edge(f"{src}_bandwidth_used_mbps", f"{tgt}_latency_ms")
        G.add_edge(f"{src}_bandwidth_used_mbps", f"{tgt}_packet_loss_percent")
        G.add_edge(f"{src}_bandwidth_used_mbps", f"{tgt}_jitter_ms")

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Causal graph has cycles: {cycles}")
    return G


# ---------------------------------------------------------------------------
# 2. Build multi-node dataframe — so inter-node edges have data to learn from
# ---------------------------------------------------------------------------

def _build_multi_node_df(df_all, node_ids, window=TELEMETRY_WINDOW):
    """
    Pivots the long-format telemetry CSV into a wide dataframe where
    each column is "<node_id>_<metric>".

    This is what makes the inter-node causal edges learnable.
    Without this, the GCM only sees one node at a time and the
    sw1_bw → srv_gaming_latency path has no data to train on.

    Aligns rows by position (same timestamp index across nodes).
    """
    frames = {}
    for nid in node_ids:
        df_n = (df_all[df_all["node_id"] == nid]
                .drop(columns=["node_id", "timestamp"], errors="ignore")
                .tail(window)
                .reset_index(drop=True))
        if df_n.empty:
            continue
        df_n = df_n.rename(columns={c: f"{nid}_{c}" for c in df_n.columns})
        frames[nid] = df_n

    if not frames:
        return pd.DataFrame()

    # Align to the shortest node history
    min_len = min(len(f) for f in frames.values())
    aligned = [f.tail(min_len).reset_index(drop=True) for f in frames.values()]
    return pd.concat(aligned, axis=1)


# ---------------------------------------------------------------------------
# 3. Fit GCM on multi-node dataframe
# ---------------------------------------------------------------------------

def _fit_gcm(G, df_wide):
    available = set(df_wide.columns)
    keep      = [n for n in G.nodes if n in available]
    G_trimmed = G.subgraph(keep).copy()

    if not nx.is_directed_acyclic_graph(G_trimmed):
        raise ValueError("Trimmed graph is not a DAG.")

    # InvertibleStructuralCausalModel required for counterfactual_samples()
    causal_model = gcm.InvertibleStructuralCausalModel(G_trimmed)
    gcm.auto.assign_causal_mechanisms(causal_model, df_wide[keep], override_models=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcm.fit(causal_model, df_wide[keep])

    return causal_model, keep


# ---------------------------------------------------------------------------
# 4. Run a single counterfactual question
# ---------------------------------------------------------------------------

def _run_counterfactual(causal_model, live_row_wide, treatment_var,
                        outcome_var, intervened_value):
    """
    Asks: "Given the network is in state live_row_wide RIGHT NOW,
           if treatment_var is fixed to intervened_value,
           what does outcome_var become?"

    Returns (p50, p95) across N_CF_SAMPLES noise samples.
    """
    repeated = pd.concat([live_row_wide] * N_CF_SAMPLES, ignore_index=True)

    cf = counterfactual_samples(
        causal_model,
        interventions={treatment_var: lambda x: intervened_value},
        observed_data=repeated
    )

    if outcome_var not in cf.columns:
        raise KeyError(f"'{outcome_var}' not in counterfactual output.")

    values = cf[outcome_var].dropna().values
    return float(np.percentile(values, 50)), float(np.percentile(values, SPIKE_PERCENTILE))


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

def _find_server_id(intent, network_state):
    target = intent.get("requirements", {}).get("application", "").lower()
    for node in network_state["topology"].get("nodes", []):
        if node.get("hosted_service", "")!= None  and node.get("hosted_service", "").lower() == target:
            return node["id"]
    return None


def _find_upstream_switch(server_id, network_state):
    """Return the switch directly connected to server_id, or None."""
    for link in network_state["topology"].get("links", []):
        src, tgt = link.get("source"), link.get("target")
        if tgt == server_id:
            return src
        if src == server_id:
            return tgt
    return None


def _get_danger_limit(metric, sla_thresholds):
    FALLBACK = {"latency_ms": 50.0, "packet_loss_percent": 1.0, "jitter_ms": 10.0}
    return sla_thresholds.get(metric, FALLBACK.get(metric, 50.0))


def _required_mbps(intent):
    bw = (intent.get("requirements", {})
               .get("constraints", {})
               .get("bandwidth_guarantee", "0Mbps"))
    try:
        return float(str(bw).replace("Mbps", "").replace("mbps", "").strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 6. Public entry point
# ---------------------------------------------------------------------------

def evaluate_intent_safety(intent, network_state):
    """
    Two counterfactual questions per metric:

    Q1 — Self-congestion check:
      "If we add +delta_mbps to srv_gaming's own bandwidth,
       does srv_gaming_latency breach SLA?"
      This catches: the intent itself overloading the target server.

    Q2 — Upstream congestion check:
      "Given sw1 is currently at X Mbps (flooding),
       what is srv_gaming_latency NOW vs if sw1 were at baseline?"
      This catches: background apps on the same switch causing lag,
      even when the gaming server's own usage is low.
      Baseline = median sw1 bandwidth over the training window.

    A metric FAILS if EITHER question shows a P95 breach.
    The counterfactual_details dict reports which question failed
    so the conflict resolver knows whether to throttle the target
    node's own bandwidth or the upstream switch's tenants.
    """
    print("\n   -> [Inference Engine] Running two-question counterfactual analysis...")

    target_server_id = _find_server_id(intent, network_state)
    if not target_server_id:
        return {"is_safe": False,
                "predicted_conflict": "Target service not found in topology.",
                "counterfactual_details": {}}

    upstream_switch_id = _find_upstream_switch(target_server_id, network_state)

    target_outcomes = (intent.get("requirements", {})
                             .get("critical_metrics", ["latency_ms"]))
    sla_thresholds  = intent.get("requirements", {}).get("sla_thresholds", {})
    delta_mbps      = _required_mbps(intent)

    print(f"   -> Target: '{target_server_id}'  "
          f"Switch: '{upstream_switch_id}'  "
          f"Metrics: {target_outcomes}  "
          f"Intent delta: +{delta_mbps} Mbps")

    # --- Load all telemetry, build wide multi-node dataframe ---
    try:
        df_all = pd.read_csv(TELEMETRY_CSV)
    except FileNotFoundError:
        return {"is_safe": False,
                "predicted_conflict": "Telemetry CSV not found.",
                "counterfactual_details": {}}

    all_node_ids = list({n["id"] for n in
                         network_state["topology"].get("nodes", [])})
    df_wide = _build_multi_node_df(df_all, all_node_ids)

    if df_wide.empty:
        return {"is_safe": False,
                "predicted_conflict": "Could not build multi-node telemetry dataframe.",
                "counterfactual_details": {}}

    # Live row = most recent snapshot (all nodes, wide format)
    live_row_wide = df_wide.tail(1).copy()

    # --- Build graph and fit GCM on the wide dataframe ---
    try:
        G = _build_nx_graph(network_state)
        causal_model, active_nodes = _fit_gcm(G, df_wide)
        print(f"   -> GCM fitted on {len(df_wide)}-row wide dataframe, "
              f"{len(active_nodes)} active nodes.")
    except Exception as e:
        print(f"   [!] GCM build/fit failed: {e}")
        return {"is_safe": False,
                "predicted_conflict": f"Causal model fitting failed: {e}",
                "counterfactual_details": {}}

    # Treatment variables
    self_treatment_var     = f"{target_server_id}_bandwidth_used_mbps"
    upstream_treatment_var = f"{upstream_switch_id}_bandwidth_used_mbps" \
                             if upstream_switch_id else None

    # Current bandwidth values from the live row
    current_self_bw = float(live_row_wide[self_treatment_var].iloc[0]) \
                      if self_treatment_var in live_row_wide.columns else 0.0
    current_sw_bw   = float(live_row_wide[upstream_treatment_var].iloc[0]) \
                      if upstream_treatment_var and \
                         upstream_treatment_var in live_row_wide.columns else 0.0

    # Upstream baseline = median switch BW from the NORMAL phase rows only
    # (first third of the training window, before congestion built up).
    # Using all 50 rows would include the crisis phase and inflate the baseline.
    normal_rows = max(1, len(df_wide) // 3)
    upstream_baseline_bw = float(
        df_wide[upstream_treatment_var].iloc[:normal_rows].median()
    ) if upstream_treatment_var and \
         upstream_treatment_var in df_wide.columns else current_sw_bw

    print(f"   -> Self BW now: {round(current_self_bw,1)} Mbps  "
          f"Switch BW now: {round(current_sw_bw,1)} Mbps  "
          f"Switch baseline: {round(upstream_baseline_bw,1)} Mbps")

    # --- Run both counterfactuals per metric ---
    details       = {}
    all_passed    = True
    first_failure = None

    for metric in target_outcomes:
        outcome_var = f"{target_server_id}_{metric}"

        if outcome_var not in df_wide.columns:
            print(f"      [!] Skipping '{metric}' — not in telemetry.")
            continue
        if outcome_var not in causal_model.graph.nodes:
            print(f"      [!] Skipping '{metric}' — not in fitted graph.")
            continue

        limit = _get_danger_limit(metric, sla_thresholds)
        unit  = "ms" if "ms" in metric else "%"
        current_val = round(float(live_row_wide[outcome_var].iloc[0]), 3) \
                      if outcome_var in live_row_wide.columns else None

        # --- Q1: What if the intent adds delta_mbps to the target server? ---
        q1_breach = False
        q1_p50 = q1_p95 = None

        if self_treatment_var in causal_model.graph.nodes:
            try:
                q1_p50, q1_p95 = _run_counterfactual(
                    causal_model, live_row_wide,
                    self_treatment_var, outcome_var,
                    current_self_bw + delta_mbps   # intervention value
                )
                q1_breach = q1_p95 > limit
                print(f"      Q1 (self +{delta_mbps}Mbps) → "
                      f"{metric}: P50={round(q1_p50,2)}{unit}  "
                      f"P95={round(q1_p95,2)}{unit}  "
                      f"{'❌ BREACH' if q1_breach else '✅ ok'}")
            except Exception as e:
                print(f"      [!] Q1 counterfactual failed for '{metric}': {e}")

        # --- Q2: What if switch drops from current flood to baseline? ---
        # This answers: "is upstream congestion causing the breach?"
        # If yes, the resolver must throttle sw1's tenants, not the target.
        q2_breach = False
        q2_p50 = q2_p95 = None
        q2_relief_p50 = q2_relief_p95 = None  # what metric becomes if sw1 is fixed

        if upstream_treatment_var and \
           upstream_treatment_var in causal_model.graph.nodes:
            try:
                # What is metric NOW given sw1 is flooding? (confirm current state)
                q2_p50, q2_p95 = _run_counterfactual(
                    causal_model, live_row_wide,
                    upstream_treatment_var, outcome_var,
                    current_sw_bw   # current flooding value
                )

                # What would metric be if sw1 returned to baseline?
                q2_relief_p50, q2_relief_p95 = _run_counterfactual(
                    causal_model, live_row_wide,
                    upstream_treatment_var, outcome_var,
                    upstream_baseline_bw   # normal switch load
                )

                q2_breach = q2_p95 > limit

                print(f"      Q2 (sw1 flood={round(current_sw_bw,1)}Mbps) → "
                      f"{metric}: P95={round(q2_p95,2)}{unit}  "
                      f"{'❌ upstream causing breach' if q2_breach else '✅ ok'}")
                print(f"      Q2 (sw1 fixed={round(upstream_baseline_bw,1)}Mbps) → "
                      f"{metric}: P95={round(q2_relief_p95,2)}{unit}  "
                      f"{'✅ would fix lag' if q2_relief_p95 <= limit else '⚠️ still over limit'}")

            except Exception as e:
                print(f"      [!] Q2 counterfactual failed for '{metric}': {e}")

        # Determine overall pass/fail and root cause
        passed        = not q1_breach and not q2_breach
        root_cause    = None
        if q2_breach and not q1_breach:
            root_cause = "upstream_switch"   # sw1's tenants are the problem
        elif q1_breach and not q2_breach:
            root_cause = "self"              # the intent overloads the server itself
        elif q1_breach and q2_breach:
            root_cause = "both"

        status_icon = "✅" if passed else "❌"
        print(f"      [{status_icon}] {metric}: now={current_val}{unit}  "
              f"limit={limit}{unit}  "
              f"root_cause={root_cause or 'none'}  "
              f"{'PASS' if passed else 'BREACH'}")

        details[metric] = {
            "current":              current_val,
            "limit":                limit,
            "unit":                 unit,
            "passed":               passed,
            "root_cause":           root_cause,
            "upstream_switch":      upstream_switch_id,
            # Switch BW values for resolver's recovery target calculation
            "_current_sw_bw_mbps":  round(current_sw_bw, 2),
            "_baseline_sw_bw_mbps": round(upstream_baseline_bw, 2),
            # Q1 — self-congestion
            "q1_p50":               round(q1_p50, 3) if q1_p50 is not None else None,
            "q1_p95":               round(q1_p95, 3) if q1_p95 is not None else None,
            "q1_breach":            q1_breach,
            # Q2 — upstream-congestion (what sw1 is doing to this node)
            "q2_p95_flood":         round(q2_p95, 3) if q2_p95 is not None else None,
            "q2_p95_if_fixed":      round(q2_relief_p95, 3) if q2_relief_p95 is not None else None,
            "q2_breach":            q2_breach,
        }

        if not passed and all_passed:
            all_passed = False
            if root_cause == "upstream_switch":
                first_failure = (
                    f"{metric} is breaching SLA ({current_val}{unit} > {limit}{unit}) "
                    f"because '{upstream_switch_id}' is flooding at "
                    f"{round(current_sw_bw,1)} Mbps. "
                    f"Throttling background apps on '{upstream_switch_id}' "
                    f"is predicted to bring {metric} to "
                    f"{round(q2_relief_p95,2)}{unit}."
                )
            elif root_cause == "self":
                first_failure = (
                    f"Adding +{delta_mbps} Mbps to '{target_server_id}' "
                    f"will push {metric} P95 to {round(q1_p95,2)}{unit} "
                    f"(SLA limit: {limit}{unit})."
                )
            else:
                first_failure = (
                    f"{metric} is breaching SLA from both self-congestion "
                    f"and upstream switch flooding."
                )

    if not details:
        return {"is_safe": False,
                "predicted_conflict": "No metrics could be evaluated.",
                "counterfactual_details": {}}

    if all_passed:
        print("\n   ✅ Both counterfactual checks passed.")
        return {"is_safe": True, "predicted_conflict": None,
                "counterfactual_details": details}

    print(f"\n   ❌ {first_failure}")
    return {"is_safe": False, "predicted_conflict": first_failure,
            "counterfactual_details": details}