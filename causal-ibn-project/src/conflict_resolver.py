"""
conflict_resolver.py — Counterfactual-driven conflict resolution.

OLD behaviour (greedy blind):
  - Sort background apps by bandwidth usage, biggest first.
  - Throttle them one by one until we've recovered enough Mbps.
  - No idea whether throttling actually fixes the breaching metric.
  - Result: throttles the wrong apps when the root cause isn't bandwidth.

NEW behaviour (counterfactual-guided):
  1. Read counterfactual_details from inference.py to know WHICH metrics
     are breaching and by HOW MUCH (P95 vs limit).
  2. Decide the resolution STRATEGY per metric:
       - latency breach  → bandwidth is likely the cause → throttle BW
       - jitter breach   → buffer pressure is the cause → throttle BW + flag QoS
       - packet_loss breach → severe congestion → throttle BW aggressively
  3. Calculate the EXACT bandwidth delta needed to push P95 below the SLA,
     using the causal slope from the counterfactual (not a fixed Mbps number).
  4. Rank background apps by their CAUSAL IMPACT on the breaching metric
     (how much their bandwidth contributes to the target node's congestion),
     not just their raw usage size.
  5. Return per-action confidence scores so the frontend can show the user
     WHY each throttle action is being taken.
"""

from __future__ import annotations

MIN_RESERVE_MBPS  = 1.0   # Never throttle a service below this
SAFETY_MARGIN     = 1.15   # Recover 15% more than the bare minimum needed
                            # — buffers against measurement noise


# ---------------------------------------------------------------------------
# Strategy selector — reads counterfactual_details to pick resolution approach
# ---------------------------------------------------------------------------

def _select_strategy(counterfactual_details: dict) -> dict:
    """
    Reads the P95 breach from each metric and returns:
      {
        "primary_metric":  str,    # the worst-breaching metric
        "breach_severity": float,  # how far over the limit P95 is (ratio)
        "needs_bw_reduction": bool,
        "needs_qos_requeue":  bool,
        "target_recovery_mbps": float  # Mbps to free up to fix the breach
      }

    How the causal slope works:
      counterfactual_details tells us:
        p95 AFTER adding delta_mbps  = cf["p95"]
        current value                = cf["current"]
        the delta_mbps was           = stored in intent constraints

      So the causal slope (metric change per Mbps added) is:
        slope = (p95 - current) / delta_mbps

      To bring p95 down to the SLA limit, we need to FREE UP:
        recovery_needed = (p95 - limit) / slope   Mbps
    """
    if not counterfactual_details:
        return {
            "primary_metric":       "latency_ms",
            "breach_severity":      1.0,
            "needs_bw_reduction":   True,
            "needs_qos_requeue":    False,
            "target_recovery_mbps": 0.0,
        }

    # Find the metric with the worst breach ratio
    worst_metric = None
    worst_ratio  = 0.0

    for metric, detail in counterfactual_details.items():
        if not isinstance(detail, dict) or not detail.get("passed") == False:
            continue
        # Use q2_p95_flood (upstream breach) or q1_p95 (self breach) — not legacy "p95"
        p95   = (detail.get("q2_p95_flood") or
                 detail.get("q1_p95") or
                 detail.get("current", 0.0))
        limit = detail.get("limit", 1.0)
        ratio = p95 / limit if limit > 0 else 1.0
        if ratio > worst_ratio:
            worst_ratio  = ratio
            worst_metric = metric

    if not worst_metric:
        # All passed — resolver shouldn't have been called, but handle gracefully
        return {
            "primary_metric":       "latency_ms",
            "breach_severity":      1.0,
            "needs_bw_reduction":   False,
            "needs_qos_requeue":    False,
            "target_recovery_mbps": 0.0,
        }

    detail     = counterfactual_details[worst_metric]
    current    = detail.get("current", 0.0)
    limit      = detail.get("limit", 1.0)
    delta      = detail.get("_delta_mbps", 5.0)
    upstream_bw   = detail.get("_current_sw_bw_mbps", 0.0)
    baseline_bw   = detail.get("_baseline_sw_bw_mbps", 0.0)
    q2_if_fixed   = detail.get("q2_p95_if_fixed")
    root_cause    = detail.get("root_cause", "both")

    # --- Recovery target: how many Mbps to free from the switch ---
    #
    # Primary path: use the CF result directly.
    #   upstream_bw = current switch load (e.g. 141 Mbps)
    #   baseline_bw = median switch load from training data (e.g. 38 Mbps)
    #   The CF already confirmed that baseline_bw produces q2_if_fixed ≤ limit.
    #   So we need to free (upstream_bw - baseline_bw) Mbps from the switch.
    #
    # Fallback: use switch BW empirical ratio if CF guidance is unavailable.

    if upstream_bw > 0 and baseline_bw > 0 and upstream_bw > baseline_bw:
        # CF-guided: cut switch load from flooding level back to baseline
        to_cut          = upstream_bw - baseline_bw
        recovery_needed = to_cut * SAFETY_MARGIN
        print(f"      CF-guided: cut switch {round(upstream_bw,1)} → "
              f"{round(baseline_bw,1)} Mbps  "
              f"(free {round(to_cut,1)} × {SAFETY_MARGIN} = "
              f"{round(recovery_needed,1)} Mbps)")
    elif upstream_bw > 0 and current > limit:
        # Fallback: empirical ratio using switch BW (not gaming's own BW)
        ms_per_mbps     = current / upstream_bw if upstream_bw > 0 else 1.0
        overshoot       = current - limit
        recovery_needed = (overshoot / ms_per_mbps) * SAFETY_MARGIN
        print(f"      Empirical fallback: {round(overshoot,1)}ms overshoot / "
              f"{round(ms_per_mbps,4)} ms/Mbps = "
              f"{round(recovery_needed,1)} Mbps")
    else:
        recovery_needed = max(delta, 10.0) * SAFETY_MARGIN

    MAX_RECOVERABLE = 200.0
    recovery_needed = min(recovery_needed, MAX_RECOVERABLE)

    needs_qos = worst_metric in ("jitter_ms",)

    print(f"\n   -> [Resolver] Strategy:")
    print(f"      Metric  : {worst_metric}  now={round(current,2)}  "
          f"limit={limit}  ratio={round(worst_ratio,2)}x")
    print(f"      Root    : {root_cause}")
    print(f"      Recovery: {round(recovery_needed,2)} Mbps to free from switch")

    return {
        "primary_metric":       worst_metric,
        "breach_severity":      worst_ratio,
        "needs_bw_reduction":   True,
        "needs_qos_requeue":    needs_qos,
        "target_recovery_mbps": max(recovery_needed, 1.0),
    }


# ---------------------------------------------------------------------------
# Causal impact ranker — ranks background apps by their contribution
# ---------------------------------------------------------------------------

def _rank_by_causal_impact(background_tasks: list, target_switch_id: str,
                            telemetry: dict) -> list:
    """
    Instead of sorting purely by usage (greedy), we score each background app
    by its CAUSAL IMPACT on the target switch's congestion.

    Impact score = bandwidth_used_mbps × (buffer_occupancy / 100)

    Rationale: an app using 30 Mbps on a 90% full buffer contributes far more
    to latency/jitter spikes than an app using 50 Mbps on a 20% full buffer.
    The first app is causing active queue pressure; the second has headroom.

    Falls back to raw usage if buffer_occupancy is not in telemetry.
    """
    switch_telemetry = telemetry.get(target_switch_id, {})
    switch_buffer    = switch_telemetry.get("buffer_occupancy", 50.0) / 100.0

    for task in background_tasks:
        node_tel = telemetry.get(task["node_id"], {})
        bw       = task["usage"]
        buf      = node_tel.get("buffer_occupancy", 50.0) / 100.0

        # Weight by how much this app is contributing to shared buffer pressure
        task["causal_impact"] = round(bw * (buf + switch_buffer) / 2, 3)
        task["buffer_pct"]    = round(buf * 100, 1)

    background_tasks.sort(key=lambda x: x["causal_impact"], reverse=True)
    return background_tasks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_bandwidth_conflict(intent: dict, network_state: dict,
                                counterfactual_details: dict = None) -> dict:
    """
    Counterfactual-guided conflict resolution.

    New parameter: counterfactual_details (from inference.py)
      Shape: { "latency_ms": {"p95": 210, "limit": 30, "current": 12, ...}, ... }

    The resolver uses this to:
      1. Know exactly which metric is breaching and by how much.
      2. Calculate the precise Mbps recovery target from the causal slope
         (instead of blindly using the intent's bandwidth_guarantee).
      3. Rank apps by causal buffer impact, not raw bandwidth size.
      4. Attach a confidence score to each action.
    """
    if counterfactual_details is None:
        counterfactual_details = {}

    print("\n   -> [Conflict Resolver] Starting counterfactual-guided resolution...")

    requirements = intent.get("requirements", {})
    target_app   = requirements.get("application", "")
    constraints  = requirements.get("constraints", {})
    bw_string    = constraints.get("bandwidth_guarantee", "0Mbps")

    try:
        delta_mbps = float(
            str(bw_string).replace("Mbps", "").replace("mbps", "").strip()
        )
    except (ValueError, AttributeError):
        delta_mbps = 0.0

    # Inject delta and current BW into counterfactual_details for strategy selector
    # We need the target node's actual current bandwidth to compute a meaningful slope
    target_server_id_pre = next(
        (n["id"] for n in network_state.get("topology", {}).get("nodes", [])
         if n.get("hosted_service", "")!= None and n.get("hosted_service", "").lower() == target_app.lower()),
        None
    )
    current_bw_mbps = (
        network_state.get("telemetry", {})
                     .get(target_server_id_pre, {})
                     .get("bandwidth_used_mbps", 10.0)
        if target_server_id_pre else 10.0
    )

    for detail in counterfactual_details.values():
        if isinstance(detail, dict):
            detail["_delta_mbps"]          = delta_mbps
            detail["_current_bw_mbps"]     = current_bw_mbps
            upstream_sw = detail.get("upstream_switch")
            if upstream_sw:
                sw_telem = network_state.get("telemetry", {}).get(upstream_sw, {})
                detail["_current_sw_bw_mbps"]  = sw_telem.get("bandwidth_used_mbps", current_bw_mbps)
                # Baseline injected by inference.py via _baseline_sw_bw_mbps
                # (training data median of sw1 bandwidth — already in the detail dict)

    topology  = network_state.get("topology", {})
    telemetry = network_state.get("telemetry", {})

    # --- 1. Find target node ---
    target_server_id = None
    for node in topology.get("nodes", []):
        if node.get("hosted_service", "")!= None and node.get("hosted_service", "").lower() == target_app.lower():
            target_server_id = node["id"]
            break

    if not target_server_id:
        return {
            "resolution_found": False, "is_safe": False,
            "reason": f"Target app '{target_app}' not found in topology."
        }

    # --- 2. Find the switch to throttle ---
    # If root_cause is upstream_switch, we throttle THAT switch's tenants.
    # If root_cause is self, we throttle the target server's own switch tenants.
    # The upstream_switch field in counterfactual_details tells us which switch.
    upstream_switch_from_cf = None
    for detail in counterfactual_details.values():
        if isinstance(detail, dict) and detail.get("root_cause") == "upstream_switch":
            upstream_switch_from_cf = detail.get("upstream_switch")
            break

    target_switch_id = upstream_switch_from_cf  # prefer CF-identified switch

    if not target_switch_id:
        # Fallback: find switch from topology links
        for link in topology.get("links", []):
            src, tgt = link.get("source"), link.get("target")
            if tgt == target_server_id:
                target_switch_id = src; break
            if src == target_server_id:
                target_switch_id = tgt; break

    if not target_switch_id:
        return {
            "resolution_found": False, "is_safe": False,
            "reason": f"No switch found for '{target_server_id}'."
        }

    # --- 3. Collect background apps on the same switch ---
    seen_servers     = set()
    background_tasks = []

    for link in topology.get("links", []):
        src, tgt = link.get("source"), link.get("target")
        other    = None
        if src == target_switch_id and tgt != target_server_id:
            other = tgt
        elif tgt == target_switch_id and src != target_server_id:
            other = src

        if not other or other in seen_servers:
            continue
        seen_servers.add(other)

        server_node = next(
            (n for n in topology.get("nodes", []) if n["id"] == other), None
        )
        if not server_node:
            continue

        app_name = server_node.get("hosted_service", other)
        usage    = telemetry.get(other, {}).get("bandwidth_used_mbps", 0.0)

        if usage > MIN_RESERVE_MBPS:
            background_tasks.append({
                "app":     app_name,
                "node_id": other,
                "usage":   usage,
            })

    if not background_tasks:
        return {
            "resolution_found": False, "is_safe": False,
            "reason": "No background apps found to throttle on this switch."
        }

    # --- 4. Select strategy from counterfactual details ---
    strategy = _select_strategy(counterfactual_details)

    # --- 5. Rank by causal impact (not raw usage) ---
    background_tasks = _rank_by_causal_impact(
        background_tasks, target_switch_id, telemetry
    )

    print(f"\n   -> Background apps ranked by causal impact:")
    for t in background_tasks:
        app_label = t['app'] or t['node_id']   # switches have no hosted_service
        print(f"      {app_label:20s}  bw={t['usage']}Mbps  "
              f"buffer={t.get('buffer_pct','-')}%  "
              f"impact={t['causal_impact']}")

    # --- 6. Throttle loop — driven by causal recovery target ---
    recovery_target   = strategy["target_recovery_mbps"]
    recovered         = 0.0
    actions_to_take   = []

    print(f"\n   -> Recovery target: {round(recovery_target, 2)} Mbps "
          f"(to fix {strategy['primary_metric']} breach)")

    for task in background_tasks:
        if recovered >= recovery_target:
            break

        available = task["usage"] - MIN_RESERVE_MBPS
        if available <= 0:
            continue

        take      = min(available, recovery_target - recovered)
        new_limit = round(task["usage"] - take, 2)
        recovered += take

        app_label  = task["app"] or task["node_id"]   # None-safe

        # Confidence: how confident are we this throttle will help?
        buf_pct    = task.get("buffer_pct", 50.0)
        confidence = round(min(0.99, 0.5 + (buf_pct / 200.0) +
                              (task["causal_impact"] /
                               max(t["causal_impact"] for t in background_tasks)
                               * 0.3)), 2)

        reason = (
            f"Causal analysis: '{app_label}' is contributing "
            f"{round(task['causal_impact'],2)} units of buffer pressure on "
            f"{target_switch_id}, which is propagating to "
            f"{strategy['primary_metric']} spikes on '{target_app}'."
        )

        print(f"      -> Throttling '{app_label}': "
              f"{task['usage']} → {new_limit} Mbps  "
              f"(confidence: {confidence})")

        actions_to_take.append({
            "target_service":           app_label,
            "node_id":                  task["node_id"],
            "new_bandwidth_limit_mbps": new_limit,
            "recovered_mbps":           round(take, 2),
            "confidence":               confidence,
            "reason":                   reason,
        })

    # --- 7. QoS flag for jitter breaches ---
    qos_recommendation = None
    if strategy["needs_qos_requeue"]:
        qos_recommendation = {
            "action":  "requeue",
            "target":  target_server_id,
            "reason":  (
                "Jitter breach detected. Bandwidth throttling alone may not fix "
                "jitter — recommend also applying a WFQ/DSCP QoS queue policy "
                "to prioritise gaming traffic at the switch level."
            )
        }
        print(f"\n   -> [QoS Flag] Jitter breach — WFQ/DSCP requeue recommended.")

    # --- 8. Outcome ---
    if recovered >= recovery_target:
        print(f"\n   ✅ Recovered {round(recovered,2)} Mbps across "
              f"{len(actions_to_take)} app(s). "
              f"Target: {round(recovery_target,2)} Mbps.")
        return {
            "resolution_found":    True,
            "is_safe":             True,
            "strategy":            strategy,
            "actions":             actions_to_take,
            "recovered_mbps":      round(recovered, 2),
            "recovery_target_mbps": round(recovery_target, 2),
            "qos_recommendation":  qos_recommendation,
        }
    else:
        print(f"\n   ❌ Only recovered {round(recovered,2)} Mbps "
              f"of {round(recovery_target,2)} Mbps needed.")
        return {
            "resolution_found": False,
            "is_safe":          False,
            "strategy":         strategy,
            "actions":          actions_to_take,   # partial — still useful for ONOS
            "recovered_mbps":   round(recovered, 2),
            "reason": (
                f"Could only recover {round(recovered,2)} Mbps of "
                f"{round(recovery_target,2)} Mbps needed to fix "
                f"{strategy['primary_metric']} breach."
            ),
            "qos_recommendation": qos_recommendation,
        }