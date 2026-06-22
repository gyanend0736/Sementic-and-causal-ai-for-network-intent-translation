"""
orchestrator.py — Main control hub.

The two TODO comments that existed before are now implemented:
  "# TODO: Send API request to ONOS here to lock in QoS queue"
  → _actuate_clean_deployment()

  "# TODO: Send API request to ONOS here using the new limits"
  → _actuate_resolution()

ONOS actuation is only attempted when ONOS_ENABLED=true.
In static/dev mode the decisions are logged but no rules are pushed.
"""

import os
from semantic_engine import parse_intent
from discovery import get_current_network_state
from inference import evaluate_intent_safety
from conflict_resolver import resolve_bandwidth_conflict

ONOS_ENABLED = os.environ.get("ONOS_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_network_intent(raw_intent_text, source_ip="192.168.1.10"):
    print(f"\n{'='*52}")
    print(f"⚙️   ORCHESTRATOR: Processing Intent from {source_ip}")
    print(f"{'='*52}")

    # Step 0 — Semantic parsing
    if isinstance(raw_intent_text, str):
        print("\n[0] Running Semantic Engine...")
        intent_json = parse_intent(raw_intent_text)
    else:
        intent_json = raw_intent_text

    if "error" in intent_json:
        print(f"   [!] Semantic Error: {intent_json['error']}")
        return {"error": intent_json["error"], "is_safe": False, "status": "error"}

    app_name = intent_json.get("requirements", {}).get("application", "Unknown")
    print(f"   -> Target application: {app_name.upper()}")

    # Step 1 — Discover live network state (ONOS or CSV)
    print("\n[1] Discovering network state...")
    current_state = get_current_network_state()
    if current_state is None:
        return {
            "error":   "Critical failure: could not load network state.",
            "is_safe": False, "status": "error"
        }

    data_source = current_state.get("source", "unknown")
    print(f"   -> Data source: {data_source}")

    # Step 2 — Counterfactual safety check
    print("\n[2] Running Counterfactual Causal Inference Engine...")
    safety_report = evaluate_intent_safety(intent_json, current_state)
    cf_details    = safety_report.get("counterfactual_details", {})

    # -----------------------------------------------------------------------
    # Step 3a — Safe: push a priority rule so the intent gets guaranteed QoS
    # -----------------------------------------------------------------------
    if safety_report.get("is_safe"):
        print("\n   ✅ [SUCCESS] Intent is causally safe.")

        onos_result = _actuate_clean_deployment(
            intent_json, current_state
        )

        return {
            "status":                 "deployed_clean",
            "is_safe":                True,
            "conflict_detected":      False,
            "target_service":         app_name,
            "data_source":            data_source,
            "counterfactual_details": cf_details,
            "onos_actuation":         onos_result,
            "message":                "Deployment successful — all SLA metrics passed."
        }

    # -----------------------------------------------------------------------
    # Step 3b — Unsafe: run conflict resolver then push throttle rules
    # -----------------------------------------------------------------------
    breaching = [m for m, d in cf_details.items()
                 if isinstance(d, dict) and not d.get("passed", True)]
    print(f"\n   ⚠️  [WARNING] Conflict: {safety_report.get('predicted_conflict')}")
    print(f"   -> Breaching metrics: {breaching}")
    print("   -> Routing to Counterfactual-Guided Conflict Resolver...")

    resolution = resolve_bandwidth_conflict(
        intent_json,
        current_state,
        counterfactual_details=cf_details
    )

    if resolution.get("resolution_found"):
        print("\n   ✅ [RESOLVED] Conflict mitigated.")

        onos_result = _actuate_resolution(
            intent_json, current_state, resolution
        )

        return {
            "status":            "deployed_after_resolution",
            "is_safe":           True,
            "conflict_detected": True,
            "target_service":    app_name,
            "data_source":       data_source,
            "causal_analysis": {
                "safety_report":          safety_report,
                "resolution":             resolution,
                "counterfactual_details": cf_details,
            },
            "onos_actuation": onos_result,
            "message":        _build_resolution_message(resolution, app_name),
        }

    # -----------------------------------------------------------------------
    # Step 3c — Cannot resolve: block and explain
    # -----------------------------------------------------------------------
    print("\n   ❌ [FAILED] Network cannot support this intent. Blocking.")
    return {
        "status":            "blocked",
        "is_safe":           False,
        "conflict_detected": True,
        "target_service":    app_name,
        "data_source":       data_source,
        "error":             "Deployment blocked to prevent cascading failure.",
        "reason":            resolution.get("reason", "Unknown"),
        "causal_analysis": {
            "safety_report":          safety_report,
            "resolution":             resolution,
            "counterfactual_details": cf_details,
        },
    }


# ---------------------------------------------------------------------------
# ONOS actuation helpers — these replace the TODO comments
# ---------------------------------------------------------------------------

def _actuate_clean_deployment(intent: dict, network_state: dict) -> dict:
    """
    Called when the intent is safe with no conflict.
    Pushes a DSCP priority rule so the target service gets priority queuing
    even before any congestion occurs — proactive QoS.
    """
    if not ONOS_ENABLED:
        print("   [ONOS] Skipped — ONOS_ENABLED=false (dev mode)")
        return {"skipped": True, "reason": "ONOS_ENABLED=false"}

    from onos_client import push_priority_rule, remove_ibn_rules, verify_rule_applied
    from conflict_resolver import MIN_RESERVE_MBPS  # reuse topology helpers

    target_server_id = _find_server_id(intent, network_state)
    target_switch_id = _find_switch_id(target_server_id, network_state)

    if not target_server_id or not target_switch_id:
        return {"pushed": False, "reason": "Could not locate node in topology"}

    # DSCP 46 = Expedited Forwarding (real-time priority)
    pushed = push_priority_rule(
        switch_id=target_switch_id,
        target_node_id=target_server_id,
        dscp_class=46
    )

    return {
        "pushed":    pushed,
        "action":    "priority_rule",
        "switch":    target_switch_id,
        "target":    target_server_id,
        "dscp":      46,
    }


def _actuate_resolution(intent: dict, network_state: dict,
                         resolution: dict) -> dict:
    """
    Called when the conflict resolver found a fix.
    Pushes one bandwidth-limit flow rule per throttle action.
    Also pushes a QoS requeue rule if the resolver flagged a jitter breach.
    """
    if not ONOS_ENABLED:
        print("   [ONOS] Skipped — ONOS_ENABLED=false (dev mode)")
        return {"skipped": True, "reason": "ONOS_ENABLED=false"}

    from onos_client import push_throttle_rule, push_priority_rule, push_queue_policy, verify_rule_applied

    target_switch_id = _find_switch_id(
        _find_server_id(intent, network_state), network_state
    )
    if not target_switch_id:
        return {"pushed": False, "reason": "Switch not found"}

    results      = []
    push_success = True
    topo_nodes   = network_state["topology"].get("nodes", [])

    # 1. Remove any stale IBN rules from a previous intent on this switch
    from onos_client import remove_ibn_rules
    remove_ibn_rules(switch_id=target_switch_id, topology_nodes=topo_nodes)

    # 2. Throttle each background app the resolver identified
    for action in resolution.get("actions", []):
        node_id    = action["node_id"]
        limit_mbps = action["new_bandwidth_limit_mbps"]

        ok = push_throttle_rule(
            switch_id=target_switch_id,
            target_node_id=node_id,
            limit_mbps=limit_mbps,
            topology_nodes=topo_nodes
        )

        # Verify the rule actually landed in the dataplane
        if ok:
            from onos_client import _build_lookup_maps
            _, ip_map = _build_lookup_maps(topo_nodes)
            target_ip = ip_map.get(node_id)
            if target_ip:
                verified = verify_rule_applied(target_switch_id, target_ip, topo_nodes)
                ok = verified

        results.append({
            "node_id":    node_id,
            "app":        action["target_service"],
            "limit_mbps": limit_mbps,
            "pushed":     ok,
            "confidence": action.get("confidence"),
        })
        if not ok:
            push_success = False

    # 3. Priority rule for the target service itself (DSCP EF)
    target_server_id = _find_server_id(intent, network_state)
    push_priority_rule(
        switch_id=target_switch_id,
        target_node_id=target_server_id,
        dscp_class=46,
        topology_nodes=topo_nodes
    )

    # 4. WFQ queue policy for jitter breach (replaces simple DSCP requeue)
    qos_rec = resolution.get("qos_recommendation")
    if qos_rec:
        print(f"\n   [ONOS] Applying WFQ queue policy for jitter breach...")
        ok = push_queue_policy(
            switch_id=target_switch_id,
            target_node_id=target_server_id,
            min_rate_mbps=5.0,   # guarantee at least 5 Mbps to this service
            max_rate_mbps=50.0,  # cap at 50 Mbps (adjust per SLA)
            queue_id=1,
            topology_nodes=topo_nodes
        )
        results.append({
            "action": "wfq_queue_policy",
            "pushed": ok,
            "reason": qos_rec["reason"]
        })

    return {
        "pushed":        push_success,
        "switch":        target_switch_id,
        "rules_applied": results,
    }


# ---------------------------------------------------------------------------
# Topology helpers (shared by both actuation functions)
# ---------------------------------------------------------------------------

def _find_server_id(intent: dict, network_state: dict) -> str | None:
    target = intent.get("requirements", {}).get("application", "").lower()
    for node in network_state["topology"].get("nodes", []):
        if node.get("hosted_service", "").lower() == target:
            return node["id"]
    return None


def _find_switch_id(server_id: str, network_state: dict) -> str | None:
    if not server_id:
        return None
    for link in network_state["topology"].get("links", []):
        src, tgt = link.get("source"), link.get("target")
        if tgt == server_id:
            return src
        if src == server_id:
            return tgt
    return None


def _build_resolution_message(resolution: dict, app_name: str) -> str:
    strategy  = resolution.get("strategy", {})
    actions   = resolution.get("actions", [])
    metric    = strategy.get("primary_metric", "network metric")
    recovered = resolution.get("recovered_mbps", 0)
    throttled = ", ".join(a["target_service"] for a in actions)

    msg = (f"Deployment successful. Fixed {metric} breach for '{app_name}' "
           f"by recovering {recovered} Mbps — throttled: {throttled}.")

    qos = resolution.get("qos_recommendation")
    if qos:
        msg += f" QoS requeue also applied for jitter."
    return msg