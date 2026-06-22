"""
onos_client.py  —  ONOS REST API client (full get + send cycle)

RESPONSIBILITIES
────────────────
GET side  (data IN from ONOS → your causal AI):
  1. get_topology()           live devices + links
  2. get_live_telemetry()     bandwidth, packet-loss, active-flows per device
                              uses delta between two port-stat snapshots so
                              bandwidth is Mbps RIGHT NOW, not total since boot
  3. get_latency_probe()      per-hop latency via ONOS SR latency probes
                              (falls back to 0 if SR app not installed)

SEND side (decisions OUT from causal AI → ONOS):
  4. push_throttle_rule()     meter + match rule: rate-limit a specific service
  5. push_priority_rule()     DSCP EF mark: give a service highest queue priority
  6. push_queue_policy()      WFQ queue for jitter-sensitive services
  7. remove_ibn_rules()       clean up all rules this system installed
  8. verify_rule_applied()    read back flow table to confirm rule landed

UTILS:
  9. is_onos_reachable()      startup health check
  10._load_ip_map()           reads topology_metadata.json for node→IP mapping

ENV VARS (set before running):
  ONOS_HOST       controller IP/hostname  (default: localhost)
  ONOS_PORT       REST port               (default: 8181)
  ONOS_USER       username                (default: onos)
  ONOS_PASSWORD   password                (default: rocks)
  ONOS_TIMEOUT    HTTP timeout seconds    (default: 5)
  ONOS_ENABLED    set to "true" to use live ONOS (default: false)
"""

from __future__ import annotations
import os, json, time, requests
from requests.auth import HTTPBasicAuth

# ── Config ──────────────────────────────────────────────────────────────────
ONOS_HOST     = os.environ.get("ONOS_HOST",     "localhost")
ONOS_PORT     = os.environ.get("ONOS_PORT",     "8181")
ONOS_USER     = os.environ.get("ONOS_USER",     "onos")
ONOS_PASSWORD = os.environ.get("ONOS_PASSWORD", "rocks")
ONOS_TIMEOUT  = int(os.environ.get("ONOS_TIMEOUT", "5"))

BASE_URL = f"http://{ONOS_HOST}:{ONOS_PORT}/onos/v1"
AUTH     = HTTPBasicAuth(ONOS_USER, ONOS_PASSWORD)
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

# IBN app label — all flow rules we push carry this so remove_ibn_rules()
# can delete exactly our rules without touching others
IBN_APP_ID = "org.onosproject.ngsdn-tutorial"

# Path to topology overlay (so we can resolve node_id → ONOS device ID → IP)
_TOPO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "topology_metadata.json"
)

# ── Port-stat snapshot store (for delta-based bandwidth calculation) ─────────
_last_port_stats: dict = {}   # { device_id: { port: {bytes, pkts, ts} } }


# ════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HTTP HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _get(path: str) -> dict | None:
    try:
        r = requests.get(f"{BASE_URL}{path}",
                         auth=AUTH, headers=HEADERS, timeout=ONOS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"   [ONOS] ❌ Cannot reach ONOS at {BASE_URL}")
        return None
    except requests.exceptions.Timeout:
        print(f"   [ONOS] ❌ Timeout: GET {path}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"   [ONOS] ❌ HTTP {e.response.status_code}: GET {path}")
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{BASE_URL}{path}",
                          auth=AUTH, headers=HEADERS,
                          data=json.dumps(payload), timeout=ONOS_TIMEOUT)
        r.raise_for_status()
        return r.json() if r.content else {"status": "ok"}
    except requests.exceptions.ConnectionError:
        print(f"   [ONOS] ❌ Cannot reach ONOS at {BASE_URL}")
        return None
    except requests.exceptions.Timeout:
        print(f"   [ONOS] ❌ Timeout: POST {path}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"   [ONOS] ❌ HTTP {e.response.status_code}: POST {path} — {e.response.text[:200]}")
        return None


def _delete(path: str) -> bool:
    try:
        r = requests.delete(f"{BASE_URL}{path}",
                            auth=AUTH, headers=HEADERS, timeout=ONOS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"   [ONOS] ❌ DELETE {path} failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# GET SIDE — pull live data from ONOS into the causal AI
# ════════════════════════════════════════════════════════════════════════════

def get_topology() -> dict | None:
    """
    GET /devices  +  GET /links

    Returns the same shape as topology_metadata.json so discovery.py
    needs zero changes:
      { "nodes": [...], "links": [...] }

    Merges with topology_metadata.json overlay to inject:
      - hosted_service  (ONOS doesn't know which app runs on which server)
      - ip              (needed for flow rule matching)
      - id alias        (your short "sw1" / "srv_gaming" names)
    """
    devices_raw = _get("/devices")
    links_raw   = _get("/links")

    if not devices_raw or not links_raw:
        return None

    # Load overlay: maps ONOS device IDs to your node metadata
    overlay = _load_ip_map()                         # onos_id → node dict
    onos_to_alias = {v["onos_id"]: k
                     for k, v in overlay.items() if "onos_id" in v}

    nodes = []
    for d in devices_raw.get("devices", []):
        onos_id = d["id"]                            # e.g. of:0000000000000001
        alias   = onos_to_alias.get(onos_id, onos_id)
        meta    = overlay.get(alias, {})

        nodes.append({
            "id":             alias,                 # short name your code uses
            "onos_id":        onos_id,               # ONOS native ID for API calls
            "type":           d.get("type", "SWITCH").lower(),
            "online":         d.get("available", False),
            "hosted_service": meta.get("hosted_service"),
            "ip":             meta.get("ip"),
        })

    links = []
    for lnk in links_raw.get("links", []):
        src_onos = lnk["src"]["device"]
        tgt_onos = lnk["dst"]["device"]
        links.append({
            "source":   onos_to_alias.get(src_onos, src_onos),
            "target":   onos_to_alias.get(tgt_onos, tgt_onos),
            "src_port": lnk["src"]["port"],
            "dst_port": lnk["dst"]["port"],
            "state":    lnk.get("state", "ACTIVE"),
        })

    print(f"   [ONOS] ✅ Topology: {len(nodes)} devices, {len(links)} links")
    return {"nodes": nodes, "links": links}


def get_live_telemetry(node_ids: list[str],
                       topology_nodes: list[dict] = None) -> dict:
    """
    GET /statistics/ports/{deviceId}  for each node
    GET /flows/{deviceId}             for flow count

    KEY FIX vs old code:
      Old: total bytes since boot ÷ 1e6 → meaningless huge number
      New: delta bytes between two snapshots ÷ elapsed_seconds → real Mbps

    First call after startup: stores snapshot, returns 0 Mbps (no delta yet).
    Second call onward: computes real delta → real Mbps.

    topology_nodes is the nodes list from get_topology() — used to look up
    ONOS device IDs (of:...) from short alias names.
    """
    global _last_port_stats
    telemetry  = {}
    now        = time.time()

    # Build alias → onos_id map from the topology nodes list
    alias_to_onos = {}
    if topology_nodes:
        for n in topology_nodes:
            alias_to_onos[n["id"]] = n.get("onos_id", n["id"])

    for node_id in node_ids:
        onos_id   = alias_to_onos.get(node_id, node_id)
        stats_raw = _get(f"/statistics/ports/{onos_id}")
        flows_raw = _get(f"/flows/{onos_id}")

        if stats_raw is None:
            print(f"   [ONOS] ⚠️  No stats for {node_id} ({onos_id})")
            continue

        # ── Collect current port counters ──────────────────────────────────
        current_ports: dict = {}
        for entry in stats_raw.get("statistics", []):
            for port in entry.get("ports", []):
                pid = str(port.get("port", "0"))
                current_ports[pid] = {
                    "bytes_rx":  port.get("bytesReceived", 0),
                    "bytes_tx":  port.get("bytesSent", 0),
                    "pkts_tx":   port.get("packetsSent", 0),
                    "dropped":   port.get("packetsTxDropped", 0) +
                                 port.get("packetsRxDropped", 0),
                    "ts":        now,
                }

        # ── Delta calculation ──────────────────────────────────────────────
        prev = _last_port_stats.get(onos_id, {})
        total_bw_bps   = 0.0
        total_dropped  = 0
        total_packets  = 0

        for pid, cur in current_ports.items():
            p = prev.get(pid)
            if p and (cur["ts"] - p["ts"]) > 0:
                dt            = cur["ts"] - p["ts"]
                delta_bytes   = max(0, cur["bytes_tx"] - p["bytes_tx"])
                delta_dropped = max(0, cur["dropped"]  - p["dropped"])
                delta_pkts    = max(0, cur["pkts_tx"]  - p["pkts_tx"])

                total_bw_bps  += (delta_bytes * 8) / dt   # bits/sec this port
                total_dropped += delta_dropped
                total_packets += delta_pkts

        # Store snapshot for next call
        _last_port_stats[onos_id] = current_ports

        bandwidth_mbps = round(total_bw_bps / 1e6, 3)
        loss_pct       = round((total_dropped / max(total_packets, 1)) * 100, 4)
        active_flows   = len(flows_raw.get("flows", [])) if flows_raw else 0

        telemetry[node_id] = {
            "bandwidth_used_mbps":     bandwidth_mbps,
            "packet_loss_percent":     loss_pct,
            "active_flows":            float(active_flows),
            # Latency + jitter come from get_latency_probe() or your CSV
            # They are filled in by discovery.py after this call returns
            "latency_ms":              0.0,
            "jitter_ms":               0.0,
            "cpu_utilization_percent": 0.0,
            "buffer_occupancy":        0.0,
        }

        print(f"   [ONOS] 📊 {node_id}: "
              f"bw={bandwidth_mbps} Mbps  "
              f"loss={loss_pct}%  "
              f"flows={active_flows}")

    return telemetry


def get_latency_probe(src_device: str, dst_device: str,
                      topology_nodes: list[dict] = None) -> float:
    """
    GET /latency/{src}/{dst}  (ONOS Segment Routing latency probe app)

    Returns one-way latency in ms between src and dst devices.
    Returns 0.0 if the SR latency app is not installed or path not found.

    Activate in ONOS CLI:
      app activate org.onosproject.segmentrouting

    If not available, latency still comes from your telemetry CSV fallback
    in discovery.py — this is additive, not a hard dependency.
    """
    alias_to_onos = {}
    if topology_nodes:
        for n in topology_nodes:
            alias_to_onos[n["id"]] = n.get("onos_id", n["id"])

    src_onos = alias_to_onos.get(src_device, src_device)
    dst_onos = alias_to_onos.get(dst_device, dst_device)

    result = _get(f"/latency/{src_onos}/{dst_onos}")
    if not result:
        return 0.0

    latency_ns = result.get("latency", 0)
    latency_ms = round(latency_ns / 1e6, 3)
    print(f"   [ONOS] 🔬 Latency {src_device} → {dst_device}: {latency_ms} ms")
    return latency_ms


# ════════════════════════════════════════════════════════════════════════════
# SEND SIDE — push causal AI decisions back to ONOS as flow/meter rules
# ════════════════════════════════════════════════════════════════════════════

def push_throttle_rule(switch_id:      str,
                       target_node_id: str,
                       limit_mbps:     float,
                       priority:       int = 40000,
                       topology_nodes: list[dict] = None) -> bool:
    """
    Installs a bandwidth-limit (meter) rule on switch_id for traffic
    destined to target_node_id.

    Full flow:
      1. Create (or reuse) a ONOS meter at limit_mbps on switch_id
      2. POST a flow rule that:
           matches:  ETH_TYPE=IPv4 + IPV4_DST=target_node IP
           actions:  apply meter → OUTPUT NORMAL

    This is what the conflict resolver calls for each throttle action.
    E.g. "limit backup_sync to 29 Mbps" becomes one call here.

    Returns True if ONOS accepted both the meter and the flow rule.
    """
    alias_to_onos, ip_map = _build_lookup_maps(topology_nodes)

    onos_switch_id = alias_to_onos.get(switch_id, switch_id)
    target_ip      = ip_map.get(target_node_id)

    if not target_ip:
        print(f"   [ONOS] ❌ No IP found for '{target_node_id}'. "
              f"Add 'ip' field to topology_metadata.json.")
        return False

    # Step 1: get or create a meter at the requested rate
    meter_id = _get_or_create_meter(onos_switch_id, limit_mbps)
    if not meter_id:
        print(f"   [ONOS] ❌ Could not create meter for {target_node_id}")
        return False

    # Step 2: install the flow rule
    flow = {
        "appId":       IBN_APP_ID,
        "priority":    priority,
        "timeout":     0,
        "isPermanent": True,
        "deviceId":    onos_switch_id,
        "treatment": {
            "instructions": [
                {"type": "METER", "meterId": str(meter_id)},
                {"type": "OUTPUT", "port":   "NORMAL"},
            ]
        },
        "selector": {
            "criteria": [
                {"type": "ETH_TYPE", "ethType": "0x0800"},
                {"type": "IPV4_DST", "ip":      target_ip},
            ]
        }
    }

    result = _post(f"/flows/{onos_switch_id}", {"flows": [flow]})
    if result:
        print(f"   [ONOS] ✅ Throttle rule: {switch_id} → "
              f"{target_node_id} ({target_ip}) capped at {limit_mbps} Mbps  "
              f"[meter {meter_id}]")
        return True

    print(f"   [ONOS] ❌ Failed to push throttle rule for {target_node_id}")
    return False


def push_priority_rule(switch_id:      str,
                       target_node_id: str,
                       dscp_class:     int = 46,
                       priority:       int = 50000,
                       topology_nodes: list[dict] = None) -> bool:
    """
    Marks traffic destined for target_node_id with DSCP dscp_class.

    DSCP 46 = Expedited Forwarding (EF) — highest real-time priority.
    Used for gaming, VoIP. Switches must support DSCP-based queuing.

    This is what the orchestrator calls for a clean (safe) deployment —
    proactive priority before any congestion occurs.
    """
    alias_to_onos, ip_map = _build_lookup_maps(topology_nodes)

    onos_switch_id = alias_to_onos.get(switch_id, switch_id)
    target_ip      = ip_map.get(target_node_id)

    if not target_ip:
        print(f"   [ONOS] ❌ No IP for '{target_node_id}'")
        return False

    flow = {
        "appId":       IBN_APP_ID,
        "priority":    priority,
        "timeout":     0,
        "isPermanent": True,
        "deviceId":    onos_switch_id,
        "treatment": {
            "instructions": [
                {"type":    "L3MODIFICATION",
                 "subtype": "IP_DSCP",
                 "ipDscp":  dscp_class},
                {"type": "OUTPUT", "port": "NORMAL"},
            ]
        },
        "selector": {
            "criteria": [
                {"type": "ETH_TYPE", "ethType": "0x0800"},
                {"type": "IPV4_DST", "ip":      target_ip},
            ]
        }
    }

    result = _post(f"/flows/{onos_switch_id}", {"flows": [flow]})
    if result:
        print(f"   [ONOS] ✅ Priority rule: DSCP {dscp_class} for "
              f"{target_node_id} on {switch_id}")
        return True
    return False


def push_queue_policy(switch_id:      str,
                      target_node_id: str,
                      min_rate_mbps:  float,
                      max_rate_mbps:  float,
                      queue_id:       int = 1,
                      topology_nodes: list[dict] = None) -> bool:
    """
    Installs a WFQ (Weighted Fair Queue) policy for jitter-sensitive services.

    Called by the orchestrator when the conflict resolver flags needs_qos_requeue
    (i.e. jitter breach). Guarantees a minimum bandwidth slice AND a maximum
    so other services are not completely starved.

    Requires ONOS Queue app:
      app activate org.onosproject.queue

    Flow:
      1. POST /qos/queues  — create queue with min/max rates
      2. POST /flows       — match target IP → send to that queue
    """
    alias_to_onos, ip_map = _build_lookup_maps(topology_nodes)

    onos_switch_id = alias_to_onos.get(switch_id, switch_id)
    target_ip      = ip_map.get(target_node_id)

    if not target_ip:
        print(f"   [ONOS] ❌ No IP for '{target_node_id}'")
        return False

    # Step 1: create a WFQ queue
    queue_payload = {
        "deviceId": onos_switch_id,
        "portNumber": "0",          # applies to all ports on this device
        "queues": [{
            "queueId":    queue_id,
            "type":       "MIN_MAX",
            "minRate":    int(min_rate_mbps * 1e6),   # bps
            "maxRate":    int(max_rate_mbps * 1e6),   # bps
            "burst":      True,
            "priority":   7,                          # highest WFQ weight
        }]
    }
    q_result = _post("/qos/queues", queue_payload)
    if not q_result:
        print(f"   [ONOS] ⚠️  Queue creation failed — falling back to DSCP only")
        return push_priority_rule(switch_id, target_node_id, 46, 60000,
                                  topology_nodes)

    # Step 2: flow rule that maps target IP → this queue
    flow = {
        "appId":       IBN_APP_ID,
        "priority":    55000,
        "timeout":     0,
        "isPermanent": True,
        "deviceId":    onos_switch_id,
        "treatment": {
            "instructions": [
                {"type": "QUEUE", "queueId": str(queue_id)},
                {"type": "OUTPUT", "port": "NORMAL"},
            ]
        },
        "selector": {
            "criteria": [
                {"type": "ETH_TYPE", "ethType": "0x0800"},
                {"type": "IPV4_DST", "ip": target_ip},
            ]
        }
    }

    result = _post(f"/flows/{onos_switch_id}", {"flows": [flow]})
    if result:
        print(f"   [ONOS] ✅ WFQ queue policy: {target_node_id} "
              f"[{min_rate_mbps}–{max_rate_mbps} Mbps, queue {queue_id}]")
        return True

    print(f"   [ONOS] ❌ Queue flow rule failed for {target_node_id}")
    return False


def remove_ibn_rules(switch_id: str = None,
                     topology_nodes: list[dict] = None) -> bool:
    """
    Removes ALL flow rules this system installed (tagged with IBN_APP_ID).

    If switch_id is given, removes only rules on that switch.
    If switch_id is None, removes all IBN rules across all devices.

    Call this after the conflict is resolved or when a new intent
    supersedes the previous one.
    """
    if switch_id:
        alias_to_onos, _ = _build_lookup_maps(topology_nodes)
        onos_id = alias_to_onos.get(switch_id, switch_id)
        ok = _delete(f"/flows/{onos_id}/app/{IBN_APP_ID}")
        if ok:
            print(f"   [ONOS] ✅ Cleared IBN rules on {switch_id}")
        return ok
    else:
        ok = _delete(f"/flows/application/{IBN_APP_ID}")
        if ok:
            print(f"   [ONOS] ✅ Cleared all IBN rules across all devices")
        return ok


def verify_rule_applied(switch_id: str, target_ip: str,
                        topology_nodes: list[dict] = None) -> bool:
    """
    Reads back the flow table on switch_id and confirms a rule matching
    target_ip was actually installed.

    Call this after push_throttle_rule() or push_priority_rule() to
    verify ONOS actually committed the rule to the dataplane.

    Returns True if the rule is found in the flow table.
    """
    alias_to_onos, _ = _build_lookup_maps(topology_nodes)
    onos_id = alias_to_onos.get(switch_id, switch_id)

    flows_raw = _get(f"/flows/{onos_id}")
    if not flows_raw:
        return False

    for flow in flows_raw.get("flows", []):
        if flow.get("appId") != IBN_APP_ID:
            continue
        criteria = (flow.get("selector", {})
                        .get("criteria", []))
        for c in criteria:
            if c.get("type") == "IPV4_DST" and c.get("ip") == target_ip:
                state = flow.get("state", "UNKNOWN")
                print(f"   [ONOS] ✅ Rule verified on {switch_id}: "
                      f"→ {target_ip}  state={state}")
                return state == "ADDED"

    print(f"   [ONOS] ⚠️  Rule for {target_ip} NOT found on {switch_id} "
          f"— ONOS may not have committed it yet")
    return False


# ════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════

def is_onos_reachable() -> bool:
    result    = _get("/devices")
    reachable = result is not None
    icon      = "✅" if reachable else "❌"
    print(f"   [ONOS] {icon} Controller at {BASE_URL} — "
          f"{'reachable' if reachable else 'NOT reachable'}")
    return reachable


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _load_ip_map() -> dict:
    """
    Reads topology_metadata.json and returns:
      { "srv_gaming": { "onos_id": "of:...", "ip": "10.0.0.1/32",
                        "hosted_service": "gaming", ... }, ... }

    If the file doesn't exist or has no onos_id fields, returns {}.
    """
    try:
        with open(_TOPO_FILE) as f:
            data = json.load(f)
        result = {}
        for node in data.get("nodes", []):
            result[node["id"]] = {
                "onos_id":        node.get("onos_id", node["id"]),
                "ip":             node.get("ip"),
                "hosted_service": node.get("hosted_service"),
            }
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _build_lookup_maps(topology_nodes: list[dict] = None):
    """
    Returns (alias_to_onos, ip_map) tuples.

    alias_to_onos : { "sw1": "of:0000000000000001", ... }
    ip_map        : { "srv_gaming": "10.0.0.1/32", ... }

    Prefers topology_nodes (from get_topology()) over the static file
    because topology_nodes already has both alias and onos_id merged.
    """
    file_overlay  = _load_ip_map()
    alias_to_onos = {k: v["onos_id"] for k, v in file_overlay.items()}
    ip_map        = {k: v["ip"] for k, v in file_overlay.items() if v.get("ip")}

    if topology_nodes:
        for n in topology_nodes:
            alias = n["id"]
            if "onos_id" in n:
                alias_to_onos[alias] = n["onos_id"]
            if n.get("ip"):
                ip_map[alias] = n["ip"]

    return alias_to_onos, ip_map


def _get_or_create_meter(onos_device_id: str, rate_mbps: float) -> int | None:
    """
    Finds an existing ONOS meter at rate_mbps on onos_device_id, or creates one.
    Returns the integer meter ID, or None on failure.

    ONOS meter unit: KB_PER_SEC
    rate_mbps × 1000 = rate_kbps
    """
    rate_kbps = int(rate_mbps * 1000)

    # Look for an existing meter at this rate to avoid duplicates
    existing = _get(f"/meters/{onos_device_id}")
    if existing:
        for meter in existing.get("meters", []):
            for band in meter.get("bands", []):
                if band.get("rate") == rate_kbps:
                    mid = meter.get("id") or meter.get("meterId")
                    print(f"   [ONOS] ♻️  Reusing meter {mid} @ {rate_mbps} Mbps "
                          f"on {onos_device_id}")
                    return int(mid)

    # Create a new meter
    payload = {
        "deviceId": onos_device_id,
        "appId":    IBN_APP_ID,
        "unit":     "KB_PER_SEC",
        "burst":    True,
        "bands": [{
            "type":      "DROP",            # drop excess traffic (policing)
            "rate":      rate_kbps,
            "burstSize": rate_kbps * 2,     # allow 2× burst headroom
        }]
    }
    result = _post(f"/meters/{onos_device_id}", payload)
    if result:
        meter_id = result.get("id") or result.get("meterId")
        if meter_id:
            print(f"   [ONOS] ✅ Created meter {meter_id} @ {rate_mbps} Mbps "
                  f"on {onos_device_id}")
            return int(meter_id)

    print(f"   [ONOS] ❌ Meter creation failed on {onos_device_id}")
    return None