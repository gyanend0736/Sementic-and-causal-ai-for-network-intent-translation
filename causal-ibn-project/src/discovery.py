"""
discovery.py — Network state discovery (STATIC / CSV mode).

For result screenshots: reads directly from raw_telemetry.csv
and topology_metadata.json — no ONOS, no network required.

To switch back to live ONOS mode later:
  set ONOS_ENABLED=true in your environment variables.
  The code automatically checks this flag and routes accordingly.
"""

import os
import json
import csv

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(BASE_DIR)
TOPOLOGY_FILE  = os.path.join(PROJECT_ROOT, 'data', 'topology_metadata.json')
TELEMETRY_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw_telemetry.csv')

# ── Set to "true" to switch to live ONOS mode when ready ──────────────────────
ONOS_ENABLED = os.environ.get("ONOS_ENABLED", "false").lower() == "true"

# Number of most-recent CSV rows per node kept as training window for inference.py
TELEMETRY_WINDOW = 50


def get_current_network_state() -> dict | None:
    """
    Main entry point called by orchestrator.py on every request.

    Returns:
        {
            "source":    "static_files",
            "topology":  { "nodes": [...], "links": [...] },
            "telemetry": { "<node_id>": { metric: value, ... } }
        }
        or None on any file error.
    """
    if ONOS_ENABLED:
        print("   [Discovery] Mode: LIVE (ONOS REST API)")
        return _get_live_state()

    print("   [Discovery] Mode: STATIC (raw_telemetry.csv + topology_metadata.json)")
    return _get_static_state()


# ─────────────────────────────────────────────────────────────────────────────
# STATIC MODE  —  reads your CSV and JSON files directly
# ─────────────────────────────────────────────────────────────────────────────

def _get_static_state() -> dict | None:
    """
    Reads topology_metadata.json for the network graph and
    raw_telemetry.csv for per-node metric values.

    The last row per node in the CSV is treated as the live reading.
    inference.py reads the full CSV independently for GCM training.
    """
    network_state = {
        "topology":  {},
        "telemetry": {},
        "source":    "static_files",
    }

    # 1. Load topology
    try:
        with open(TOPOLOGY_FILE, 'r') as f:
            network_state["topology"] = json.load(f)
        n_nodes = len(network_state["topology"].get("nodes", []))
        n_links = len(network_state["topology"].get("links", []))
        print(f"   [Discovery] Topology loaded: {n_nodes} nodes, {n_links} links")
    except FileNotFoundError:
        print(f"   [Discovery] ERROR: Topology file not found: {TOPOLOGY_FILE}")
        return None
    except json.JSONDecodeError as e:
        print(f"   [Discovery] ERROR: Topology JSON malformed: {e}")
        return None

    # 2. Load telemetry
    telemetry = _load_csv_telemetry()
    if not telemetry:
        print(f"   [Discovery] ERROR: Telemetry CSV not found or empty: {TELEMETRY_FILE}")
        return None
    network_state["telemetry"] = telemetry

    # 3. Sanity check — node IDs must match between topology and CSV
    topo_ids      = {n["id"] for n in network_state["topology"].get("nodes", [])}
    telemetry_ids = set(telemetry.keys())
    missing = topo_ids - telemetry_ids
    if missing:
        print(f"   [Discovery] WARN: Nodes in topology with NO telemetry: {missing}")
        print(f"               Ensure node_id values in raw_telemetry.csv match topology_metadata.json")

    # 4. Print live readings for confirmation in terminal
    print(f"   [Discovery] Live readings (last CSV row per node):")
    for nid, data in sorted(telemetry.items()):
        print(f"      {nid:10s}  "
              f"bw={data['bandwidth_used_mbps']:7.3f} Mbps  "
              f"lat={data['latency_ms']:7.3f} ms  "
              f"loss={data['packet_loss_percent']:.4f}%  "
              f"buf={data['buffer_occupancy']:5.1f}%  "
              f"jitter={data['jitter_ms']:6.3f} ms")

    return network_state


def _load_csv_telemetry() -> dict:
    """
    Reads raw_telemetry.csv.
    Groups rows by node_id, keeps last TELEMETRY_WINDOW rows per node,
    returns only the LAST row as the live telemetry value per node.
    All columns parsed defensively — missing columns default to 0.0.
    """
    telemetry = {}

    def safe_float(row, key, default=0.0):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return default

    try:
        rows_by_node = {}
        with open(TELEMETRY_FILE, 'r') as f:
            for row in csv.DictReader(f):
                nid = row.get('node_id', '').strip()
                if not nid:
                    continue
                rows_by_node.setdefault(nid, []).append(row)

        for nid, rows in rows_by_node.items():
            last = rows[-TELEMETRY_WINDOW:][-1]   # last row of the window
            telemetry[nid] = {
                "bandwidth_used_mbps":     safe_float(last, 'bandwidth_used_mbps'),
                "latency_ms":              safe_float(last, 'latency_ms'),
                "packet_loss_percent":     safe_float(last, 'packet_loss_percent'),
                "cpu_utilization_percent": safe_float(last, 'cpu_utilization_percent'),
                "buffer_occupancy":        safe_float(last, 'buffer_occupancy'),
                "jitter_ms":               safe_float(last, 'jitter_ms'),
                "active_flows":            safe_float(last, 'active_flows'),
            }

    except FileNotFoundError:
        pass   # caller handles the empty dict case

    return telemetry


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ONOS MODE  —  only called when ONOS_ENABLED=true
# ─────────────────────────────────────────────────────────────────────────────

def _get_live_state() -> dict | None:
    """
    Fetches live topology and telemetry from ONOS REST API.
    Falls back to static CSV if ONOS is unreachable.
    """
    try:
        import sys
        src_dir = os.path.dirname(os.path.abspath(__file__))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from onos_client import get_topology, get_live_telemetry, is_onos_reachable
    except ImportError:
        print("   [Discovery] ERROR: onos_client.py not found — falling back to CSV.")
        return _get_static_state()

    if not is_onos_reachable():
        print("   [Discovery] WARN: ONOS unreachable — falling back to CSV.")
        return _get_static_state()

    onos_topology = get_topology()
    if not onos_topology:
        print("   [Discovery] ERROR: Could not fetch topology from ONOS.")
        return None

    overlay = _load_service_overlay()
    for node in onos_topology["nodes"]:
        nid = node["id"]
        node["hosted_service"] = overlay.get(nid, {}).get("hosted_service")
        node["ip"]             = overlay.get(nid, {}).get("ip")

    node_ids   = [n["id"] for n in onos_topology["nodes"]]
    onos_telem = get_live_telemetry(node_ids, onos_topology["nodes"])

    csv_telem = _load_csv_telemetry()
    for nid, data in onos_telem.items():
        csv_row = csv_telem.get(nid, {})
        data["latency_ms"]              = csv_row.get("latency_ms",              0.0)
        data["jitter_ms"]               = csv_row.get("jitter_ms",               0.0)
        data["cpu_utilization_percent"] = csv_row.get("cpu_utilization_percent", 0.0)
        data["buffer_occupancy"]        = csv_row.get("buffer_occupancy",        0.0)

    return {"topology": onos_topology, "telemetry": onos_telem, "source": "onos_live"}


def _load_service_overlay() -> dict:
    overlay = {}
    try:
        with open(TOPOLOGY_FILE, 'r') as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            overlay[node["id"]] = {
                "hosted_service": node.get("hosted_service"),
                "ip":             node.get("ip"),
            }
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return overlay