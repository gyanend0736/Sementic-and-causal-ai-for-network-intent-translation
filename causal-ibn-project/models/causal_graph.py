def build_causal_graph_from_topology(network_state, target_node_id):
    """
    Builds a GML causal graph dynamically from your topology_metadata.json.
    Metric nodes are per-device; edges follow your actual switch→server links.
    """
    topology = network_state["topology"]
    nodes = topology.get("nodes", [])
    links = topology.get("links", [])

    # Metric columns that exist per-node in your telemetry CSV
    METRICS = [
        "active_flows", "bandwidth_used_mbps",
        "buffer_occupancy", "latency_ms",
        "packet_loss_percent", "cpu_utilization_percent"
    ]

    # Causal edges WITHIN a single node (physics of the device)
    INTRA_NODE_EDGES = [
        ("active_flows",          "bandwidth_used_mbps"),
        ("bandwidth_used_mbps",     "buffer_occupancy"),
        ("buffer_occupancy",        "latency_ms"),
        ("buffer_occupancy",        "packet_loss_percent"),
        ("cpu_utilization_percent", "latency_ms"),
        ("cpu_utilization_percent", "packet_loss_percent"),
    ]

    gml_nodes = []
    gml_edges = []
    seen_node_ids = set()

    # 1. Add metric nodes for every device in the topology
    for node in nodes:
        nid = node["id"]
        for metric in METRICS:
            var = f"{nid}_{metric}"
            if var not in seen_node_ids:
                gml_nodes.append(
                    f'  node [ id "{var}" label "{var}" ]'
                )
                seen_node_ids.add(var)

            # Intra-node causal edges (congestion physics)
        for src_m, tgt_m in INTRA_NODE_EDGES:
            gml_edges.append(
                f'  edge [ source "{nid}_{src_m}" target "{nid}_{tgt_m}" ]'
            )

        # 2. Inter-node edges: upstream switch bandwidth → downstream server latency
    for link in links:
        src, tgt = link["source"], link["target"]
        gml_edges.append(
            f'  edge [ source "{src}_bandwidth_used_mbps"'
            f' target "{tgt}_latency_ms" ]'
        )
        gml_edges.append(
            f'  edge [ source "{src}_bandwidth_used_mbps"'
            f' target "{tgt}_packet_loss_percent" ]'
        )

    gml = "graph [\n  directed 1\n"
    gml += "\n".join(gml_nodes) + "\n"
    gml += "\n".join(gml_edges) + "\n]"
    return gml