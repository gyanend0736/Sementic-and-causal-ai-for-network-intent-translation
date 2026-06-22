import pandas as pd
from dowhy import CausalModel

def run_causal_discovery():
    # 1. Load the telemetry data you generated
    df = pd.read_csv('data/raw_telemetry.csv')

    # 2. Define the Causal Graph (The DAG)
    # This represents the physical logic: Flows -> Bandwidth -> Buffer -> Latency
    causal_graph = """
    graph [
        active_flows -> bandwidth_mbps;
        bandwidth_mbps -> buffer_occupancy;
        buffer_occupancy -> latency_ms;
        cpu_utilization -> latency_ms;
    ]
    """

    # 3. Initialize the Causal Model
    model = CausalModel(
        data=df,
        treatment='bandwidth_mbps', # The "Cause" we want to test
        outcome='latency_ms',       # The "Effect" we are measuring
        graph=causal_graph
    )

    # 4. Identification: Can we actually calculate this effect?
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 5. Estimation: Calculate how much 1Mbps of bandwidth affects Latency
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression")

    print(f"*** Causal Discovery Result ***")
    print(f"Causal Estimate Value: {estimate.value}")
    
    return model, estimate

if __name__ == "__main__":
    run_causal_discovery()