import multiprocessing
import subprocess
import time
import os
import sys
import numpy as np
from datetime import datetime
import csv

# --- General Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAKE_NETWORK_SCRIPT = os.path.join(BASE_DIR, "make_network_for_nwsim.py")
TARGET_SIM_SCRIPT = os.path.join(BASE_DIR, "nwsim.py")

# --- Graph Generation Parameters ---
NUMBER_OF_NODES = 100
MIN_EDGE_WEIGHT = 0.5
MAX_EDGE_WEIGHT = 0.5

# Define topologies and their specific parameters for graph generation
TOPOLOGIES_TO_SIMULATE = {
    "random": {"PROB_EDGE_RANDOM": 0.2},
    "grid": {},
    "barabasi_albert_m2": {"BARABASI_M": 2},
    "barabasi_albert_m3": {"BARABASI_M": 3},
    "k_nearest_neighbor_k2": {"K_NEIGHBORS_K": 2},
    "k_nearest_neighbor_k3": {"K_NEIGHBORS_K": 3},
    "rgg": {"CONNECTION_RADIUS_RGG": 0.15},
}

# --- Simulation Sweep Parameters ---
# General simulation settings
BUFFER_SIZE = -1  # -1 for infinite
TTL = 100  # "inf" for infinite
PACKET_COUNT = 2000

# Parameter sweeps for connect_rate and disconnect_rate
rate_sweep_part1 = np.arange(0.1, 1.0, 0.1)
rate_sweep_part2 = np.arange(1.0, 4.1, 1.0)
RATE_SWEEP_VALUES = np.round(
    np.unique(np.concatenate((rate_sweep_part1, rate_sweep_part2))), 1
)

# Strategies to run with their parameter types and ranges
STRATEGIES_WITH_PARAMS = {
    "dynamic_wait_strategy_with_faillink": ("dynamic_factor", np.arange(0.5, 3.1, 0.5)),
    "dynamic_wait_strategy_with_node_count": (
        "dynamic_factor",
        np.arange(0.5, 3.1, 0.5),
    ),
    "ratio_based_wait_strategy": ("ratio_factor", np.arange(0.5, 3.1, 0.5)),
    "fixed_wait_duration_strategy": ("base_wait_time", np.arange(0.5, 5.1, 0.5)),
    "no_wait_strategy": (None, [0]),
    "infinite_wait_strategy": (None, [0]),
}

# Strategy name abbreviations for filenames and directories
STRATEGY_ABBREVIATIONS = {
    "dynamic_wait_strategy_with_faillink": "dynfail",
    "dynamic_wait_strategy_with_node_count": "dynnode",
    "ratio_based_wait_strategy": "ratio",
    "fixed_wait_duration_strategy": "fixed",
    "no_wait_strategy": "nowait",
    "infinite_wait_strategy": "infwait",
}

# Routing strategy abbreviations
ROUTING_STRATEGY_ABBREVIATIONS = {
    "dijkstra": "dijk",
    "reliable": "reli",
}

# --- Parallelism Configuration ---
NUM_PROCESSES = 18
NUM_LOGICAL_CORES = os.cpu_count() if hasattr(os, "cpu_count") else "N/A"


def run_command_for_graph(command, description):
    """Helper to run a command and capture output, for graph generation."""
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, cwd=BASE_DIR
        )
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(
            f"ERROR: Command not found. Make sure Python and scripts are in PATH or correctly specified."
        )
        return False


def run_single_simulation(task_info):
    """
    Runs a single simulation task. This function is executed by worker processes.
    """
    # Unpack all task info
    strategy = task_info["strategy"]
    routing_strategy = task_info["routing_strategy"]
    connect_rate = task_info["connect_rate"]
    disconnect_rate = task_info["disconnect_rate"]
    param_type = task_info["param_type"]
    param_value = task_info["param_value"]
    task_id = task_info["task_id"]

    node_file = task_info["node_file"]
    edge_file = task_info["edge_file"]
    src_node = task_info["src_node"]
    dst_node = task_info["dst_node"]

    buffer_size = task_info["buffer_size"]
    ttl = task_info["ttl"]
    packets = task_info["packets"]

    topology_name = task_info["topology_name"]
    topo_result_dir = task_info["topo_result_dir"]

    strategy_abbr = task_info["strategy_abbr"]
    routing_strategy_abbr = task_info["routing_strategy_abbr"]

    # --- 1. Determine directory and file names based on the new structure ---
    if param_type:
        param_path_str = f"{param_value:.1f}"
        param_file_str = f"_{param_value:.1f}"
    else:
        param_path_str = "no_params"
        param_file_str = "_0"

    # Base directory for this routing strategy
    routing_base_dir = os.path.join(topo_result_dir, routing_strategy_abbr)

    # Log directory
    log_dir = os.path.join(routing_base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # Final output directory for summary file
    output_dir = os.path.join(
        routing_base_dir,
        strategy_abbr,
        param_path_str,
        f"c{connect_rate:.1f}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Log filename
    log_base_name = (
        f"l_c{connect_rate:.1f}_d{disconnect_rate:.1f}"
        f"_{strategy_abbr}{param_file_str}"
    )
    log_filename = f"{log_base_name}.txt"
    log_file_path = os.path.join(log_dir, log_filename)

    # Summary filename
    summary_base_name = (
        f"s_c{connect_rate:.1f}_d{disconnect_rate:.1f}"
        f"_{strategy_abbr}{param_file_str}"
    )
    summary_filename = f"{summary_base_name}.csv"

    # --- 2. Construct the command for nwsim.py ---
    command = [
        sys.executable,
        TARGET_SIM_SCRIPT,
        "--node_file",
        node_file,
        "--edge_file",
        edge_file,
        "--src_node",
        str(src_node),
        "--dst_node",
        str(dst_node),
        "--strategy",
        strategy,
        "--routing_strategy",
        routing_strategy,
        "--connect_rate",
        str(connect_rate),
        "--disconnect_rate",
        str(disconnect_rate),
        "--packets",
        str(packets),
        "--output_base_dir",
        output_dir,  # Pass the final leaf directory
        "--summary_filename",
        summary_filename,
        "--buffer_size",
        str(buffer_size),
        "--ttl",
        str(ttl),
    ]

    if param_type == "base_wait_time":
        command.extend(["--base_wait_time", str(param_value)])
    elif param_type == "dynamic_factor":
        command.extend(["--dynamic_factor", str(param_value)])
    elif param_type == "ratio_factor":
        command.extend(["--ratio_factor", str(param_value)])

    # --- 3. Execute the simulation ---
    start_time = time.time()
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                encoding="utf-8",
            )
            end_time = time.time()
            execution_time = end_time - start_time
            print(
                f"[Task {task_id}][{topology_name}][{routing_strategy_abbr}][{strategy_abbr}] Finished in {execution_time:.2f}s."
            )
            return (task_id, True, execution_time, "")
        except subprocess.CalledProcessError as e:
            error_message = f"[Task {task_id}][{topology_name}][{routing_strategy_abbr}][{strategy_abbr}] FAILED. Check log: {log_file_path} (Code: {e.returncode})"
            print(error_message)
            return (task_id, False, 0, error_message)
        except Exception as e:
            error_message = f"[Task {task_id}][{topology_name}][{routing_strategy_abbr}][{strategy_abbr}] FAILED with unexpected error. Check log: {log_file_path} (Error: {e})"
            print(error_message)
            return (task_id, False, 0, error_message)


def main():
    """Main orchestrator for the entire simulation process."""

    # --- PHASE 1: Generate all graph files sequentially ---
    print("--- PHASE 1: Generating all graph files ---")
    generated_graphs_info = []
    for topo_name, params in TOPOLOGIES_TO_SIMULATE.items():
        print(f"\n--- Processing Topology: {topo_name} ---")

        graph_output_dir = os.path.join(BASE_DIR, "graph_data", topo_name)
        os.makedirs(graph_output_dir, exist_ok=True)
        print(f"Graph data will be saved in: {graph_output_dir}")

        node_file_path = os.path.join(graph_output_dir, "node.csv")
        edge_file_path = os.path.join(graph_output_dir, "edge.csv")
        diameter_endpoints_file_path = os.path.join(
            graph_output_dir, "diameter_endpoints.csv"
        )

        if "barabasi_albert" in topo_name:
            topology_type_for_script = "barabasi_albert"
        elif "k_nearest_neighbor" in topo_name:
            topology_type_for_script = "k_nearest_neighbor"
        else:
            topology_type_for_script = topo_name.split("_")[0]

        make_network_cmd = [
            sys.executable,
            MAKE_NETWORK_SCRIPT,
            f"--num_nodes={NUMBER_OF_NODES}",
            f"--min_weight={MIN_EDGE_WEIGHT}",
            f"--max_weight={MAX_EDGE_WEIGHT}",
            f"--node_output_path={node_file_path}",
            f"--edge_output_path={edge_file_path}",
            f"--topology_type={topology_type_for_script}",
        ]

        if "PROB_EDGE_RANDOM" in params:
            make_network_cmd.append(
                f"--probability_of_edge={params['PROB_EDGE_RANDOM']}"
            )
        if "K_NEIGHBORS_K" in params:
            make_network_cmd.append(f"--k_neighbors={params['K_NEIGHBORS_K']}")
        if "CONNECTION_RADIUS_RGG" in params:
            make_network_cmd.append(
                f"--connection_radius={params['CONNECTION_RADIUS_RGG']}"
            )
        if "BARABASI_M" in params:
            make_network_cmd.append(f"--barabasi_m={params['BARABASI_M']}")

        if not run_command_for_graph(
            make_network_cmd, f"Generating graph for {topo_name}"
        ):
            print(
                f"FATAL: Skipping all simulations for {topo_name} due to graph generation failure."
            )
            continue

        try:
            with open(diameter_endpoints_file_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                first_row = next(reader)
                source_node = int(first_row["node1"])
                dest_node = int(first_row["node2"])
                print(
                    f"Successfully read source: {source_node}, destination: {dest_node}"
                )

                generated_graphs_info.append(
                    {
                        "topology_name": topo_name,
                        "node_file": node_file_path,
                        "edge_file": edge_file_path,
                        "src_node": source_node,
                        "dst_node": dest_node,
                    }
                )
        except (FileNotFoundError, StopIteration) as e:
            print(
                f"FATAL: Could not read diameter endpoints from {diameter_endpoints_file_path}. Error: {e}"
            )
            print(f"Skipping all simulations for {topo_name}.")
            continue

    # --- PHASE 2: Aggregate all simulation tasks ---
    print("\n\n--- PHASE 2: Aggregating all simulation tasks ---")
    all_tasks = []
    task_id_counter = 0

    ROUTING_STRATEGIES_TO_SIMULATE = ["dijkstra", "reliable"]

    # Create the root result directory
    results_root_dir = os.path.join(BASE_DIR, "result_data")
    os.makedirs(results_root_dir, exist_ok=True)

    for graph_info in generated_graphs_info:
        topo_name = graph_info["topology_name"]
        # Directory for this specific topology's results
        topo_result_dir = os.path.join(
            results_root_dir, f"{topo_name}_{datetime.now().strftime('%Y%m%d')}"
        )
        os.makedirs(topo_result_dir, exist_ok=True)

        for routing_strategy in ROUTING_STRATEGIES_TO_SIMULATE:
            for con_rate in RATE_SWEEP_VALUES:
                for dis_rate in RATE_SWEEP_VALUES:
                    for strategy_name, (
                        param_type,
                        param_range,
                    ) in STRATEGIES_WITH_PARAMS.items():
                        for param_value in param_range:
                            task = {
                                "strategy": strategy_name,
                                "routing_strategy": routing_strategy,
                                "connect_rate": con_rate,
                                "disconnect_rate": dis_rate,
                                "param_type": param_type,
                                "param_value": param_value,
                                "task_id": task_id_counter,
                                "node_file": graph_info["node_file"],
                                "edge_file": graph_info["edge_file"],
                                "src_node": graph_info["src_node"],
                                "dst_node": graph_info["dst_node"],
                                "topology_name": topo_name,
                                "topo_result_dir": topo_result_dir,
                                "buffer_size": BUFFER_SIZE,
                                "ttl": TTL,
                                "packets": PACKET_COUNT,
                                "strategy_abbr": STRATEGY_ABBREVIATIONS[strategy_name],
                                "routing_strategy_abbr": ROUTING_STRATEGY_ABBREVIATIONS[
                                    routing_strategy
                                ],
                            }
                            all_tasks.append(task)
                            task_id_counter += 1

    if not all_tasks:
        print("No simulation tasks were generated. Exiting.")
        return

    print(f"Created a total of {len(all_tasks)} simulation tasks.")
    print(f"Results will be stored in: {results_root_dir}")
    print(
        f"Parallel processes to use: {NUM_PROCESSES} (Logical cores available: {NUM_LOGICAL_CORES})"
    )

    # --- PHASE 3: Execute all tasks in parallel ---
    print("\n\n--- PHASE 3: Running all simulations in parallel ---")
    start_total_time = time.time()
    results = []

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        result_iterator = pool.imap_unordered(run_single_simulation, all_tasks)
        for i, result in enumerate(result_iterator):
            results.append(result)
            progress = (i + 1) / len(all_tasks) * 100
            print(
                f"Progress: {i + 1}/{len(all_tasks)} ({progress:.2f}%) tasks completed.",
                end="\r",
            )

    print("\n")
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    # --- PHASE 4: Process and report results ---
    print("\n--- PHASE 4: All Simulations Completed ---")
    successful_runs = sum(1 for r in results if r[1])
    failed_runs = len(results) - successful_runs

    print(
        f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")

    if failed_runs > 0:
        print("\n--- FAILED RUNS LOG ---")
        for r in results:
            if not r[1]:
                print(r[3])

    print(f"\nAll simulation results are stored in: {results_root_dir}")


if __name__ == "__main__":
    main()
