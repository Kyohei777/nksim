import subprocess
import os
from datetime import datetime
import csv
import sys

# Base directory for the new project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the scripts
MAKE_NETWORK_SCRIPT = os.path.join(BASE_DIR, "make_network_for_nwsim.py")
RUN_CUSTOM_SWEEP_SCRIPT = os.path.join(BASE_DIR, "run_custom_sweep.py")
RUN_SIMULATION_PYTHON_SCRIPT = os.path.join(BASE_DIR, "run_simulation_python.py") # Not directly used, but for reference

# Simulation parameters
NUMBER_OF_NODES = 100
MIN_EDGE_WEIGHT = 0.5
MAX_EDGE_WEIGHT = 0.5
BUFFER_SIZE = 20  # -1 for infinite
TTL = "inf"  # "inf" for infinite

# Define topologies and their specific parameters
# Excluded: 'path', 'ring', 'multi_hub_star'
TOPOLOGIES_TO_SIMULATE = {
    "random": {"PROB_EDGE_RANDOM": 0.2},
    "grid": {},
    "barabasi_albert_m2": {"BARABASI_M": 2},
    "barabasi_albert_m3": {"BARABASI_M": 3},
    "k_nearest_neighbor_k2": {"K_NEIGHBORS_K": 2},
    "k_nearest_neighbor_k3": {"K_NEIGHBORS_K": 3},
    "rgg": {"CONNECTION_RADIUS_RGG": 0.15},
}

def run_command(command, description):
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=BASE_DIR)
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
        print(f"ERROR: Command not found. Make sure Python and scripts are in PATH or correctly specified.")
        return False

def main():
    print("--- Starting Orchestrated Network Simulations ---")

    for topo_name, params in TOPOLOGIES_TO_SIMULATE.items():
        print(f"\n--- Processing Topology: {topo_name} ---")

        # Create a unique directory for each topology's generated graph files
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topo_output_dir = os.path.join(BASE_DIR, f"graph_data_{topo_name}_{current_timestamp}")
        os.makedirs(topo_output_dir, exist_ok=True)
        print(f"Graph data will be saved in: {topo_output_dir}")

        node_file_path = os.path.join(topo_output_dir, "node.csv")
        edge_file_path = os.path.join(topo_output_dir, "edge.csv")
        diameter_endpoints_file_path = os.path.join(topo_output_dir, "diameter_endpoints.csv")

        # 1. Generate graph files using make_network_for_nwsim.py
        topology_type_for_script = topo_name.split('_')[0]
        if "k_nearest_neighbor" in topo_name:
            topology_type_for_script = "k_nearest_neighbor"
        
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
        
        # Add topology-specific parameters
        if "PROB_EDGE_RANDOM" in params:
            make_network_cmd.append(f"--probability_of_edge={params['PROB_EDGE_RANDOM']}")
        if "K_NEIGHBORS_K" in params:
            make_network_cmd.append(f"--k_neighbors={params['K_NEIGHBORS_K']}")
        if "CONNECTION_RADIUS_RGG" in params:
            make_network_cmd.append(f"--connection_radius={params['CONNECTION_RADIUS_RGG']}")
        if "BARABASI_M" in params:
            make_network_cmd.append(f"--barabasi_m={params['BARABASI_M']}")


        if not run_command(make_network_cmd, f"Generating graph for {topo_name}"):
            print(f"Skipping simulation for {topo_name} due to graph generation failure.")
            continue

        # # 2. Read source and destination nodes from diameter_endpoints.csv
        # source_node = None
        # dest_node = None
        # try:
        #     with open(diameter_endpoints_file_path, 'r', newline='') as f:
        #         reader = csv.DictReader(f)
        #         first_row = next(reader)
        #         source_node = int(first_row['node1'])
        #         dest_node = int(first_row['node2'])
        #     print(f"Read source node: {source_node}, destination node: {dest_node} from {diameter_endpoints_file_path}")
        # except FileNotFoundError:
        #     print(f"ERROR: {diameter_endpoints_file_path} not found. Cannot determine source/destination nodes.")
        #     print(f"Skipping simulation for {topo_name}.")
        #     continue
        # except Exception as e:
        #     print(f"ERROR: Could not read source/destination nodes from {diameter_endpoints_file_path}: {e}")
        #     print(f"Skipping simulation for {topo_name}.")
        #     continue

        # # 3. Run the custom sweep simulation
        # simulation_output_dir = os.path.join(BASE_DIR, f"result_{topo_name}_{datetime.now().strftime('%Y%m%d')}")
        # run_sweep_cmd = [
        #     sys.executable,
        #     RUN_CUSTOM_SWEEP_SCRIPT,
        #     node_file_path,
        #     edge_file_path,
        #     str(source_node),
        #     str(dest_node),
        #     simulation_output_dir,
        #     "--buffer_size",
        #     str(BUFFER_SIZE),
        #     "--ttl",
        #     str(TTL),
        # ]
        # if not run_command(run_sweep_cmd, f"Running custom sweep for {topo_name}"):
        #     print(f"Simulation sweep for {topo_name} failed.")
        #     continue
        
        print(f"--- Finished Processing Topology: {topo_name} ---")

    print("\n--- All Orchestrated Network Simulations Completed ---")

if __name__ == "__main__":
    main()
