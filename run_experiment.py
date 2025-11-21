import argparse
import subprocess
import os
import sys
import time
import itertools
import concurrent.futures
import numpy as np
import random
import csv

# --- グローバル設定 ---
MAX_WORKERS = 18
SIMULATION_BASE_PARAMS = {
    "packets": "1000",
    "max_sim_time": "inf",
    "ttl": "100.0",
    "buffer_size": "-1"
}
ROUTING_STRATEGIES = ["dijkstra", "reliable"]
STRATEGIES_WITH_PARAMS = {
    "dynamic_wait_strategy_with_faillink": ("dynamic_factor", np.arange(0.5, 3.1, 0.5)),
    "dynamic_wait_strategy_with_node_count": ("dynamic_factor", np.arange(0.5, 3.1, 0.5)),
    "ratio_based_wait_strategy": ("ratio_factor", np.arange(0.5, 3.1, 0.5)),
    "fixed_wait_duration_strategy": ("base_wait_time", np.arange(0.5, 5.1, 0.5)),
    "no_wait_strategy": (None, [0]),
    "infinite_wait_strategy": (None, [0]),
}
STRATEGY_ABBREVIATIONS = {
    "dynamic_wait_strategy_with_faillink": "dynfail",
    "dynamic_wait_strategy_with_node_count": "dynnode",
    "ratio_based_wait_strategy": "ratio",
    "fixed_wait_duration_strategy": "fixed",
    "no_wait_strategy": "nowait",
    "infinite_wait_strategy": "infwait",
}
ROUTING_STRATEGY_ABBREVIATIONS = {
    "dijkstra": "dijk",
    "reliable": "reli",
}

# --- ヘルパー関数 ---
def run_single_simulation_task(task_info):
    """
    単一のシミュレーションタスクを実行するワーカー関数。
    """
    # Unpack all task info
    for key, value in task_info.items():
        globals()[key] = value

    start_time = time.time()
    
    command = [
        sys.executable, "nwsim.py",
        "--node_file", node_file,
        "--edge_file", edge_file,
        "--src_node", str(src_node),
        "--dst_node", str(dst_node),
        "--packets", SIMULATION_BASE_PARAMS["packets"],
        "--max_sim_time", SIMULATION_BASE_PARAMS["max_sim_time"],
        "--ttl", SIMULATION_BASE_PARAMS["ttl"],
        "--buffer_size", SIMULATION_BASE_PARAMS["buffer_size"],
        "--routing_strategy", routing,
        "--strategy", wait,
        "--output_base_dir", output_dir,
        "--summary_filename", summary_filename
    ]

    if param_type:
        command.extend([f"--{param_type}", str(param_value)])
    
    # Pass through any other arguments
    if unknown_args:
        command.extend(unknown_args)

    try:
        result = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[Task {task_id}] SUCCESS in {execution_time:.2f}s: {topo_str}/run_{run_idx}/{routing_abbr}/{wait_abbr}/p={param_value}")
        return True, None
    except subprocess.CalledProcessError as e:
        error_message = (
            f"[Task {task_id}] FAILED: {topo_str}/run_{run_idx}"
            f"  CMD: {' '.join(command)}"
            f"  Output:\n{e.stdout}\n{e.stderr}\n"
        )
        print(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred for task {task_id}: {e}"
        print(error_message)
        return False, error_message

def main():
    parser = argparse.ArgumentParser(description="Run a full experiment by generating multiple graphs and running sweeps on them.")
    parser.add_argument("--num_runs", type=int, required=True, help="Number of random graphs to generate and simulate for each topology.")
    parser.add_argument("--topology_type", type=str, default="all", help="Topology type to generate. 'all' for every supported topology.")
    
    args, unknown_args = parser.parse_known_args()

    # --- トポロジの定義 ---
    all_supported_topologies = [
        "random", 
        # "grid", 
        "barabasi_albert", 
        # "path", 
        # "ring", 
        "k_nearest_neighbor", 
        "rgg", 
        "multi_hub_star"
    ]

    if args.topology_type.lower() == 'all':
        target_topologies = all_supported_topologies
    elif args.topology_type in all_supported_topologies:
        target_topologies = [args.topology_type]
    else:
        print(f"Error: Unknown topology type '{args.topology_type}'.")
        sys.exit(1)

    # --- ディレクトリ定義 ---
    GRAPHS_BASE_DIR = "graph_data"
    RESULTS_BASE_DIR = "result_data"
    os.makedirs(GRAPHS_BASE_DIR, exist_ok=True)
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    all_simulation_tasks = []
    task_counter = 0

    # --- トポロジごとの特別パラメータ設定 ---
    topology_specific_params = {
        "barabasi_albert": {"name": "barabasi_m", "values": [2, 3]},
        "k_nearest_neighbor": {"name": "k_neighbors", "values": [2, 3]},
    }

    # --- PHASE 1: グラフ生成 & タスク集約 ---
    print("="*80 + "\nPHASE 1: Generating Graphs & Aggregating Tasks\n" + "="*80)

    for topo in target_topologies:
        
        # このトポロジで実行するパラメータセットを決定
        params_to_run = [{}]  # デフォルトは追加パラメータなし
        topo_param_config = topology_specific_params.get(topo)
        
        if topo_param_config:
            param_name = topo_param_config["name"]
            # コマンドライン引数でパラメータが指定されているか確認
            is_param_in_args = any(f'--{param_name}' in arg for arg in unknown_args)
            if not is_param_in_args:
                params_to_run = [
                    {"name": param_name, "value": v, "dir_suffix": f"_{param_name.replace('_','')}{v}"} 
                    for v in topo_param_config["values"]
                ]

        for i in range(args.num_runs):
            run_idx = i + 1
            
            for param_set in params_to_run:
                param_name = param_set.get("name")
                param_value = param_set.get("value")
                dir_suffix = param_set.get("dir_suffix", "")
                topo_str = f"{topo}{dir_suffix}"

                print(f"\n--- Processing: {topo_str}, Run {run_idx}/{args.num_runs} ---")
                
                # --- グラフ生成 ---
                run_graph_dir = os.path.join(GRAPHS_BASE_DIR, topo_str, f"run_{run_idx}")
                os.makedirs(run_graph_dir, exist_ok=True)
                
                node_path = os.path.join(run_graph_dir, "node.csv")
                edge_path = os.path.join(run_graph_dir, "edge.csv")
                diameter_path = os.path.join(run_graph_dir, "diameter_endpoints.csv")
                
                seed = random.randint(0, 99999)

                make_cmd = [
                    sys.executable, "make_network_degraded.py",
                    "--topology_type", topo,
                    "--node_output_path", node_path,
                    "--edge_output_path", edge_path,
                    "--seed", str(seed),
                ]
                if param_name:
                    make_cmd.extend([f"--{param_name}", str(param_value)])
                
                make_cmd.extend(unknown_args)

                try:
                    subprocess.run(make_cmd, check=True, text=True, encoding='utf-8', capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"    ERROR: Graph generation failed for {topo_str}, run {run_idx}.\n      CMD: {' '.join(e.cmd)}\n      Output:\n{e.stdout}\n{e.stderr}")
                    continue

                # --- タスク集約 ---
                try:
                    with open(diameter_path, 'r') as f:
                        reader = csv.DictReader(f)
                        first_row = next(reader)
                        src_node = int(first_row['node1'])
                        dst_node = int(first_row['node2'])
                except (FileNotFoundError, StopIteration):
                    print(f"  WARNING: Could not read diameter file for {topo_str}, run {run_idx}. Using default src=0, dst=-1.")
                    src_node, dst_node = 0, -1 

                run_result_dir = os.path.join(RESULTS_BASE_DIR, topo_str, f"run_{run_idx}")

                for routing in ROUTING_STRATEGIES:
                    routing_abbr = ROUTING_STRATEGY_ABBREVIATIONS[routing]
                    for wait_strategy_name, (sim_param_type, sim_param_values) in STRATEGIES_WITH_PARAMS.items():
                        wait_abbr = STRATEGY_ABBREVIATIONS[wait_strategy_name]
                        for sim_param_value in sim_param_values:
                            result_output_dir = os.path.join(run_result_dir, routing_abbr, wait_abbr)
                            summary_filename = f"summary_p{sim_param_value:.1f}.csv" if sim_param_type else "summary.csv"
                            task_counter += 1
                            all_simulation_tasks.append({
                                "topo_str": topo_str, "run_idx": run_idx, "routing": routing, "wait": wait_strategy_name,
                                "param_type": sim_param_type, "param_value": sim_param_value, "node_file": node_path,
                                "edge_file": edge_path, "src_node": src_node, "dst_node": dst_node,
                                "output_dir": result_output_dir, "summary_filename": summary_filename,
                                "task_id": task_counter, "total_tasks": None,
                                "routing_abbr": routing_abbr, "wait_abbr": wait_abbr,
                                "unknown_args": unknown_args
                            })
    
    if not all_simulation_tasks:
        print("No simulation tasks were generated. Exiting.")
        sys.exit(0)

    for task in all_simulation_tasks:
        task["total_tasks"] = len(all_simulation_tasks)

    print(f"\nAggregated a total of {len(all_simulation_tasks)} simulation tasks.")

    # --- PHASE 2: シミュレーションタスクの並列実行 ---
    print("\n" + "="*80 + "\nPHASE 2: Running All Simulation Tasks in Parallel\n" + "="*80)
    workers = MAX_WORKERS if MAX_WORKERS is not None else os.cpu_count()
    print(f"Using {workers} worker processes.")

    success_count = 0
    failed_tasks_info = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_simulation_task, task) for task in all_simulation_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                success, error_message = future.result()
                if success: success_count += 1
                else:
                    failed_tasks_info.append(error_message)
            except Exception as exc:
                failed_tasks_info.append(f"A task generated an exception: {exc}")

    print("\n" + "="*80 + "\nAll Simulation Tasks Completed.\n" + "="*80)
    print(f"Total tasks: {len(all_simulation_tasks)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {len(failed_tasks_info)}")
    
    if failed_tasks_info:
        print("\n--- FAILED TASKS SUMMARY ---")
        for info in failed_tasks_info:
            print(info)

    print(f"\nGraph data saved in: {GRAPHS_BASE_DIR}")
    print(f"Simulation results saved in: {RESULTS_BASE_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()