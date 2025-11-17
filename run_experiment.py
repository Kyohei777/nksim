import argparse
import subprocess
import os
import sys
import time
import itertools
import concurrent.futures
import numpy as np # np.arange を使用するため

# --- グローバル設定 ---
# 並列実行するプロセス数 (NoneにするとCPUのコア数が自動的に使われます)
MAX_WORKERS = 4 

# シミュレーションの基本パラメータ (nwsim.py に渡すもの)
SIMULATION_BASE_PARAMS = {
    "packets": "1000",
    "max_sim_time": "inf",
    "ttl": "100.0",
    "buffer_size": "-1"
}

# ルーティング戦略の定義
ROUTING_STRATEGIES = [
    "dijkstra",
    "reliable"
]

# 待機戦略とそのパラメータ範囲の定義 (run_all_topology_simulations.py から流用)
STRATEGIES_WITH_PARAMS = {
    "dynamic_wait_strategy_with_faillink": ("dynamic_factor", np.arange(0.5, 3.1, 0.5)),
    "dynamic_wait_strategy_with_node_count": ("dynamic_factor", np.arange(0.5, 3.1, 0.5)),
    "ratio_based_wait_strategy": ("ratio_factor", np.arange(0.5, 3.1, 0.5)),
    "fixed_wait_duration_strategy": ("base_wait_time", np.arange(0.5, 5.1, 0.5)),
    "no_wait_strategy": (None, [0]), # パラメータなし
    "infinite_wait_strategy": (None, [0]), # パラメータなし
}

# 戦略名の短縮形 (ディレクトリ名・ファイル名用)
STRATEGY_ABBREVIATIONS = {
    "dynamic_wait_strategy_with_faillink": "dynfail",
    "dynamic_wait_strategy_with_node_count": "dynnode",
    "ratio_based_wait_strategy": "ratio",
    "fixed_wait_duration_strategy": "fixed",
    "no_wait_strategy": "nowait",
    "infinite_wait_strategy": "infwait",
}

# ルーティング戦略の短縮形
ROUTING_STRATEGY_ABBREVIATIONS = {
    "dijkstra": "dijk",
    "reliable": "reli",
}

# --- ヘルパー関数 ---
def run_single_simulation_task(task_info):
    """
    単一のシミュレーションタスクを実行するワーカー関数。
    ProcessPoolExecutorによって呼び出される。
    """
    topo = task_info["topo"]
    graph_idx = task_info["graph_idx"]
    routing = task_info["routing"]
    wait = task_info["wait"]
    param_type = task_info["param_type"]
    param_value = task_info["param_value"]
    node_file = task_info["node_file"]
    edge_file = task_info["edge_file"]
    output_dir = task_info["output_dir"]
    summary_filename = task_info["summary_filename"]
    task_id = task_info["task_id"]
    total_tasks = task_info["total_tasks"]

    start_time = time.time()
    
    # nwsim.py を実行するコマンドをリストとして組み立て
    command = [
        sys.executable,
        "nwsim.py",
        "--node_file", node_file,
        "--edge_file", edge_file,
        "--packets", SIMULATION_BASE_PARAMS["packets"],
        "--max_sim_time", SIMULATION_BASE_PARAMS["max_sim_time"],
        "--ttl", SIMULATION_BASE_PARAMS["ttl"],
        "--buffer_size", SIMULATION_BASE_PARAMS["buffer_size"],
        "--routing_strategy", routing,
        "--strategy", wait,
        "--output_base_dir", output_dir,
        "--summary_filename", summary_filename
    ]

    # 待機戦略のパラメータを追加
    if param_type:
        command.extend([f"--{param_type}", str(param_value)])

    try:
        # コマンドを実行し、出力をキャプチャ
        result = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[Task {task_id}/{total_tasks}] SUCCESS in {execution_time:.2f}s: {topo}/graph_{graph_idx}/{routing}/{wait}/{param_type}={param_value}")
        return True, None
    except subprocess.CalledProcessError as e:
        error_message = (
            f"[Task {task_id}/{total_tasks}] FAILED: {topo}/graph_{graph_idx}/{routing}/{wait}/{param_type}={param_value}\n"
            f"  Return Code: {e.returncode}\n"
            f"  Output:\n{e.stdout}\n{e.stderr}\n"
        )
        print(error_message)
        return False, error_message
    except FileNotFoundError:
        error_message = f"Error: '{sys.executable}' command not found. Make sure Python is in your PATH."
        print(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred for task {task_id}: {e}"
        print(error_message)
        return False, error_message


def main():
    # --- 引数の設定 ---
    parser = argparse.ArgumentParser(description="Run a full experiment by generating multiple graphs and running sweeps on them.")
    
    # このスクリプト自身が使用する引数
    parser.add_argument("--num_graphs", type=int, required=True, help="Number of random graphs to generate for each topology.")
    parser.add_argument("--topology_type", type=str, default="all", help="Topology type to generate. 'all' for every supported topology.")
    
    # make_network_for_nwsim.py に渡すための引数 (unknown_argsとして取得)
    args, unknown_args = parser.parse_known_args()

    # --- トポロジの定義 ---
    all_supported_topologies = [
        "random", 
        "grid", 
        "barabasi_albert", 
        "path", 
        "ring", 
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
        print(f"Available topologies: {', '.join(all_supported_topologies)}")
        sys.exit(1)

    # --- ベースディレクトリの定義と作成 ---
    BASE_GRAPH_DIR = "graph"
    BASE_RESULT_DIR = "result"

    os.makedirs(BASE_GRAPH_DIR, exist_ok=True)
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    
    print(f"Graph data will be saved in: {BASE_GRAPH_DIR}")
    print(f"Simulation results will be saved in: {BASE_RESULT_DIR}")

    all_simulation_tasks = []
    task_counter = 0

    # --- PHASE 1: グラフ生成とシミュレーションタスクの集約 ---
    print("\n" + "=" * 80)
    print("PHASE 1: Generating Graphs and Aggregating Simulation Tasks")
    print("=" * 80)

    for topo in target_topologies:
        print("\n" + "-" * 80)
        print(f"Processing Topology: {topo}")
        print("-" * 80)
        
        # トポロジごとのグラフ保存ディレクトリ
        topo_graph_dir = os.path.join(BASE_GRAPH_DIR, topo)
        os.makedirs(topo_graph_dir, exist_ok=True)

        for i in range(args.num_graphs):
            print(f"  Generating Graph {i+1}/{args.num_graphs} for {topo}...")
            
            # グラフファイル保存用のユニークなディレクトリ
            graph_instance_dir = os.path.join(topo_graph_dir, f"graph_{i}")
            os.makedirs(graph_instance_dir, exist_ok=True)
            
            node_path = os.path.join(graph_instance_dir, "node.csv")
            edge_path = os.path.join(graph_instance_dir, "edge.csv")
            
            # --- グラフ生成コマンドの組み立てと実行 ---
            make_cmd = [
                sys.executable, "make_network_for_nwsim.py",
                "--topology_type", topo,
                "--node_output_path", node_path,
                "--edge_output_path", edge_path,
            ] + unknown_args # make_network_for_nwsim.py 向けの他の引数を追加

            try:
                subprocess.run(make_cmd, check=True, text=True, encoding='utf-8', capture_output=True)
                print(f"    Successfully generated graph files in: {graph_instance_dir}")
            except subprocess.CalledProcessError as e:
                print(f"    ERROR: Graph generation failed for {topo}, graph {i}.")
                print(f"      Command: {' '.join(e.cmd)}")
                print(f"      Return Code: {e.returncode}")
                print(f"      Output:\n{e.stdout}\n{e.stderr}")
                continue # このグラフでのシミュレーションタスクは生成しない
            except FileNotFoundError:
                print("    ERROR: 'make_network_for_nwsim.py' not found. Make sure it's in the same directory.")
                sys.exit(1)

            # --- シミュレーションタスクの集約 ---
            for routing in ROUTING_STRATEGIES:
                routing_abbr = ROUTING_STRATEGY_ABBREVIATIONS[routing]
                
                for wait_strategy_name, (param_type, param_values) in STRATEGIES_WITH_PARAMS.items():
                    wait_abbr = STRATEGY_ABBREVIATIONS[wait_strategy_name]

                    for param_value in param_values:
                        # 結果保存ディレクトリのパスを組み立て
                        result_output_dir = os.path.join(
                            BASE_RESULT_DIR,
                            topo,
                            f"graph_{i}",
                            routing_abbr,
                            wait_abbr
                        )
                        os.makedirs(result_output_dir, exist_ok=True)

                        # サマリーファイル名を組み立て (パラメータ値を含む)
                        param_str = f"_p{param_value:.1f}" if param_type else ""
                        summary_filename = f"summary_{routing_abbr}_{wait_abbr}{param_str}.csv"

                        task_counter += 1
                        all_simulation_tasks.append({
                            "topo": topo,
                            "graph_idx": i,
                            "routing": routing,
                            "wait": wait_strategy_name,
                            "param_type": param_type,
                            "param_value": param_value,
                            "node_file": node_path,
                            "edge_file": edge_path,
                            "output_dir": result_output_dir,
                            "summary_filename": summary_filename,
                            "task_id": task_counter,
                            "total_tasks": None # 後で設定
                        })
    
    if not all_simulation_tasks:
        print("No simulation tasks were generated. Exiting.")
        sys.exit(0)

    # total_tasks を設定
    for task in all_simulation_tasks:
        task["total_tasks"] = len(all_simulation_tasks)

    print(f"\nAggregated a total of {len(all_simulation_tasks)} simulation tasks.")

    # --- PHASE 2: シミュレーションタスクの並列実行 ---
    print("\n" + "=" * 80)
    print("PHASE 2: Running All Simulation Tasks in Parallel")
    print("=" * 80)

    workers = MAX_WORKERS if MAX_WORKERS is not None else os.cpu_count()
    print(f"Using {workers} worker processes.")

    success_count = 0
    failure_count = 0
    failed_tasks_info = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # タスクを投入
        futures = [executor.submit(run_single_simulation_task, task) for task in all_simulation_tasks]

        # 完了した順に結果を処理
        for future in concurrent.futures.as_completed(futures):
            try:
                success, error_message = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    if error_message:
                        failed_tasks_info.append(error_message)
            except Exception as exc:
                failure_count += 1
                error_info = f"An unexpected error occurred during task execution: {exc}"
                print(error_info)
                failed_tasks_info.append(error_info)

    print("\n" + "=" * 80)
    print("All Simulation Tasks Completed.")
    print("=" * 80)
    print(f"Total tasks: {len(all_simulation_tasks)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failure_count}")
    
    if failed_tasks_info:
        print("\n--- FAILED TASKS SUMMARY ---")
        for info in failed_tasks_info:
            print(info)

    print(f"\nGraph data saved in: {BASE_GRAPH_DIR}")
    print(f"Simulation results saved in: {BASE_RESULT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
