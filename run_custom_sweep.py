import multiprocessing
import subprocess
import time
import os
import shutil
import sys
import numpy as np
from datetime import datetime
import argparse

# --- Parameters from the specification ---

# Program to run
TARGET_SCRIPT = "nwsim.py"  # 前のパートで作成した v2 のファイル名に変更してください (例: nwsim_numba_final_v2.py)

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

# Strategy name abbreviations for filenames
STRATEGY_ABBREVIATIONS = {
    "dynamic_wait_strategy_with_faillink": "dynfail",
    "dynamic_wait_strategy_with_node_count": "dynnode",
    "ratio_based_wait_strategy": "ratio",
    "fixed_wait_duration_strategy": "fixed",
    "no_wait_strategy": "nowait",
    "infinite_wait_strategy": "infwait",
}


# --- Execution settings ---

# Parallelism
NUM_PROCESSES = 6
NUM_LOGICAL_CORES = os.cpu_count() if hasattr(os, "cpu_count") else "N/A"


def run_single_simulation(task_info):
    """
    Runs a single simulation task based on the new directory and file naming conventions.
    """
    # Unpack task info
    strategy = task_info["strategy"]
    connect_rate = task_info["connect_rate"]
    disconnect_rate = task_info["disconnect_rate"]
    param_type = task_info["param_type"]
    param_value = task_info["param_value"]
    task_id = task_info["task_id"]
    node_file = task_info["node_file"]
    edge_file = task_info["edge_file"]
    src_node = task_info["src_node"]
    dst_node = task_info["dst_node"]
    root_output_dir = task_info["root_output_dir"]
    # --- 変更点: 新しいパラメータをアンパック ---
    buffer_size = task_info["buffer_size"]
    ttl = task_info["ttl"]

    # --- 1. Determine directory and file names based on the new plan ---

    strategy_abbr = STRATEGY_ABBREVIATIONS.get(strategy, "unknown")

    if param_type:
        param_dir_str = f"{param_value:.1f}"
        param_file_str = f"_sp-{param_value:.1f}"
    else:
        param_dir_str = "no_params"
        param_file_str = ""

    # --- 変更点: ディレクトリ構造から buffer_size と ttl を削除 ---
    # {root}/{strategy}/{param}/c{...}/
    output_dir = os.path.join(
        root_output_dir,
        strategy,
        param_dir_str,
        f"c{connect_rate:.1f}",
    )
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(root_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    yyyymmdd = datetime.now().strftime("%Y%m%d")

    # --- 変更点: ファイル名から buffer_size と ttl を削除 ---
    # summary_{...}_{abbr}{...}.csv
    base_csv_name = (
        f"summary_{yyyymmdd}_c{connect_rate:.1f}_d{disconnect_rate:.1f}"
        f"_{strategy_abbr}{param_file_str}_{task_id}"
    )
    summary_filename = f"{base_csv_name}.csv"

    log_base_name = (
        f"log_c{connect_rate:.1f}_d{disconnect_rate:.1f}"
        f"_{strategy_abbr}{param_file_str}_{task_id}"
    )
    log_filename = f"{log_base_name}.txt"
    log_file_path = os.path.join(log_dir, log_filename)

    # --- 2. Construct the command for nwsim.py ---

    command = [
        sys.executable,
        TARGET_SCRIPT,
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
        "--connect_rate",
        str(connect_rate),
        "--disconnect_rate",
        str(disconnect_rate),
        "--packets",
        "2000",
        "--output_base_dir",
        output_dir,
        "--summary_filename",
        summary_filename,
        # --- 変更点: 新しい引数をコマンドに追加 ---
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

    print(
        f"[Task {task_id}] Starting: {strategy} c={connect_rate:.1f} d={disconnect_rate:.1f} param={param_dir_str} (Log: {log_file_path})"
    )
    start_time = time.time()

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        try:
            process = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                encoding="utf-8",
            )
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[Task {task_id}] Finished in {execution_time:.2f}s.")
            return (task_id, True, execution_time, "")
        except subprocess.CalledProcessError as e:
            error_message = f"[Task {task_id}] FAILED. Check log file: {log_file_path}\nReturn Code: {e.returncode}"
            print(error_message)
            return (task_id, False, 0, error_message)
        except Exception as e:
            error_message = f"[Task {task_id}] FAILED with unexpected error. Check log file: {log_file_path}\nError: {e}"
            print(error_message)
            return (task_id, False, 0, error_message)



def main(args):
    # --- Setup ---
    print(f"--- Starting Custom Simulation Sweep ---")
    print(f"Root results directory: {args.output_dir}")
    print(f"Detailed logs will be saved in: {os.path.join(args.output_dir, 'logs')}")

    print(f"Network files: {args.node_file}, {args.edge_file}")
    print(f"Source: {args.source_node}, Destination: {args.dest_node}")
    print(f"Buffer Size: {args.buffer_size}, TTL: {args.ttl}")
    print(f"Parallel processes: {NUM_PROCESSES} (Logical cores: {NUM_LOGICAL_CORES})")

    # --- Prepare all simulation tasks ---
    tasks = []
    task_id_counter = 0
    for con_rate in RATE_SWEEP_VALUES:
        for dis_rate in RATE_SWEEP_VALUES:
            for strategy_name, (
                param_type,
                param_range,
            ) in STRATEGIES_WITH_PARAMS.items():
                for param_value in param_range:
                    task = {
                        "strategy": strategy_name,
                        "connect_rate": con_rate,
                        "disconnect_rate": dis_rate,
                        "param_type": param_type,
                        "param_value": param_value,
                        "task_id": task_id_counter,
                        "node_file": args.node_file,
                        "edge_file": args.edge_file,
                        "src_node": args.source_node,
                        "dst_node": args.dest_node,
                        "root_output_dir": args.output_dir,
                        # --- 変更点: 新しいパラメータをタスク辞書に追加 ---
                        "buffer_size": args.buffer_size,
                        "ttl": args.ttl,
                    }
                    tasks.append(task)
                    task_id_counter += 1

    print(f"Total simulation tasks to run: {len(tasks)}")
    if len(tasks) == 0:
        print("No tasks to run. Exiting.")
        return

    # --- Execute simulations in parallel ---
    start_total_time = time.time()
    results = []

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        result_iterator = pool.imap_unordered(run_single_simulation, tasks)

        print("\n--- Running simulations ---")
        for i, result in enumerate(result_iterator):
            results.append(result)
            progress = (i + 1) / len(tasks) * 100
            print(
                f"Progress: {i + 1}/{len(tasks)} ({progress:.2f}%) tasks completed.",
                end="\r",
            )

    print("\n")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    # --- Process results ---
    print("--- All Simulations Completed ---")

    successful_runs = sum(1 for r in results if r[1])
    failed_runs = len(results) - successful_runs

    print(f"Total execution time: {total_duration:.2f} seconds")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")

    if failed_runs > 0:
        print("\n--- FAILED RUNS LOG ---")
        for r in results:
            if not r[1]:
                print(r[3])

    print(f"\nAll simulation results are stored in the '{args.output_dir}' directory.")


if __name__ == "__main__":
    # --- 変更点: argparse を使用してコマンドライン引数を処理 ---
    parser = argparse.ArgumentParser(description="Run a sweep of network simulations.")
    parser.add_argument("node_file", type=str, help="Path to the node CSV file.")
    parser.add_argument("edge_file", type=str, help="Path to the edge CSV file.")
    parser.add_argument("source_node", type=int, help="ID of the source node.")
    parser.add_argument("dest_node", type=int, help="ID of the destination node.")
    parser.add_argument(
        "output_dir", type=str, help="Root directory to save simulation results."
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10,
        help="Buffer size for each node in the simulation. -1 for infinite.",
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=100.0,
        help="Time To Live for packets. Use a large number for effectively infinite.",
    )

    args = parser.parse_args()

    # --- 変更点: TARGET_SCRIPTのファイル名を確認 ---
    # `nwsim.py` が古いバージョンの可能性があるため、新しいファイル名を使うように促す
    if TARGET_SCRIPT == "nwsim.py":
        print("Warning: TARGET_SCRIPT is set to 'nwsim.py'.")
        print(
            "Please ensure this is the new version with buffer/TTL support, or update the script to the new filename (e.g., 'nwsim_numba_final_v2.py')."
        )

    main(args)
