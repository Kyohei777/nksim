# nwsim_numba_final.py (パーツ1/4 の最終確定・修正版)

import networkx as nx
import csv
import numpy as np
import heapq
from datetime import datetime
import random
import traceback
import os
import argparse
import sys
from functools import partial
import time

# ★★★ NUMBA ★★★
import numba
from numba.core import types
from numba.typed import Dict, List

# --- グローバル変数 ---
LINK_STATE_RNG_LEGACY = None
PACKET_GEN_RNG = None
insertion_counter = 0
RATIO_STRATEGY_CACHE = {}
DEBUG_LOG_WRITER = None
DEBUG_LOG_FILE_HANDLE = None
TARGET_DEBUG_PACKET_IDS = []

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここからがNumba化による最終高速化 (エラー最終修正済み) ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# Numbaが理解できるように、隣接リストの値の型（タプルのリスト）を事前に定義
NEIGHBOR_TUPLE_TYPE = types.Tuple((types.int64, types.float64))
NEIGHBOR_LIST_TYPE = types.ListType(NEIGHBOR_TUPLE_TYPE)


@numba.njit
def _bfs_has_path_numba(start_node, end_node, adj, paththrough_nodes, broken_links):
    """Numbaでコンパイルされる、超高速な幅優先探索（BFS）"""
    if start_node not in adj:
        return False

    q = List()
    q.append(start_node)

    visited = Dict.empty(key_type=types.int64, value_type=types.uint8)
    visited[start_node] = np.uint8(1)

    paththrough_set = set(paththrough_nodes)

    broken_set = set()
    for i in range(len(broken_links)):
        link = broken_links[i]
        u, v = link[0], link[1]
        if u < v:
            broken_set.add((u, v))
        else:
            broken_set.add((v, u))

    while len(q) > 0:
        current_node = q.pop(0)
        if current_node == end_node:
            return True

        # //////////////////////////////////////////////////////////////////
        # //                  ★★★★★ ここがエラーの修正箇所 ★★★★★                  //
        # //////////////////////////////////////////////////////////////////
        empty_list = List.empty_list(NEIGHBOR_TUPLE_TYPE)
        for neighbor, weight in adj.get(current_node, empty_list):
            if neighbor not in visited:
                if neighbor in paththrough_set and neighbor != end_node:
                    continue

                u, v = current_node, neighbor
                if u > v:
                    u, v = v, u
                if (u, v) in broken_set:
                    continue

                visited[neighbor] = np.uint8(1)
                q.append(neighbor)
    return False


@numba.njit
def _dijkstra_get_next_hop_numba(
    start_node, end_node, adj, paththrough_nodes, broken_links
):
    """Numbaでコンパイルされた、超高速なダイクストラ法"""
    if start_node not in adj:
        return -1

    paththrough_set = set(paththrough_nodes)

    broken_set = set()
    for i in range(len(broken_links)):
        link = broken_links[i]
        u, v = link[0], link[1]
        if u < v:
            broken_set.add((u, v))
        else:
            broken_set.add((v, u))

    dist = Dict.empty(key_type=types.int64, value_type=types.float64)
    for node in adj:
        dist[node] = np.inf
    if start_node not in dist:
        return -1
    dist[start_node] = 0

    prev = Dict.empty(key_type=types.int64, value_type=types.int64)
    for node in adj:
        prev[node] = -1

    pq = [(0.0, start_node)]

    while len(pq) > 0:
        d, current_node = heapq.heappop(pq)
        if current_node not in dist or d > dist[current_node]:
            continue
        if current_node == end_node:
            break

        # //////////////////////////////////////////////////////////////////
        # //                  ★★★★★ ここがエラーの修正箇所 ★★★★★                  //
        # //////////////////////////////////////////////////////////////////
        empty_list = List.empty_list(NEIGHBOR_TUPLE_TYPE)
        for neighbor, weight in adj.get(current_node, empty_list):
            if neighbor in paththrough_set and neighbor != end_node:
                continue

            u, v = current_node, neighbor
            if u > v:
                u, v = v, u
            if (u, v) in broken_set:
                continue

            if neighbor not in dist:
                continue
            new_dist = dist[current_node] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    if prev.get(end_node, -1) == -1:
        return -1

    path_node = end_node
    for _ in range(len(adj)):
        prev_node = prev.get(path_node, -1)
        if prev_node == -1:
            return -1
        if prev_node == start_node:
            return path_node
        path_node = prev_node
    return -1


# --- クラス定義 ---
class Packet:
    def __init__(self, src, dst, time, wait_duration_func):
        self.src, self.dst, self.start_time = src, dst, time
        self.present_node, self.present_link = src, None
        self.end_time, self.movecount, self.failed_link_attempts = 0, 0, 0
        self.paththrough_node, self.brokenlink = [], []
        self.movetime = time
        self.is_arrived = False
        self.wait_max_duration_func = wait_duration_func
        self.packet_id = -1


class Edge:
    # ... (変更なし)
    def __init__(
        self, source, target, weight, name, initial_time, connect_rate, disconnect_rate
    ):
        self.source, self.target, self.weight, self.name = source, target, weight, name
        self.is_connected, self.next_event_time = True, 0
        self.is_transmitting_until, self.current_transmitting_packet_id = 0.0, None
        self.initialize(initial_time, connect_rate, disconnect_rate)

    def initialize(self, current_time, connect_rate, disconnect_rate):
        global LINK_STATE_RNG_LEGACY
        if LINK_STATE_RNG_LEGACY is None:
            LINK_STATE_RNG_LEGACY = random.Random()
        self.is_connected = LINK_STATE_RNG_LEGACY.choice([True, False])
        rate = disconnect_rate if self.is_connected else connect_rate
        self.next_event_time = current_time + LINK_STATE_RNG_LEGACY.expovariate(rate)

    def update_status(self, current_time, connect_rate, disconnect_rate):
        global LINK_STATE_RNG
        if current_time >= self.next_event_time:
            self.is_connected = not self.is_connected
            rate = disconnect_rate if self.is_connected else connect_rate
            self.next_event_time = current_time + LINK_STATE_RNG.expovariate(rate)
            if self.is_connected != self.is_connected and TARGET_DEBUG_PACKET_IDS:
                log_packet_event(
                    current_time,
                    -1,
                    "EDGE_STATUS_CHANGE",
                    f"Edge={self.name}({self.source}-{self.target}) from {not self.is_connected} to {self.is_connected}",
                )


class Graph:
    def __init__(
        self,
        node_file_name="node.csv",
        edge_file_name="edge.csv",
        connect_rate=0.4,
        disconnect_rate=0.2,
    ):
        self.G = nx.Graph()
        self.node_file_name, self.edge_file_name = node_file_name, edge_file_name
        self.connect_rate, self.disconnect_rate = connect_rate, disconnect_rate
        self.edges, self.unique_edges_list = {}, []
        self.adj = Dict.empty(key_type=types.int64, value_type=NEIGHBOR_LIST_TYPE)
        if os.path.exists(self.node_file_name) and os.path.exists(self.edge_file_name):
            self.make_graph()
            self._build_adjacency_list_for_numba()

    def make_graph(self):
        try:
            with open(self.node_file_name, "r", newline="") as f:
                for row in csv.DictReader(f):
                    self.G.add_node(int(row["node_id"]), label=row["label"])
        except Exception as e:
            print(f"ERROR reading node file: {e}")
        try:
            with open(self.edge_file_name, "r", newline="") as f:
                processed_pairs = set()
                for row in csv.DictReader(f):
                    source, target = int(row["source"]), int(row["target"])
                    pair = tuple(sorted((source, target)))
                    if pair not in processed_pairs:
                        edge = Edge(
                            source,
                            target,
                            float(row["weight"]),
                            row["name"],
                            0,
                            self.connect_rate,
                            self.disconnect_rate,
                        )
                        self.unique_edges_list.append(edge)
                        self.edges[(source, target)] = self.edges[(target, source)] = (
                            edge
                        )
                        self.G.add_edge(
                            source,
                            target,
                            weight=float(row["weight"]),
                            name=row["name"],
                        )
                        processed_pairs.add(pair)
        except Exception as e:
            print(f"ERROR reading edge file: {e}")

    def _build_adjacency_list_for_numba(self):
        temp_adj = {node: [] for node in self.G.nodes()}
        for u, v, data in self.G.edges(data=True):
            temp_adj[u].append((v, data["weight"]))
            temp_adj[v].append((u, data["weight"]))
        for node, neighbors in temp_adj.items():
            neighbors.sort()
            numba_neighbors = List.empty_list(NEIGHBOR_TUPLE_TYPE)
            for neighbor, weight in neighbors:
                numba_neighbors.append((neighbor, weight))
            self.adj[node] = numba_neighbors

    def find_route_for_packet(self, packet):
        paththrough_nodes = np.array(
            list(set(packet.paththrough_node) - {packet.present_node}), dtype=np.int64
        )
        broken_links = np.array(packet.brokenlink, dtype=np.int64).reshape(-1, 2)
        next_hop = _dijkstra_get_next_hop_numba(
            packet.present_node, packet.dst, self.adj, paththrough_nodes, broken_links
        )
        return next_hop if next_hop != -1 else None

    def get_viable_next_hops(self, packet):
        viable_next_nodes_map = {}
        present_node = packet.present_node
        paththrough_nodes = np.array(
            list(set(packet.paththrough_node + [present_node])), dtype=np.int64
        )
        broken_links = np.array(packet.brokenlink, dtype=np.int64).reshape(-1, 2)

        empty_neighbor_list = List.empty_list(NEIGHBOR_TUPLE_TYPE)
        for neighbor, weight in self.adj.get(present_node, empty_neighbor_list):
            paththrough_set = set(packet.paththrough_node)
            broken_set = {tuple(sorted(link)) for link in packet.brokenlink}
            if neighbor in paththrough_set:
                continue
            if tuple(sorted((present_node, neighbor))) in broken_set:
                continue
            if (
                _bfs_has_path_numba(
                    neighbor, packet.dst, self.adj, paththrough_nodes, broken_links
                )
                or neighbor == packet.dst
            ):
                viable_next_nodes_map[neighbor] = True
        return list(viable_next_nodes_map.keys()), len(viable_next_nodes_map)


# --- ログ記録関数 (ベースライン版と全く同じ) ---
def log_packet_event(time, packet_id, event_type, details=""):
    """指定されたパケットIDのイベントをCSVログファイルに記録する"""
    if DEBUG_LOG_WRITER and (packet_id in TARGET_DEBUG_PACKET_IDS or packet_id == -1):
        try:
            DEBUG_LOG_WRITER.writerow(
                [
                    f"{time:.4f}",
                    packet_id if packet_id != -1 else "SYSTEM",
                    event_type,
                    details,
                ]
            )
            if DEBUG_LOG_FILE_HANDLE:
                DEBUG_LOG_FILE_HANDLE.flush()
        except Exception as e:
            print(f"ERROR writing to debug log file: {e}", file=sys.stderr)


# --- 待機時間決定関数 (ベースライン版と全く同じ) ---
def fixed_wait_duration_strategy(packet, graph, link_to_wait_for, base_wait_time=5.0):
    return base_wait_time


def no_wait_strategy(packet, graph, link_to_wait_for):
    return 0.0


def dynamic_wait_strategy_with_faillink(
    packet, graph, link_to_wait_for, dynamic_factor=0.5
):
    return dynamic_factor * packet.failed_link_attempts


def infinite_wait_strategy(packet, graph, link_to_wait_for):
    return 999999.0


def dynamic_wait_strategy_with_node_count(
    packet, graph, link_to_wait_for, dynamic_factor=0.5
):
    return dynamic_factor * packet.movecount


def ratio_based_wait_strategy(packet, ideal_graph, link_to_wait_for, ratio_factor=1.0):
    global RATIO_STRATEGY_CACHE
    source_node, dest_node = packet.present_node, packet.dst
    cache_key = (
        source_node,
        dest_node,
        frozenset(packet.brokenlink),
        frozenset(packet.paththrough_node),
        link_to_wait_for,
    )
    if cache_key in RATIO_STRATEGY_CACHE:
        return RATIO_STRATEGY_CACHE[cache_key]

    base_graph = ideal_graph.copy()
    base_graph.remove_nodes_from(
        [n for n in packet.paththrough_node if n != source_node]
    )
    for lnk in packet.brokenlink:
        if base_graph.has_edge(*lnk):
            base_graph.remove_edge(*lnk)
    try:
        original_length = nx.shortest_path_length(
            base_graph, source=source_node, target=dest_node, weight="weight"
        )
    except nx.NetworkXNoPath:
        RATIO_STRATEGY_CACHE[cache_key] = 0.0
        return 0.0
    edge_to_remove = link_to_wait_for
    if base_graph.has_edge(*edge_to_remove):
        edge_data = base_graph.get_edge_data(*edge_to_remove)
        base_graph.remove_edge(*edge_to_remove)
        try:
            detour_length = nx.shortest_path_length(
                base_graph, source=source_node, target=dest_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            detour_length = float("inf")
        finally:
            base_graph.add_edge(*edge_to_remove, **edge_data)
    else:
        detour_length = original_length
    if detour_length == float("inf"):
        result = 999999.0
    else:
        result = (
            (detour_length / original_length) * ratio_factor
            if original_length > 0
            else 0.0
        )
    RATIO_STRATEGY_CACHE[cache_key] = result
    return result


# --- パケット生成とサマリー書き出し (ベースライン版と全く同じ) ---
def generate_packets(
    lambd, size, number_of_nodes, wait_func, src_node=0, dst_node=None
):
    global PACKET_GEN_RNG
    if PACKET_GEN_RNG is None:
        PACKET_GEN_RNG = np.random.default_rng()
    if number_of_nodes == 0:
        return []

    arrival_intervals = PACKET_GEN_RNG.exponential(1 / lambd, size)
    arrival_times = np.cumsum(arrival_intervals)

    packet_list = []
    actual_dst = dst_node if dst_node is not None else number_of_nodes - 1

    for i in range(size):
        packet = Packet(src_node, actual_dst, arrival_times[i], wait_func)
        packet.packet_id = i
        packet_list.append(packet)
    return packet_list


def write_final_packet_summary(final_packets, filename):
    try:
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            header = [
                "Packet ID",
                "Source",
                "Destination",
                "Start Time",
                "End Time",
                "Move Count",
                "Failed Attempts",
                "Path",
                "Arrived",
                "Broken Links (Final)",
            ]
            writer.writerow(header)

            for p in final_packets:
                arrived = p.present_node == p.dst

                path_nodes = [str(p.src)] + [str(node) for node in p.paththrough_node]
                if arrived:
                    if not path_nodes or path_nodes[-1] != str(p.dst):
                        path_nodes.append(str(p.dst))
                path_str = ",".join(path_nodes)

                broken_links_str = ";".join([f"({u},{v})" for u, v in p.brokenlink])

                writer.writerow(
                    [
                        p.packet_id,
                        p.src,
                        p.dst,
                        f"{p.start_time:.4f}",
                        f"{p.end_time:.4f}",
                        p.movecount,
                        p.failed_link_attempts,
                        path_str,
                        arrived,
                        broken_links_str,
                    ]
                )
    except Exception as e:
        print(f"ERROR writing summary file {filename}: {e}")


# --- シミュレーション本体 ---
def simulate_network(
    node_file,
    edge_file,
    lambd,
    packet_size,
    wait_strategy_func,
    ideal_graph,
    connect_rate,
    disconnect_rate,
    base_wait_time,
    dynamic_factor,
    ratio_factor,
    max_simulation_time=float("inf"),
    output_base_dir="simulation_results",
    src_node=0,
    dst_node=None,
    progress_interval_time=1.0,
    debug_packet_ids_str="",
    debug_log_filename="packet_trace_log.csv",
    run_identifier="",
):
    global insertion_counter, DEBUG_LOG_WRITER, DEBUG_LOG_FILE_HANDLE, TARGET_DEBUG_PACKET_IDS

    strategy_name = wait_strategy_func.__name__
    if "dynamic" in strategy_name:
        wait_func_with_args = partial(wait_strategy_func, dynamic_factor=dynamic_factor)
    elif "fixed" in strategy_name:
        wait_func_with_args = partial(wait_strategy_func, base_wait_time=base_wait_time)
    elif "ratio" in strategy_name:
        wait_func_with_args = partial(wait_strategy_func, ratio_factor=ratio_factor)
    else:
        wait_func_with_args = wait_strategy_func

    sim_real_start_time = datetime.now()
    # The output_base_dir is now the final, deep directory path from the sweep script
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)

    if debug_packet_ids_str:
        try:
            TARGET_DEBUG_PACKET_IDS = [
                int(pid.strip()) for pid in debug_packet_ids_str.split(",")
            ]
            log_file_path = os.path.join(output_dir, debug_log_filename)
            DEBUG_LOG_FILE_HANDLE = open(
                log_file_path, "w", newline="", encoding="utf-8"
            )
            DEBUG_LOG_WRITER = csv.writer(DEBUG_LOG_FILE_HANDLE)
            DEBUG_LOG_WRITER.writerow(["Time", "PacketID", "EventType", "Details"])
        except (ValueError, IOError) as e:
            print(
                f"WARNING: Could not create/write debug log '{log_file_path}': {e}.",
                file=sys.stderr,
            )
            TARGET_DEBUG_PACKET_IDS, DEBUG_LOG_FILE_HANDLE, DEBUG_LOG_WRITER = (
                [],
                None,
                None,
            )
    else:
        TARGET_DEBUG_PACKET_IDS = []

    graph = Graph(
        node_file, edge_file, connect_rate=connect_rate, disconnect_rate=disconnect_rate
    )
    if not graph.G:
        return [], "."

    actual_dst_node = (
        dst_node if dst_node is not None else (graph.G.number_of_nodes() - 1)
    )

    all_packets = generate_packets(
        lambd, packet_size, graph.G.number_of_nodes(), None, src_node, actual_dst_node
    )
    for p in all_packets:
        p.wait_max_duration_func = wait_func_with_args

    active_packets = []
    insertion_counter = 0
    for p in all_packets:
        heapq.heappush(active_packets, (p.movetime, insertion_counter, p))
        insertion_counter += 1

    current_time = 0.0
    processed_packets = 0
    simulation_processing_start_real_time = datetime.now()
    last_progress_print_time_real = sim_real_start_time

    while (
        active_packets
        and processed_packets < len(all_packets)
        and current_time < max_simulation_time
    ):
        _, _, packet = heapq.heappop(active_packets)
        if packet.is_arrived:
            continue

        current_time = packet.movetime

        current_real_time_for_progress = datetime.now()
        if (
            current_real_time_for_progress - last_progress_print_time_real
        ).total_seconds() >= progress_interval_time:
            percentage_done = (
                (processed_packets / packet_size * 100) if packet_size > 0 else 0.0
            )
            elapsed_s = (
                current_real_time_for_progress - simulation_processing_start_real_time
            ).total_seconds()
            eta_str = "N/A"
            if processed_packets > 0 and percentage_done > 0.1:
                total_est_s = (elapsed_s / percentage_done) * 100
                rem_s = total_est_s - elapsed_s
                if rem_s < 0:
                    rem_s = 0
                eta_m, eta_s_rem = divmod(rem_s, 60)
                eta_h, eta_m_rem = divmod(eta_m, 60)
                eta_str = f"{int(eta_h):02d}h{int(eta_m_rem):02d}m{int(eta_s_rem):02d}s"
            bar_len = 25
            fill_len = int(
                bar_len * processed_packets // packet_size if packet_size > 0 else 0
            )
            bar = "#" * fill_len + "-" * (bar_len - fill_len)
            progress_line = f"PROGRESS: SimTime={current_time:<8.2f} |{bar}| {processed_packets:>5}/{packet_size} ({percentage_done:>5.1f}%) ETA: {eta_str:<12s}"
            try:
                term_width = os.get_terminal_size().columns
                print(" " * (term_width - 1), end="\r")
                print(progress_line, end="\r")
            except OSError:
                print(progress_line, end="\r")
            sys.stdout.flush()
            last_progress_print_time_real = current_real_time_for_progress

        for edge in graph.unique_edges_list:
            edge.update_status(current_time, connect_rate, disconnect_rate)

        log_packet_event(
            current_time,
            packet.packet_id,
            "PROC_START",
            f"At={packet.present_node}, Dst={packet.dst}",
        )

        if packet.present_node == packet.dst:
            packet.is_arrived, packet.end_time, processed_packets = (
                True,
                current_time,
                processed_packets + 1,
            )
            log_packet_event(
                current_time, packet.packet_id, "PACKET_ARRIVED", f"Dest={packet.dst}"
            )
            continue

        viable_hops, num_viable = graph.get_viable_next_hops(packet)
        chosen_hop = None
        if num_viable >= 1:
            chosen_hop = graph.find_route_for_packet(packet)

        if chosen_hop is None:
            packet.is_arrived, packet.end_time, processed_packets = (
                True,
                current_time,
                processed_packets + 1,
            )
            log_packet_event(
                current_time, packet.packet_id, "NO_ROUTE_FAIL", "No viable/chosen hop."
            )
        else:
            link_tpl = tuple(sorted((packet.present_node, chosen_hop)))
            edge = graph.edges.get(link_tpl)
            log_packet_event(
                current_time, packet.packet_id, "ROUTE_CHOSEN", f"NextHop={chosen_hop}"
            )

            if (
                edge
                and edge.is_connected
                and current_time >= edge.is_transmitting_until
            ):
                edge.is_transmitting_until, packet.movetime = (
                    current_time + edge.weight,
                    current_time + edge.weight,
                )
                packet.paththrough_node.append(packet.present_node)
                packet.present_node, packet.movecount, packet.brokenlink = (
                    chosen_hop,
                    packet.movecount + 1,
                    [],
                )
                heapq.heappush(
                    active_packets, (packet.movetime, insertion_counter, packet)
                )
                insertion_counter += 1
                log_packet_event(
                    current_time,
                    packet.packet_id,
                    "MOVE_EXECUTE",
                    f"To={chosen_hop}, FinishAt={packet.movetime:.4f}",
                )

            elif edge and not edge.is_connected:
                log_packet_event(
                    current_time,
                    packet.packet_id,
                    "LINK_BROKEN",
                    f"Link to {chosen_hop} is broken.",
                )
                if num_viable == 1:
                    packet.movetime = edge.next_event_time
                    heapq.heappush(
                        active_packets, (packet.movetime, insertion_counter, packet)
                    )
                    insertion_counter += 1
                    log_packet_event(
                        current_time,
                        packet.packet_id,
                        "WAIT_DECISION",
                        f"SINGLE_VIABLE. Forced wait until {packet.movetime:.4f}",
                    )
                else:
                    # ステップ1: オブジェクトの種類を安全にチェック
                    if isinstance(packet.wait_max_duration_func, partial):
                        # もし partial でラップされていたら、.func で中の関数を取り出す
                        original_func = packet.wait_max_duration_func.func
                    else:
                        # そうでなければ (no_wait, infinite_wait の場合)、
                        # それ自体が素の関数なので、そのまま使う
                        original_func = packet.wait_max_duration_func

                    # ステップ2: 取り出した元の関数の名前で判定
                    is_ratio_strategy = (
                        original_func.__name__ == "ratio_based_wait_strategy"
                    )

                    # ステップ3: 判定結果に基づいて、正しい引数で呼び出す
                    if is_ratio_strategy:
                        # ratio戦略には ideal_graph を渡す
                        max_w = packet.wait_max_duration_func(
                            packet, ideal_graph, link_tpl
                        )
                    else:
                        # それ以外の戦略には通常の graph を渡す
                        max_w = packet.wait_max_duration_func(packet, graph, link_tpl)

                    if max_w > 0:
                        wait_deadline = current_time + max_w
                        if edge.next_event_time <= wait_deadline:
                            packet.movetime = edge.next_event_time
                            heapq.heappush(
                                active_packets,
                                (packet.movetime, insertion_counter, packet),
                            )
                            insertion_counter += 1
                            log_packet_event(
                                current_time,
                                packet.packet_id,
                                "WAIT_DECISION",
                                f"FINITE_WAIT. Wait until {packet.movetime:.4f}",
                            )
                        else:
                            packet.brokenlink.append(link_tpl)
                            packet.failed_link_attempts += 1
                            packet.movetime = current_time + 1e-5
                            heapq.heappush(
                                active_packets,
                                (packet.movetime, insertion_counter, packet),
                            )
                            insertion_counter += 1
                            log_packet_event(
                                current_time,
                                packet.packet_id,
                                "WAIT_DECISION",
                                "FINITE_WAIT_TIMEOUT. Give up link.",
                            )
                    else:
                        packet.brokenlink.append(link_tpl)
                        packet.failed_link_attempts += 1
                        packet.movetime = current_time + 1e-5
                        heapq.heappush(
                            active_packets, (packet.movetime, insertion_counter, packet)
                        )
                        insertion_counter += 1
                        log_packet_event(
                            current_time,
                            packet.packet_id,
                            "WAIT_DECISION",
                            "NO_WAIT. Give up link.",
                        )

            elif edge and current_time < edge.is_transmitting_until:
                packet.movetime = edge.is_transmitting_until
                heapq.heappush(
                    active_packets, (packet.movetime, insertion_counter, packet)
                )
                insertion_counter += 1
                log_packet_event(
                    current_time,
                    packet.packet_id,
                    "LINK_BUSY",
                    f"Wait until {packet.movetime:.4f}",
                )

            else:
                packet.brokenlink.append(link_tpl)
                packet.failed_link_attempts += 1
                packet.movetime = current_time + 1e-5
                heapq.heappush(
                    active_packets, (packet.movetime, insertion_counter, packet)
                )
                insertion_counter += 1
                log_packet_event(
                    current_time,
                    packet.packet_id,
                    "LINK_ERROR",
                    "Edge object not found.",
                )

    for p in all_packets:
        if not p.is_arrived:
            p.is_arrived, p.end_time = True, current_time

    if DEBUG_LOG_FILE_HANDLE:
        DEBUG_LOG_FILE_HANDLE.close()
        DEBUG_LOG_FILE_HANDLE, DEBUG_LOG_WRITER, TARGET_DEBUG_PACKET_IDS = (
            None,
            None,
            [],
        )

    return all_packets, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Network Simulation (Fast Routing + Numba)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sim_group = parser.add_argument_group("Simulation Parameters")
    strategy_group = parser.add_argument_group("Strategy Parameters")
    file_group = parser.add_argument_group("File and Directory Paths")
    debug_group = parser.add_argument_group("Execution and Debugging")
    strategy_choices = [
        "fixed_wait_duration_strategy",
        "no_wait_strategy",
        "infinite_wait_strategy",
        "dynamic_wait_strategy_with_faillink",
        "dynamic_wait_strategy_with_node_count",
        "ratio_based_wait_strategy",
    ]
    strategy_group.add_argument(
        "--strategy",
        type=str,
        default="ratio_based_wait_strategy",
        choices=strategy_choices,
        help="Waiting strategy function.",
    )
    sim_group.add_argument(
        "--packets", type=int, default=1000, help="Number of packets."
    )
    sim_group.add_argument(
        "--lambda_rate",
        type=float,
        default=0.5,
        help="Packet generation rate (lambda).",
    )
    sim_group.add_argument(
        "--max_sim_time",
        type=float,
        default=float("inf"),
        help="Maximum simulation time.",
    )
    sim_group.add_argument(
        "--connect_rate", type=float, default=0.4, help="Link connect rate."
    )
    sim_group.add_argument(
        "--disconnect_rate", type=float, default=0.2, help="Link disconnect rate."
    )
    sim_group.add_argument(
        "--link_seed", type=int, default=12345, help="RNG seed for link state."
    )
    sim_group.add_argument(
        "--packet_seed", type=int, default=42, help="RNG seed for packet generation."
    )
    strategy_group.add_argument(
        "--base_wait_time",
        type=float,
        default=2.0,
        help="Base wait time for applicable strategies.",
    )
    strategy_group.add_argument(
        "--dynamic_factor",
        type=float,
        default=0.5,
        help="Factor for dynamic strategies.",
    )
    strategy_group.add_argument(
        "--ratio_factor",
        type=float,
        default=1.0,
        help="Factor for ratio_based strategy.",
    )
    file_group.add_argument(
        "--node_file", type=str, default="node.csv", help="Node CSV file."
    )
    file_group.add_argument(
        "--edge_file", type=str, default="edge.csv", help="Edge CSV file."
    )
    file_group.add_argument(
        "--output_base_dir",
        type=str,
        default="simulation_results",
        help="Base directory for outputs.",
    )
    file_group.add_argument(
        "--summary_filename",
        type=str,
        default=None,
        help="Optional specific filename for the summary CSV. Overrides default naming.",
    )
    debug_group.add_argument(
        "--run_id_suffix",
        type=str,
        default="",
        help="Optional suffix for run identifier.",
    )
    debug_group.add_argument(
        "--src_node", type=int, default=0, help="Fixed source node ID."
    )
    debug_group.add_argument(
        "--dst_node", type=int, default=None, help="Fixed destination node ID."
    )
    debug_group.add_argument(
        "--debug_packet_ids",
        type=str,
        default="",
        help="Comma-separated packet IDs for detailed CSV logging.",
    )
    debug_group.add_argument(
        "--debug_log_filename",
        type=str,
        default="packet_trace_log.csv",
        help="Filename for packet trace log.",
    )

    args = parser.parse_args()

    # RNGの初期化
    LINK_STATE_RNG = random.Random(args.link_seed)
    PACKET_GEN_RNG = np.random.default_rng(args.packet_seed)

    strategy_functions_map = {
        "fixed_wait_duration_strategy": fixed_wait_duration_strategy,
        "no_wait_strategy": no_wait_strategy,
        "infinite_wait_strategy": infinite_wait_strategy,
        "dynamic_wait_strategy_with_faillink": dynamic_wait_strategy_with_faillink,
        "dynamic_wait_strategy_with_node_count": dynamic_wait_strategy_with_node_count,
        "ratio_based_wait_strategy": ratio_based_wait_strategy,
    }
    selected_wait_strategy_func = strategy_functions_map[args.strategy]

    ideal_graph_obj = nx.Graph()
    with open(args.node_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            ideal_graph_obj.add_node(int(row["node_id"]), label=row["label"])
    with open(args.edge_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            ideal_graph_obj.add_edge(
                int(row["source"]), int(row["target"]), weight=float(row["weight"])
            )

    print(f"--- Starting Network Simulation (Numba Final Version) ---")

    start_exec_time = time.time()

    final_packets, output_dir = simulate_network(
        node_file=args.node_file,
        edge_file=args.edge_file,
        lambd=args.lambda_rate,
        packet_size=args.packets,
        wait_strategy_func=selected_wait_strategy_func,
        ideal_graph=ideal_graph_obj,
        connect_rate=args.connect_rate,
        disconnect_rate=args.disconnect_rate,
        base_wait_time=args.base_wait_time,
        dynamic_factor=args.dynamic_factor,
        ratio_factor=args.ratio_factor,
        src_node=args.src_node,
        dst_node=args.dst_node,
        max_simulation_time=args.max_sim_time,
        output_base_dir=args.output_base_dir,
        debug_packet_ids_str=args.debug_packet_ids,
        debug_log_filename=args.debug_log_filename,
        run_identifier=args.run_id_suffix,
    )

    end_exec_time = time.time()

    # Determine final summary path
    if args.summary_filename:
        # When called from a sweep script, output_dir is the final directory
        summary_path = os.path.join(output_dir, args.summary_filename)
    else:
        # Fallback for direct execution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_filename = f"summary_direct_run_{timestamp}{args.run_id_suffix}.csv"
        summary_path = os.path.join(output_dir, fallback_filename)

    write_final_packet_summary(final_packets, summary_path)

    print(f"\nFinal packet summary has been written to {summary_path}")

    if final_packets:
        arrived_count = sum(1 for p in final_packets if p.present_node == p.dst)
        num_total = len(final_packets)
        arrival_rate = (arrived_count / num_total * 100) if num_total > 0 else 0
        print(f"Arrival rate: {arrival_rate:.2f}% ({arrived_count}/{num_total})")

        successful_packets = [p for p in final_packets if p.present_node == p.dst]
        if successful_packets:
            avg_delay = sum(
                p.end_time - p.start_time for p in successful_packets
            ) / len(successful_packets)
            avg_hops = sum(p.movecount for p in successful_packets) / len(
                successful_packets
            )
            avg_failed_attempts_success = sum(
                p.failed_link_attempts for p in successful_packets
            ) / len(successful_packets)
            print(f"For packets that ARRIVED SUCCESSFULLY:")
            print(f"  Average delay: {avg_delay:.4f}")
            print(f"  Average hops: {avg_hops:.2f}")
            print(f"  Average failed link attempts: {avg_failed_attempts_success:.2f}")

    print("-" * 50)
    print(">>> NUMBA FINAL VERSION FINISHED <<<")
    print(f"Total execution time: {end_exec_time - start_exec_time:.4f} seconds")
    print("-" * 50)
