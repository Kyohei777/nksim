# nwsim_numba_final_v9_final.py
# (データ指向設計 + 最初のプログラムのロジックを完全に再現した最終安定版)

import networkx as nx
import csv
import numpy as np
import heapq
from datetime import datetime
import random
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
STRATEGY_ID_MAP = {}

# --- データ指向設計のためのインデックス定数 ---
P_STATE_STATUS = 0
P_STATE_SRC = 1
P_STATE_DST = 2
P_STATE_PRESENT_NODE = 3
P_STATE_START_TIME = 5
P_STATE_END_TIME = 6
P_STATE_MOVE_COUNT = 8
P_STATE_FAILED_ATTEMPTS = 9
P_STATE_PATH_LEN = 10
P_STATE_BROKEN_LEN = 11
STATUS_IN_FLIGHT = 0
STATUS_ARRIVED = 1
STATUS_DROPPED_TTL = 2
STATUS_DROPPED_BUFFER = 3
STATUS_DROPPED_NO_ROUTE = 4
STATUS_IN_TRANSIT = 5
E_STATE_IS_CONNECTED = 0
E_STATE_NEXT_EVENT_TIME = 1
E_STATE_TRANSMITTING_UNTIL = 2
E_STATE_WEIGHT = 3
E_STATE_CONNECT_RATE = 4
E_STATE_DISCONNECT_RATE = 5
STRATEGY_ID_FIXED = 0
STRATEGY_ID_NOWAIT = 1
STRATEGY_ID_DYNAMIC_FAIL = 2
STRATEGY_ID_INFINITE = 3
STRATEGY_ID_DYNAMIC_NODE = 4
STRATEGY_ID_RATIO = 5

MAX_PATH_LEN = 64
MAX_BROKEN_LINKS = 32

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ Numba化された高速経路探索エンジン ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


@numba.njit
def _bfs_has_path_numba(
    start_node, end_node, adj_nodes, paththrough_nodes, broken_links
):
    num_nodes = len(adj_nodes)
    if start_node >= num_nodes or end_node >= num_nodes:
        return False

    q = List()
    q.append(start_node)

    visited = np.zeros(num_nodes, dtype=np.bool_)
    visited[start_node] = True

    paththrough_set = set(paththrough_nodes)
    broken_set = set()
    for i in range(len(broken_links)):
        u, v = broken_links[i, 0], broken_links[i, 1]
        if u > v:
            u, v = v, u
        broken_set.add((u, v))

    while len(q) > 0:
        current_node = q.pop(0)
        if current_node == end_node:
            return True

        neighbors = adj_nodes[current_node]
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            if not visited[neighbor]:
                if neighbor in paththrough_set and neighbor != end_node:
                    continue
                u, v = current_node, neighbor
                if u > v:
                    u, v = v, u
                if (u, v) in broken_set:
                    continue

                visited[neighbor] = True
                q.append(neighbor)
    return False


@numba.njit
def _dijkstra_get_next_hop(
    start_node, end_node, adj_nodes, adj_costs, paththrough_nodes, broken_links
):
    num_nodes = len(adj_nodes)
    if start_node >= num_nodes:
        return -1
    dist = np.full(num_nodes, np.inf, dtype=np.float64)
    dist[start_node] = 0
    prev = np.full(num_nodes, -1, dtype=np.int64)

    pq = [(0.0, start_node)]

    paththrough_set = set(paththrough_nodes)
    broken_set = set()
    for i in range(len(broken_links)):
        u, v = broken_links[i, 0], broken_links[i, 1]
        if u > v:
            u, v = v, u
        broken_set.add((u, v))

    while len(pq) > 0:
        d, current_node = heapq.heappop(pq)
        if d > dist[current_node]:
            continue
        if current_node == end_node:
            break

        neighbors = adj_nodes[current_node]
        costs = adj_costs[current_node]
        for i in range(len(neighbors)):
            neighbor, cost = neighbors[i], costs[i]

            if neighbor in paththrough_set and neighbor != end_node:
                continue
            u, v = current_node, neighbor
            if u > v:
                u, v = v, u
            if (u, v) in broken_set:
                continue

            new_dist = dist[current_node] + cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    if end_node >= len(prev) or prev[end_node] == -1:
        return -1

    path_node = end_node
    for _ in range(num_nodes):
        prev_node = prev[path_node]
        if prev_node == -1:
            return -1
        if prev_node == start_node:
            return path_node
        path_node = prev_node
    return -1


@numba.njit
def _dijkstra_get_path_length(
    start_node, end_node, adj_nodes, adj_costs, paththrough_nodes, broken_links
):
    num_nodes = len(adj_nodes)
    if start_node >= num_nodes:
        return np.inf
    dist = np.full(num_nodes, np.inf, dtype=np.float64)
    dist[start_node] = 0

    pq = [(0.0, start_node)]

    paththrough_set = set(paththrough_nodes)
    broken_set = set()
    for i in range(len(broken_links)):
        u, v = broken_links[i, 0], broken_links[i, 1]
        if u > v:
            u, v = v, u
        broken_set.add((u, v))

    while len(pq) > 0:
        d, current_node = heapq.heappop(pq)
        if d > dist[current_node]:
            continue
        if current_node == end_node:
            return d

        neighbors = adj_nodes[current_node]
        costs = adj_costs[current_node]
        for i in range(len(neighbors)):
            neighbor, cost = neighbors[i], costs[i]

            if neighbor in paththrough_set and neighbor != end_node:
                continue
            u, v = current_node, neighbor
            if u > v:
                u, v = v, u
            if (u, v) in broken_set:
                continue

            new_dist = dist[current_node] + cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    if end_node >= len(dist):
        return np.inf
    return dist[end_node]


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ 新しい高速シミュレーションコア (最終安定版) ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


@numba.njit
def _simulation_core_numba(
    packets_state,
    packets_path_hist,
    packets_broken_links,
    edges_state,
    node_buffers,
    node_buffer_counts,
    adj_nodes,
    adj_costs_time,
    adj_costs_reliable,
    edge_to_idx_map,
    max_sim_time,
    ttl,
    strategy_id,
    base_wait_time,
    dynamic_factor,
    ratio_factor,
    link_seed,
    ideal_next_hop_table,
    detour_penalty,
    detour_penalty_randomization,
    routing_strategy_id,
):
    num_packets = len(packets_state)
    num_edges = len(edges_state)
    processed_packets = 0
    current_time = 0.0

    event_queue = [(0.0, 0, -1, 0.0)]
    heapq.heappop(event_queue)

    for pid in range(num_packets):
        start_time = packets_state[pid, P_STATE_START_TIME]
        src_node = int(packets_state[pid, P_STATE_SRC])
        node_buffer_counts[src_node] += 1
        heapq.heappush(event_queue, (start_time, 0, pid, 0.0))

    rng_state = np.uint64(link_seed)

    def random_float():
        nonlocal rng_state
        rng_state = np.uint64(1103515245) * rng_state + np.uint64(12345)
        return float(rng_state & np.uint64(0x7FFFFFFF)) / float(0x7FFFFFFF)

    LOG_1_minus_random = lambda: -np.log(1.0 - random_float())

    while (
        len(event_queue) > 0
        and processed_packets < num_packets
        and current_time < max_sim_time
    ):
        event_time, event_type, pid, value = heapq.heappop(event_queue)

        if packets_state[pid, P_STATE_STATUS] != STATUS_IN_FLIGHT:
            continue

        current_time = event_time

        if current_time - packets_state[pid, P_STATE_START_TIME] > ttl:
            packets_state[pid, P_STATE_STATUS] = STATUS_DROPPED_TTL
            packets_state[pid, P_STATE_END_TIME] = current_time
            node_buffer_counts[int(packets_state[pid, P_STATE_PRESENT_NODE])] -= 1
            processed_packets += 1
            continue

        for eid in range(num_edges):
            if current_time >= edges_state[eid, E_STATE_NEXT_EVENT_TIME]:
                is_connected_after = not (edges_state[eid, E_STATE_IS_CONNECTED] == 1.0)
                edges_state[eid, E_STATE_IS_CONNECTED] = (
                    1.0 if is_connected_after else 0.0
                )
                rate = edges_state[eid, E_STATE_DISCONNECT_RATE] if is_connected_after else edges_state[eid, E_STATE_CONNECT_RATE]
                interval = LOG_1_minus_random() / rate
                edges_state[eid, E_STATE_NEXT_EVENT_TIME] = current_time + interval

        if event_type == 1:  # ARRIVE_AT_NODE
            arrival_node = int(value)

            if node_buffer_counts[arrival_node] < node_buffers[arrival_node]:
                node_buffer_counts[arrival_node] += 1
                path_len = int(packets_state[pid, P_STATE_PATH_LEN])
                if path_len < MAX_PATH_LEN:
                    packets_path_hist[pid, path_len] = int(
                        packets_state[pid, P_STATE_PRESENT_NODE]
                    )
                    packets_state[pid, P_STATE_PATH_LEN] += 1

                packets_state[pid, P_STATE_PRESENT_NODE] = arrival_node
                packets_state[pid, P_STATE_MOVE_COUNT] += 1
                packets_state[pid, P_STATE_BROKEN_LEN] = 0
                heapq.heappush(event_queue, (current_time, 0, pid, 0.0))
            else:
                packets_state[pid, P_STATE_STATUS] = STATUS_DROPPED_BUFFER
                packets_state[pid, P_STATE_END_TIME] = current_time
                processed_packets += 1

        elif event_type == 0:  # PROCESS_AT_NODE
            present_node = int(packets_state[pid, P_STATE_PRESENT_NODE])
            dst_node = int(packets_state[pid, P_STATE_DST])

            if present_node == dst_node:
                packets_state[pid, P_STATE_STATUS] = STATUS_ARRIVED
                packets_state[pid, P_STATE_END_TIME] = current_time
                node_buffer_counts[present_node] -= 1
                processed_packets += 1
                continue

            path_len = int(packets_state[pid, P_STATE_PATH_LEN])
            pt_nodes_arr = packets_path_hist[pid, :path_len]

            broken_len = int(packets_state[pid, P_STATE_BROKEN_LEN])
            bl_links_arr = packets_broken_links[pid, :broken_len]

            viable_hops = List.empty_list(types.int64)
            neighbors = adj_nodes[present_node]
            for i in range(len(neighbors)):
                neighbor = neighbors[i]
                is_paththrough = False
                for j in range(path_len):
                    if pt_nodes_arr[j] == neighbor:
                        is_paththrough = True
                        break
                if is_paththrough:
                    continue
                is_broken = False
                u_b, v_b = present_node, neighbor
                if u_b > v_b:
                    u_b, v_b = v_b, u_b
                for j in range(broken_len):
                    if bl_links_arr[j, 0] == u_b and bl_links_arr[j, 1] == v_b:
                        is_broken = True
                        break
                if is_broken:
                    continue

                # BFSの引数を修正
                bfs_pt_nodes = np.append(
                    pt_nodes_arr, np.array([present_node], dtype=np.int64)
                )
                if _bfs_has_path_numba(
                    neighbor, dst_node, adj_nodes, bfs_pt_nodes, bl_links_arr
                ):
                    viable_hops.append(neighbor)

            num_viable = len(viable_hops)

            if num_viable == 0:
                earliest_next_event_time = np.inf
                for eid in range(num_edges):
                    if (
                        edges_state[eid, E_STATE_NEXT_EVENT_TIME]
                        < earliest_next_event_time
                    ):
                        earliest_next_event_time = edges_state[
                            eid, E_STATE_NEXT_EVENT_TIME
                        ]
                if (
                    earliest_next_event_time == np.inf
                    or earliest_next_event_time <= current_time
                ):
                    earliest_next_event_time = current_time + 1e-4
                heapq.heappush(event_queue, (earliest_next_event_time, 0, pid, 0.0))
                continue

            adj_costs = (
                adj_costs_reliable if routing_strategy_id == 1 else adj_costs_time
            )
            next_hop = _dijkstra_get_next_hop(
                present_node, dst_node, adj_nodes, adj_costs, pt_nodes_arr, bl_links_arr
            )

            if next_hop == -1:
                heapq.heappush(event_queue, (current_time + 1e-5, 0, pid, 0.0))
                continue

            penalty = 1.0
            if detour_penalty > 1.0:
                ideal_next_hop = ideal_next_hop_table[present_node, dst_node]
                if ideal_next_hop != -1 and next_hop != ideal_next_hop:
                    if detour_penalty_randomization > 0.0:
                        random_shift = (
                            random_float() - 0.5
                        ) * 2.0 * detour_penalty_randomization
                        penalty = detour_penalty + random_shift
                        if penalty < 0.0:
                            penalty = 0.0
                    else:
                        penalty = detour_penalty

            u, v = present_node, next_hop
            if u > v:
                u, v = v, u
            link_tuple = (u, v)

            edge_idx_opt = edge_to_idx_map.get(link_tuple)

            if edge_idx_opt is not None:
                edge_idx = int(edge_idx_opt)
                edge = edges_state[edge_idx]
                if (
                    edge[E_STATE_IS_CONNECTED] == 1.0
                    and current_time >= edge[E_STATE_TRANSMITTING_UNTIL]
                ):
                    node_buffer_counts[present_node] -= 1
                    transmission_time = edge[E_STATE_WEIGHT] * penalty
                    arrival_time = current_time + transmission_time
                    edges_state[edge_idx, E_STATE_TRANSMITTING_UNTIL] = arrival_time
                    heapq.heappush(event_queue, (arrival_time, 1, pid, float(next_hop)))

                elif edge[E_STATE_IS_CONNECTED] == 0.0:
                    if num_viable == 1:
                        heapq.heappush(
                            event_queue, (edge[E_STATE_NEXT_EVENT_TIME], 0, pid, 0.0)
                        )
                    else:
                        max_w = 0.0
                        if strategy_id == STRATEGY_ID_FIXED:
                            max_w = base_wait_time
                        elif strategy_id == STRATEGY_ID_NOWAIT:
                            max_w = 0.0
                        elif strategy_id == STRATEGY_ID_INFINITE:
                            max_w = 999999.0
                        elif strategy_id == STRATEGY_ID_DYNAMIC_FAIL:
                            max_w = (
                                dynamic_factor
                                * packets_state[pid, P_STATE_FAILED_ATTEMPTS]
                            )
                        elif strategy_id == STRATEGY_ID_DYNAMIC_NODE:
                            max_w = (
                                dynamic_factor * packets_state[pid, P_STATE_MOVE_COUNT]
                            )
                        elif strategy_id == STRATEGY_ID_RATIO:
                            original_length = _dijkstra_get_path_length(
                                present_node,
                                dst_node,
                                adj_nodes,
                                adj_costs,
                                pt_nodes_arr,
                                bl_links_arr,
                            )

                            temp_bl_len = broken_len + 1
                            temp_bl_links_arr = np.empty(
                                (temp_bl_len, 2), dtype=np.int64
                            )
                            if broken_len > 0:
                                temp_bl_links_arr[:broken_len, :] = bl_links_arr
                            u_b, v_b = present_node, next_hop
                            if u_b > v_b:
                                u_b, v_b = v_b, u_b
                            temp_bl_links_arr[broken_len, 0] = u_b
                            temp_bl_links_arr[broken_len, 1] = v_b

                            detour_length = _dijkstra_get_path_length(
                                present_node,
                                dst_node,
                                adj_nodes,
                                adj_costs,
                                pt_nodes_arr,
                                temp_bl_links_arr,
                            )

                            if detour_length == np.inf:
                                max_w = 999999.0
                            elif original_length > 0:
                                max_w = (detour_length / original_length) * ratio_factor
                            else:
                                max_w = 0.0

                        if (
                            max_w > 0
                            and edge[E_STATE_NEXT_EVENT_TIME] <= current_time + max_w
                        ):
                            heapq.heappush(
                                event_queue,
                                (edge[E_STATE_NEXT_EVENT_TIME], 0, pid, 0.0),
                            )
                        else:
                            packets_state[pid, P_STATE_FAILED_ATTEMPTS] += 1
                            if broken_len < MAX_BROKEN_LINKS:
                                packets_broken_links[pid, broken_len, 0] = u
                                packets_broken_links[pid, broken_len, 1] = v
                                packets_state[pid, P_STATE_BROKEN_LEN] += 1
                            heapq.heappush(
                                event_queue, (current_time + 1e-5, 0, pid, 0.0)
                            )

                elif current_time < edge[E_STATE_TRANSMITTING_UNTIL]:
                    heapq.heappush(
                        event_queue, (edge[E_STATE_TRANSMITTING_UNTIL], 0, pid, 0.0)
                    )

    for pid in range(num_packets):
        if packets_state[pid, P_STATE_STATUS] == STATUS_IN_FLIGHT:
            packets_state[pid, P_STATE_STATUS] = STATUS_IN_TRANSIT
            packets_state[pid, P_STATE_END_TIME] = current_time

    return packets_state, packets_path_hist, packets_broken_links


# --- クラス定義 (データ準備用) ---
class Packet:
    def __init__(self, src, dst, time):
        self.src, self.dst, self.start_time = src, dst, time
        self.packet_id = -1


class Edge:
    def __init__(self, s, t, w, n, it, cr, dr):
        self.source, self.target, self.weight, self.name = s, t, w, n
        self.connect_rate, self.disconnect_rate = cr, dr
        self.is_connected, self.next_event_time = True, 0
        self.initialize(it, self.connect_rate, self.disconnect_rate)

    def initialize(self, ct, cr, dr):
        global LINK_STATE_RNG_LEGACY
        if LINK_STATE_RNG_LEGACY is None:
            LINK_STATE_RNG_LEGACY = random.Random()
        self.is_connected = LINK_STATE_RNG_LEGACY.choice([True, False])
        rate = dr if self.is_connected else cr
        self.next_event_time = ct + LINK_STATE_RNG_LEGACY.expovariate(rate)


class Graph:
    def __init__(self, nf, ef, cr, dr, bs, sid):
        self.G = nx.Graph()
        self.node_file_name, self.edge_file_name = nf, ef
        self.connect_rate, self.disconnect_rate = cr, dr
        self.unique_edges_list = []
        self.edge_to_idx_map_for_numba = Dict.empty(
            key_type=types.UniTuple(types.int64, 2), value_type=types.int64
        )
        self.node_buffers = np.array([0], dtype=np.int64)
        self.global_buffer_size = bs if bs != -1 else np.iinfo(np.int64).max
        self.src_node_id = sid
        self.adj_nodes = List.empty_list(types.int64[:])
        self.adj_costs_time = List.empty_list(types.float64[:])
        self.adj_costs_reliable = List.empty_list(types.float64[:])
        if os.path.exists(self.node_file_name) and os.path.exists(self.edge_file_name):
            self.make_graph()

    def make_graph(self):
        nodes = []
        with open(self.node_file_name, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["node_id"])
                self.G.add_node(node_id, label=row["label"])
                nodes.append(node_id)

        max_node_id = max(nodes) if nodes else -1
        self.node_buffers = np.full(
            max_node_id + 1, self.global_buffer_size, dtype=np.int64
        )
        if self.src_node_id <= max_node_id:
            self.node_buffers[self.src_node_id] = np.iinfo(np.int64).max

        with open(self.edge_file_name, "r", newline="") as f:
            reader = csv.DictReader(f)
            # Check for individual rate columns
            has_individual_rates = ('connect_rate' in reader.fieldnames and 
                                    'disconnect_rate' in reader.fieldnames)
            
            processed_pairs = set()
            edge_idx_counter = 0
            for row in reader:
                s, t = int(row["source"]), int(row["target"])
                pair = tuple(sorted((s, t)))
                if pair not in processed_pairs:
                    
                    cr = float(row["connect_rate"]) if has_individual_rates else self.connect_rate
                    dr = float(row["disconnect_rate"]) if has_individual_rates else self.disconnect_rate

                    edge = Edge(
                        s,
                        t,
                        float(row["weight"]),
                        row["name"],
                        0,
                        cr,
                        dr,
                    )
                    self.unique_edges_list.append(edge)
                    self.G.add_edge(s, t, weight=float(row["weight"]), name=row["name"])
                    self.edge_to_idx_map_for_numba[pair] = edge_idx_counter
                    edge_idx_counter += 1
                    processed_pairs.add(pair)

    def build_adjacency_list_for_numba(self, link_evaluation_data=None):
        max_node_id = max(self.G.nodes()) if self.G.nodes() else -1

        adj_nodes_list = [[] for _ in range(max_node_id + 1)]
        adj_costs_time_list = [[] for _ in range(max_node_id + 1)]
        adj_costs_reliable_list = [[] for _ in range(max_node_id + 1)]

        for u, v, data in self.G.edges(data=True):
            tc = data["weight"]
            cc = tc
            if link_evaluation_data:
                lt = tuple(sorted((u, v)))
                cc = link_evaluation_data.get(lt, {}).get("composite_cost", tc)

            # ホップ数ベースの経路選択のため、dijkstra用のコストを常に1.0に設定
            adj_nodes_list[u].append(v)
            adj_costs_time_list[u].append(1.0)
            adj_costs_reliable_list[u].append(cc)

            adj_nodes_list[v].append(u)
            adj_costs_time_list[v].append(1.0)
            adj_costs_reliable_list[v].append(cc)

        for i in range(max_node_id + 1):
            self.adj_nodes.append(np.array(adj_nodes_list[i], dtype=np.int64))
            self.adj_costs_time.append(
                np.array(adj_costs_time_list[i], dtype=np.float64)
            )
            self.adj_costs_reliable.append(
                np.array(adj_costs_reliable_list[i], dtype=np.float64)
            )


# --- パケット生成とサマリー書き出し ---
def generate_packets_for_state_array(
    lambd, size, number_of_nodes, src_node=0, dst_node=None
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
        packet = Packet(src_node, actual_dst, arrival_times[i])
        packet.packet_id = i
        packet_list.append(packet)
    return packet_list


def write_final_packet_summary_from_state(
    packets_state, path_hist, broken_links, filename
):
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
                "Status",
                "Broken Links (Final)",
            ]
            writer.writerow(header)
            status_map = {
                STATUS_ARRIVED: "ARRIVED",
                STATUS_DROPPED_TTL: "DROPPED_TTL",
                STATUS_DROPPED_BUFFER: "DROPPED_BUFFER",
                STATUS_DROPPED_NO_ROUTE: "DROPPED_NO_ROUTE",
                STATUS_IN_TRANSIT: "IN_TRANSIT",
                STATUS_IN_FLIGHT: "IN_FLIGHT",
            }
            for pid in range(len(packets_state)):
                p_data = packets_state[pid]
                status_str = status_map.get(int(p_data[P_STATE_STATUS]), "UNKNOWN")
                src = int(p_data[P_STATE_SRC])
                dst = int(p_data[P_STATE_DST])
                path_len = int(p_data[P_STATE_PATH_LEN])
                path_nodes_list = [str(src)] + [
                    str(node) for node in path_hist[pid, :path_len]
                ]
                if status_str == "ARRIVED":
                    if not path_nodes_list or path_nodes_list[-1] != str(dst):
                        path_nodes_list.append(str(dst))
                path_str = ",".join(path_nodes_list)
                broken_len = int(p_data[P_STATE_BROKEN_LEN])
                broken_links_list = broken_links[pid, :broken_len]
                broken_links_str = ";".join(
                    [f"({u},{v})" for u, v in broken_links_list]
                )
                writer.writerow(
                    [
                        pid,
                        src,
                        dst,
                        f"{p_data[P_STATE_START_TIME]:.4f}",
                        f"{p_data[P_STATE_END_TIME]:.4f}",
                        int(p_data[P_STATE_MOVE_COUNT]),
                        int(p_data[P_STATE_FAILED_ATTEMPTS]),
                        path_str,
                        status_str,
                        broken_links_str,
                    ]
                )
    except Exception as e:
        print(f"ERROR writing summary file {filename}: {e}")


# --- 学習フェーズ (変更なし) ---
def run_link_evaluation_phase(args):
    global LINK_STATE_RNG_LEGACY
    if LINK_STATE_RNG_LEGACY is None:
        LINK_STATE_RNG_LEGACY = random.Random(args.link_seed)

    print(
        f"--- Starting Link Evaluation Phase (Duration: {args.monitoring_duration}s) ---"
    )

    eval_graph_obj = nx.Graph()
    edge_list_for_eval = []
    with open(args.node_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            eval_graph_obj.add_node(int(row["node_id"]), label=row["label"])
            
    with open(args.edge_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        has_individual_rates = ('connect_rate' in reader.fieldnames and 
                                'disconnect_rate' in reader.fieldnames)
        processed_pairs = set()
        for row in reader:
            s, t = int(row["source"]), int(row["target"])
            pair = tuple(sorted((s, t)))
            if pair not in processed_pairs:
                cr = float(row["connect_rate"]) if has_individual_rates else args.connect_rate
                dr = float(row["disconnect_rate"]) if has_individual_rates else args.disconnect_rate
                
                eval_graph_obj.add_edge(s, t, weight=float(row["weight"]))
                edge_list_for_eval.append(
                    {"pair": pair, "weight": float(row["weight"]), "connect_rate": cr, "disconnect_rate": dr}
                )
                processed_pairs.add(pair)

    num_edges = len(edge_list_for_eval)

    link_states = np.zeros((num_edges, 2), dtype=np.float64)
    link_data = {
        edge["pair"]: {"connected_time": 0, "disconnected_time": 0}
        for edge in edge_list_for_eval
    }

    for i in range(num_edges):
        is_connected = LINK_STATE_RNG_LEGACY.choice([True, False])
        link_states[i, 0] = 1.0 if is_connected else 0.0
        edge_info = edge_list_for_eval[i]
        rate = edge_info["disconnect_rate"] if is_connected else edge_info["connect_rate"]
        link_states[i, 1] = LINK_STATE_RNG_LEGACY.expovariate(rate)

    time_step = 0.1
    for t in np.arange(0, args.monitoring_duration, time_step):
        for i in range(num_edges):
            if t >= link_states[i, 1]:
                is_connected_after = not (link_states[i, 0] == 1.0)
                link_states[i, 0] = 1.0 if is_connected_after else 0.0
                edge_info = edge_list_for_eval[i]
                rate = edge_info["disconnect_rate"] if is_connected_after else edge_info["connect_rate"]
                interval = LINK_STATE_RNG_LEGACY.expovariate(rate)
                link_states[i, 1] = t + interval

            pair = edge_list_for_eval[i]["pair"]
            if link_states[i, 0] == 1.0:
                link_data[pair]["connected_time"] += time_step
            else:
                link_data[pair]["disconnected_time"] += time_step

    for i in range(num_edges):
        pair = edge_list_for_eval[i]["pair"]
        data = link_data[pair]
        total_time = data["connected_time"] + data["disconnected_time"]
        reliability = (data["connected_time"] / total_time) if total_time > 0 else 0.5
        data["reliability_score"] = reliability
        original_weight = edge_list_for_eval[i]["weight"]
        composite_cost = (1 - reliability) # Changed to exclude original_weight
        data["composite_cost"] = composite_cost

    print("--- Link Evaluation Complete ---")
    return link_data


# --- シミュレーションラッパー関数 ---
def simulate_network(args, strategy_id, ideal_graph, link_evaluation_data=None):

    # --- 1. データ構造の準備 ---
    graph = Graph(
        args.node_file,
        args.edge_file,
        args.connect_rate,
        args.disconnect_rate,
        args.buffer_size,
        args.src_node,
    )
    if not graph.G:
        return None, None, None, "."
    graph.build_adjacency_list_for_numba(link_evaluation_data)

    max_node_id = max(graph.G.nodes()) if graph.G.nodes() else -1
    ideal_next_hop_table = np.full(
        (max_node_id + 1, max_node_id + 1), -1, dtype=np.int64
    )
    if args.detour_penalty > 1.0:
        print("Pre-calculating ideal shortest paths for detour detection...")
        all_ideal_paths = dict(nx.all_pairs_dijkstra_path(graph.G, weight="weight"))
        for src_node in range(max_node_id + 1):
            if src_node not in all_ideal_paths:
                continue
            for dst_node in range(max_node_id + 1):
                if dst_node not in all_ideal_paths[src_node]:
                    continue
                path = all_ideal_paths[src_node][dst_node]
                if len(path) > 1:
                    ideal_next_hop_table[src_node, dst_node] = path[1]

    actual_dst_node = (
        args.dst_node
        if args.dst_node is not None
        else (graph.G.number_of_nodes() - 1)
    )

    packets_list = generate_packets_for_state_array(
        args.lambda_rate,
        args.packets,
        graph.G.number_of_nodes(),
        args.src_node,
        actual_dst_node,
    )
    num_packets = len(packets_list)

    packets_state = np.zeros((num_packets, 12), dtype=np.float64)
    for p in packets_list:
        pid = p.packet_id
        packets_state[pid, P_STATE_STATUS] = STATUS_IN_FLIGHT
        packets_state[pid, P_STATE_SRC] = p.src
        packets_state[pid, P_STATE_DST] = p.dst
        packets_state[pid, P_STATE_PRESENT_NODE] = p.src
        packets_state[pid, P_STATE_START_TIME] = p.start_time

    packets_path_hist = np.full((num_packets, MAX_PATH_LEN), -1, dtype=np.int64)
    packets_broken_links = np.full(
        (num_packets, MAX_BROKEN_LINKS, 2), -1, dtype=np.int64
    )

    edges_state = np.zeros((len(graph.unique_edges_list), 6), dtype=np.float64)
    for idx, edge_obj in enumerate(graph.unique_edges_list):
        edges_state[idx, E_STATE_IS_CONNECTED] = 1.0 if edge_obj.is_connected else 0.0
        edges_state[idx, E_STATE_NEXT_EVENT_TIME] = edge_obj.next_event_time
        edges_state[idx, E_STATE_WEIGHT] = edge_obj.weight
        edges_state[idx, E_STATE_CONNECT_RATE] = edge_obj.connect_rate
        edges_state[idx, E_STATE_DISCONNECT_RATE] = edge_obj.disconnect_rate


    node_buffer_counts = np.zeros_like(graph.node_buffers, dtype=np.int64)
    routing_strategy_id = 1 if args.routing_strategy == "reliable" else 0

    # --- 2. Numbaコア関数の実行 ---
    print(f"Starting Numba core simulation... ({args.packets} packets)")

    final_packets_state, final_path_hist, final_broken_links = _simulation_core_numba(
        packets_state,
        packets_path_hist,
        packets_broken_links,
        edges_state,
        graph.node_buffers,
        node_buffer_counts,
        graph.adj_nodes,
        graph.adj_costs_time,
        graph.adj_costs_reliable,
        graph.edge_to_idx_map_for_numba,
        args.max_sim_time,
        args.ttl,
        strategy_id,
        args.base_wait_time,
        args.dynamic_factor,
        args.ratio_factor,
        args.link_seed,
        ideal_next_hop_table,
        args.detour_penalty,
        args.detour_penalty_randomization,
        routing_strategy_id,
    )

    # --- 3. 後処理 ---
    print("Numba core simulation finished.")

    # output_dir_name = f"{args.routing_strategy}_{strategy_name}" # Removed this line
    output_dir = args.output_base_dir # Modified to use output_base_dir directly
    os.makedirs(output_dir, exist_ok=True)

    return final_packets_state, final_path_hist, final_broken_links, output_dir


def main(args):
    global LINK_STATE_RNG_LEGACY, PACKET_GEN_RNG
    LINK_STATE_RNG_LEGACY = random.Random(args.link_seed)
    PACKET_GEN_RNG = np.random.default_rng(args.packet_seed)

    selected_strategy_id = STRATEGY_ID_MAP[args.strategy]

    ideal_graph_obj = None  # Not used in this version
    link_evaluation_results = None
    if args.routing_strategy == "reliable":
        link_evaluation_results = run_link_evaluation_phase(args)

    print(
        f"\n--- Starting Simulation Phase (Routing: {args.routing_strategy}, Wait: {args.strategy}) ---"
    )
    start_exec_time = time.time()

    final_packets_state, final_path_hist, final_broken_links, output_dir = (
        simulate_network(
            args, selected_strategy_id, ideal_graph_obj, link_evaluation_results
        )
    )

    end_exec_time = time.time()

    if final_packets_state is not None:
        if args.summary_filename:
            summary_path = os.path.join(output_dir, args.summary_filename)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(
                output_dir,
                f"summary_{args.routing_strategy}_{args.run_id_suffix}_{timestamp}.csv",
            )

        write_final_packet_summary_from_state(
            final_packets_state, final_path_hist, final_broken_links, summary_path
        )
        print(f"\nFinal packet summary has been written to {summary_path}")

        num_total = len(final_packets_state)
        status_col = final_packets_state[:, P_STATE_STATUS].astype(np.int32)

        status_counts = {
            "ARRIVED": np.sum(status_col == STATUS_ARRIVED),
            "DROPPED_TTL": np.sum(status_col == STATUS_DROPPED_TTL),
            "DROPPED_BUFFER": np.sum(status_col == STATUS_DROPPED_BUFFER),
            "DROPPED_NO_ROUTE": np.sum(status_col == STATUS_DROPPED_NO_ROUTE),
            "IN_TRANSIT": np.sum(status_col == STATUS_IN_TRANSIT),
        }

        print("-" * 50)
        print("SIMULATION RESULTS:")
        for status, count in status_counts.items():
            rate = (count / num_total * 100) if num_total > 0 else 0
            print(f"  {status:<18}: {count:>6} packets ({rate:6.2f}%)")
        print("-" * 50)

        successful_mask = status_col == STATUS_ARRIVED
        if np.any(successful_mask):
            successful_packets = final_packets_state[successful_mask]
            avg_delay = np.mean(
                successful_packets[:, P_STATE_END_TIME]
                - successful_packets[:, P_STATE_START_TIME]
            )
            avg_hops = np.mean(successful_packets[:, P_STATE_MOVE_COUNT])
            avg_failed_attempts_success = np.mean(
                successful_packets[:, P_STATE_FAILED_ATTEMPTS]
            )
            print("For packets that ARRIVED SUCCESSFULLY:")
            print(f"  Average delay: {avg_delay:.4f}")
            print(f"  Average hops: {avg_hops:.2f}")
            print(f"  Average failed link attempts: {avg_failed_attempts_success:.2f}")

    print("-" * 50)
    print(">>> HIGH-PERFORMANCE REFACTORED VERSION (FINAL) FINISHED <<<")
    print(f"Total execution time: {end_exec_time - start_exec_time:.4f} seconds")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Network Simulation (High-Performance Refactored Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sim_group = parser.add_argument_group("Simulation Parameters")
    strategy_group = parser.add_argument_group("Strategy Parameters")
    file_group = parser.add_argument_group("File and Directory Paths")
    debug_group = parser.add_argument_group("Execution and Debugging")

    strategy_group.add_argument(
        "--routing_strategy",
        type=str,
        default="dijkstra",
        choices=["dijkstra", "reliable"],
        help="Routing strategy to use.",
    )
    strategy_group.add_argument(
        "--monitoring_duration",
        type=float,
        default=100.0,
        help="Duration for the link evaluation phase.",
    )
    strategy_group.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha factor for 'reliable' routing."
    )
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
        "--buffer_size",
        type=int,
        default=-1,
        help="Buffer size for each node (source node is always infinite). Use -1 for infinite.",
    )
    sim_group.add_argument(
        "--ttl",
        type=float,
        default=float("inf"),
        help="Time To Live for packets in seconds.",
    )
    sim_group.add_argument(
        "--connect_rate", type=float, default=0.4, help="Default link connect rate (used if not in edge file)."
    )
    sim_group.add_argument(
        "--disconnect_rate", type=float, default=0.2, help="Default link disconnect rate (used if not in edge file)."
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
    strategy_group.add_argument(
        "--detour_penalty",
        type=float,
        default=1.0,
        help="Penalty factor for transmission time on detours from the ideal shortest path. Default 1.0 (no penalty).",
    )
    strategy_group.add_argument(
        "--detour_penalty_randomization",
        type=float,
        default=0.0,
        help="Randomization range for detour penalty. The penalty will be a random value in [detour_penalty +/- this value].",
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
        help="Optional specific filename for the summary CSV.",
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

    if not (os.path.exists(args.node_file) and os.path.exists(args.edge_file)):
        print(f"ERROR: Input files not found: {args.node_file}, {args.edge_file}")
        sys.exit(1)

    STRATEGY_ID_MAP.update(
        {
            "fixed_wait_duration_strategy": STRATEGY_ID_FIXED,
            "no_wait_strategy": STRATEGY_ID_NOWAIT,
            "dynamic_wait_strategy_with_faillink": STRATEGY_ID_DYNAMIC_FAIL,
            "infinite_wait_strategy": STRATEGY_ID_INFINITE,
            "dynamic_wait_strategy_with_node_count": STRATEGY_ID_DYNAMIC_NODE,
            "ratio_based_wait_strategy": STRATEGY_ID_RATIO,
        }
    )

    main(args)
