import csv
import random
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime


# (ensure_min_degree 関数 と generate_sample_input_files 関数は前回の回答のものをそのまま使用)
# (ここでは省略しますが、実際にはこれらの関数の定義がこの上に必要です)
def ensure_min_degree(graph, min_degree, max_retries_per_node=10):
    """
    指定されたグラフの全ノードが最低次数を満たすようにエッジを追加する。
    Args:
        graph (nx.Graph): 対象のNetworkXグラフオブジェクト。
        min_degree (int): 保証したい最低次数。
        max_retries_per_node (int): 1つのノードに対して適切な接続先を見つける最大試行回数。
    """
    if graph.number_of_nodes() == 0 or min_degree <= 0:
        return

    modified = True
    loop_guard = 0
    max_loops = graph.number_of_nodes() * min_degree * 2

    while modified and loop_guard < max_loops:
        modified = False
        loop_guard += 1
        current_nodes = list(graph.nodes())
        random.shuffle(current_nodes)

        for node_u in current_nodes:
            current_degree_u = graph.degree(node_u)
            if current_degree_u < min_degree:
                needed_edges = min_degree - current_degree_u
                possible_neighbors = [
                    n
                    for n in graph.nodes()
                    if n != node_u and not graph.has_edge(node_u, n)
                ]
                random.shuffle(possible_neighbors)

                added_count_for_u = 0
                for node_v in possible_neighbors:
                    if added_count_for_u >= needed_edges:
                        break
                    graph.add_edge(node_u, node_v)
                    print(
                        f"    EnsureMinDegree: Added edge ({node_u}, {node_v}) to meet min_degree for node {node_u}"
                    )
                    added_count_for_u += 1
                    modified = True
                if added_count_for_u < needed_edges:
                    print(
                        f"    Warning: Could not add enough edges for node {node_u} to meet min_degree {min_degree}. (Needed {needed_edges}, Added {added_count_for_u})"
                    )
        if not modified:
            break
    if loop_guard >= max_loops:
        print(
            "    Warning: ensure_min_degree reached max_loops. Minimum degree might not be fully guaranteed."
        )

def ensure_graph_is_connected(graph, min_weight, max_weight):
    """
    グラフが連結であることを保証する。不連結な場合、コンポーネント間を接続する。
    Args:
        graph (nx.Graph): 対象のNetworkXグラフオブジェクト。
        min_weight (float): 追加するエッジの最小重み。
        max_weight (float): 追加するエッジの最大重み。
    Returns:
        nx.Graph: 連結されたグラフ。
    """
    if graph.number_of_nodes() <= 1:
        return graph

    if nx.is_connected(graph):
        print("    Graph is already connected.")
        return graph

    print("    Graph is not connected. Attempting to connect components...")
    components = list(nx.connected_components(graph))
    num_components = len(components)
    added_edges_count = 0

    for i in range(num_components - 1):
        # 異なるコンポーネントからランダムにノードを選択
        node1 = random.choice(list(components[i]))
        node2 = random.choice(list(components[i+1]))

        # 既に辺が存在しないことを確認して追加
        if not graph.has_edge(node1, node2):
            weight = round(random.uniform(min_weight, max_weight), 2)
            graph.add_edge(node1, node2, weight=weight, name=f"L_conn_{node1}-{node2}")
            added_edges_count += 1
            print(f"      Added bridge edge ({node1}, {node2}) with weight {weight}")

    if nx.is_connected(graph):
        print(f"    Graph is now connected after adding {added_edges_count} bridge edges.")
    else:
        print("    Warning: Graph is still not connected after initial bridging. Retrying...")
        # 再度連結性を確認し、必要であればさらに接続を試みる（より堅牢にするため）
        # ただし、無限ループにならないように注意
        max_retries = graph.number_of_nodes() # 適当な上限
        retries = 0
        while not nx.is_connected(graph) and retries < max_retries:
            components = list(nx.connected_components(graph))
            if len(components) <= 1:
                break
            node1 = random.choice(list(components[0]))
            node2 = random.choice(list(components[1])) # 最初の2つのコンポーネントを接続
            if not graph.has_edge(node1, node2):
                weight = round(random.uniform(min_weight, max_weight), 2)
                graph.add_edge(node1, node2, weight=weight, name=f"L_conn_{node1}-{node2}_retry{retries}")
                added_edges_count += 1
                print(f"      Added retry bridge edge ({node1}, {node2}) with weight {weight}")
            retries += 1
        if nx.is_connected(graph):
            print(f"    Graph is now connected after further {retries} retries.")
        else:
            print("    ERROR: Graph could not be connected after multiple attempts.")

    return graph

def generate_sample_input_files(
    num_nodes=100,
    probability_of_edge=0.3,
    min_degree_guarantee=0,
    min_weight=0.5,
    max_weight=0.5,
    node_output_path="node.csv",
    edge_output_path="edge.csv",
    topology_type="barabasi_albert",
    k_neighbors=3,
    connection_radius=0.2,
    num_hubs=3,
    connect_hubs_complete=True,
    barabasi_m=2,
):
    print(f"Generating sample input files for {num_nodes} nodes...")
    print(f"Topology type: {topology_type}")
    if topology_type == "barabasi_albert":
        print(f"Barabasi-Albert with m={barabasi_m}")
    elif topology_type == "k_nearest_neighbor":
        print(f"K-Nearest Neighbors: {k_neighbors}")
    elif topology_type == "rgg":
        print(f"Random Geometric Graph (RGG) with radius: {connection_radius}")
    elif topology_type == "multi_hub_star":
        print(f"Multi-hub Star with {num_hubs} hubs. Hubs connected: {connect_hubs_complete}")
    if min_degree_guarantee > 0:
        print(
            f"Attempting to ensure minimum degree of {min_degree_guarantee} for all nodes."
        )

    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({"node_id": i, "label": f"Node{i}"})

    with open(node_output_path, "w", newline="") as nf:
        fieldnames = ["node_id", "label"]
        writer = csv.DictWriter(nf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodes_data)
    print(f"Generated {node_output_path} with {len(nodes_data)} nodes.")

    G = None
    if topology_type == "random":
        G = nx.erdos_renyi_graph(
            num_nodes, probability_of_edge, seed=random.randint(0, 10000)
        )
        if not nx.is_connected(G) and G.number_of_nodes() > 0:
            print("    Warning: Generated random graph is not connected.")
            components = list(nx.connected_components(G))
            if len(components) > 1:
                print(
                    f"    Found {len(components)} components. Attempting to connect them."
                )
                for i in range(len(components) - 1):
                    node_from_c1 = random.choice(list(components[i]))
                    node_from_c2 = random.choice(list(components[i + 1]))
                    if (
                        not G.has_edge(node_from_c1, node_from_c2)
                        and node_from_c1 != node_from_c2
                    ):
                        G.add_edge(node_from_c1, node_from_c2)
                        print(
                            f"      Added bridge edge ({node_from_c1}, {node_from_c2})"
                        )
                if nx.is_connected(G):
                    print("    Graph is now connected.")
                else:
                    print(
                        "    Warning: Could not fully connect the graph with simple bridging."
                    )
    elif topology_type == "grid":
        m = int(num_nodes**0.5)
        n = (num_nodes + m - 1) // m
        if m > 0 and n > 0:
            G = nx.grid_2d_graph(m, n)
            node_map = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, node_map)
            if G.number_of_nodes() > num_nodes:
                G = G.subgraph(list(range(num_nodes))).copy()
        else:
            G = nx.path_graph(num_nodes) if num_nodes > 1 else nx.Graph()
    elif topology_type == "barabasi_albert" or topology_type == "barabasi":
        m_edges = barabasi_m
        if num_nodes <= 1:
            G = nx.Graph()
            if num_nodes == 1:
                G.add_node(0)
        elif m_edges >= num_nodes:
            G = nx.complete_graph(num_nodes)
        else:
            G = nx.barabasi_albert_graph(
                num_nodes, m_edges, seed=random.randint(0, 10000)
            )
    elif topology_type == "path":
        G = (
            nx.path_graph(num_nodes)
            if num_nodes > 1
            else (nx.Graph().add_node(0) if num_nodes == 1 else nx.Graph())
        )
    elif topology_type == "ring":
        if num_nodes > 2:
            G = nx.cycle_graph(num_nodes)
        elif num_nodes == 2:
            G = nx.path_graph(2)
        else:
            G = nx.Graph().add_node(0) if num_nodes == 1 else nx.Graph()
    elif topology_type == "k_nearest_neighbor":
        if num_nodes <= k_neighbors:
            G = nx.complete_graph(num_nodes)
        else:
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            pos = {i: (random.random(), random.random()) for i in range(num_nodes)}
            for u in range(num_nodes):
                distances = []
                for v in range(num_nodes):
                    if u == v:
                        continue
                    dist = (
                        (pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2
                    ) ** 0.5
                    distances.append((dist, v))
                distances.sort()
                for i in range(k_neighbors):
                    v = distances[i][1]
                    G.add_edge(u, v)
    elif topology_type == "rgg":
        G = nx.random_geometric_graph(num_nodes, connection_radius)
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            pos = {i: (random.random(), random.random()) for i in range(num_nodes)}
            nx.set_node_attributes(G, pos, "pos")
    elif topology_type == "multi_hub_star":
        if num_hubs <= 0 or num_hubs > num_nodes:
            print(f"    Warning: Invalid number of hubs ({num_hubs}). Setting to 1.")
            num_hubs = 1
        
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        hub_nodes = list(range(num_hubs))
        spoke_nodes = list(range(num_hubs, num_nodes))
        
        # Connect hubs
        if connect_hubs_complete and num_hubs > 1:
            for i in range(num_hubs):
                for j in range(i + 1, num_hubs):
                    G.add_edge(hub_nodes[i], hub_nodes[j])
        elif num_hubs > 1: # Default to a ring if not complete
            for i in range(num_hubs):
                G.add_edge(hub_nodes[i], hub_nodes[(i + 1) % num_hubs])

        # Connect spokes to hubs
        if spoke_nodes:
            for i, spoke_node in enumerate(spoke_nodes):
                hub_to_connect = hub_nodes[i % num_hubs]
                G.add_edge(spoke_node, hub_to_connect)
    else:
        print(f"Unknown topology type: {topology_type}. Defaulting to barabasi_albert.")
        m_edges = max(
            1, min(num_nodes - 1 if num_nodes > 1 else 1, int(probability_of_edge * 5))
        )
        if num_nodes == 1:
            m_edges = 0
        if num_nodes > 0 and m_edges < num_nodes:
            if num_nodes <= m_edges and num_nodes > 1:
                m_edges = num_nodes - 1
            if m_edges == 0 and num_nodes > 1:
                m_edges = 1
            if num_nodes == 0:
                G = nx.Graph()
            elif num_nodes == 1:
                G = nx.Graph()
                G.add_node(0)
            elif m_edges >= num_nodes:
                G = nx.complete_graph(num_nodes)
            else:
                G = nx.barabasi_albert_graph(
                    num_nodes, m_edges, seed=random.randint(0, 10000)
                )
        else:
            G = (
                nx.path_graph(num_nodes)
                if num_nodes > 1
                else (nx.Graph().add_node(0) if num_nodes == 1 else nx.Graph())
            )

    if G is None:
        G = nx.Graph()

    # グラフが連結であることを保証する
    if G.number_of_nodes() > 1: # ノードが1つの場合は常に連結
        G = ensure_graph_is_connected(G, min_weight, max_weight)

    if G.number_of_nodes() > 0 and min_degree_guarantee > 0:
        if min_degree_guarantee >= G.number_of_nodes() and G.number_of_nodes() > 1:
            print(
                f"    Warning: min_degree_guarantee ({min_degree_guarantee}) is >= number of nodes ({G.number_of_nodes()}). Making complete."
            )
            if not nx.is_isomorphic(G, nx.complete_graph(G.number_of_nodes())):
                CG = nx.complete_graph(G.nodes())
                G.add_edges_from(CG.edges())
        elif min_degree_guarantee < G.number_of_nodes():
            print(f"    Ensuring minimum degree of {min_degree_guarantee}...")
            ensure_min_degree(G, min_degree_guarantee)
            all_nodes_meet_min_degree = all(
                G.degree(node) >= min_degree_guarantee for node in G.nodes()
            )
            if all_nodes_meet_min_degree:
                print(
                    f"    Minimum degree of {min_degree_guarantee} successfully ensured."
                )
            else:
                print(
                    f"    Warning: Minimum degree of {min_degree_guarantee} NOT fully ensured after processing."
                )

    edges_data = []
    for u, v in G.edges():
        # 既存の重みがあればそれを使用し、なければランダムに生成
        if 'weight' in G[u][v]:
            weight = G[u][v]['weight']
        else:
            weight = round(random.uniform(min_weight, max_weight), 2)
        
        # 既存の名前があればそれを使用し、なければ生成
        if 'name' in G[u][v]:
            name = G[u][v]['name']
        else:
            name = f"L{u}-{v}"

        edges_data.append(
            {"source": u, "target": v, "weight": weight, "name": name}
        )
    with open(edge_output_path, "w", newline="") as ef:
        fieldnames = ["source", "target", "weight", "name"]
        writer = csv.DictWriter(ef, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges_data)
    print(
        f"Generated {edge_output_path} with {len(edges_data)} edges for {G.number_of_nodes()} nodes in the graph G."
    )
    print("File generation complete.")
    return G


def calculate_all_pairs_shortest_paths(graph):
    """
    グラフ内のすべてのノードペア間の最短経路長を計算し、リストとして返す。
    非連結なグラフの場合、到達不可能なペアはNoneとして返す。
    Args:
        graph (nx.Graph): 対象のNetworkXグラフオブジェクト。
    Returns:
        list: 各ペアの最短経路長を含む辞書のリスト。
              例: [{"source": 0, "target": 1, "length": 5}, ...]
              到達不可能な場合は "length": None
    """
    all_shortest_paths = []
    if graph.number_of_nodes() == 0:
        return all_shortest_paths

    nodes = list(graph.nodes())

    # 処理中の表示用カウンター
    processed_pairs = 0
    total_pairs = len(nodes) * (len(nodes) - 1) // 2

    print(f"\nCalculating all-pairs shortest paths... (Total {total_pairs} pairs)")

    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if (
                source_node < target_node
            ):  # 重複を避ける (source, target) と (target, source) は同じ
                processed_pairs += 1
                if processed_pairs % 1000 == 0:
                    print(f"  Processed {processed_pairs}/{total_pairs} pairs...")

                try:
                    length = nx.shortest_path_length(
                        graph, source=source_node, target=target_node
                    )
                    all_shortest_paths.append(
                        {"source": source_node, "target": target_node, "length": length}
                    )
                except nx.NetworkXNoPath:
                    all_shortest_paths.append(
                        {
                            "source": source_node,
                            "target": target_node,
                            "length": None,  # パスがない場合はNone
                        }
                    )
                except Exception as e:
                    print(
                        f"  Error calculating path between {source_node} and {target_node}: {e}"
                    )
                    all_shortest_paths.append(
                        {
                            "source": source_node,
                            "target": target_node,
                            "length": "ERROR",
                        }
                    )
    return all_shortest_paths


def export_diameter_endpoints(
    diameter_endpoints_data, output_filename="diameter_endpoints.csv"
):
    """
    直径を構成する端点のペアと直径の長さ（該当する連結成分の直径）をCSVファイルに出力する。
    Args:
        diameter_endpoints_data (list): 直径の端点データを含む辞書のリスト。
                                      例: [{"component_id": 1, "node1": 0, "node2": 5, "diameter": 3}]
        output_filename (str): 出力するCSVファイル名。
    """
    if not diameter_endpoints_data:
        print("No diameter endpoints data to export.")
        return

    print(f"Exporting diameter endpoints to {output_filename}...")
    with open(output_filename, "w", newline="") as defile:
        fieldnames = ["component_id", "node1", "node2", "diameter"]
        writer = csv.DictWriter(defile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diameter_endpoints_data)
    print(
        f"Successfully exported {len(diameter_endpoints_data)} diameter endpoint entries."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate network graph files for simulation.")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes in the graph.")
    parser.add_argument("--min_weight", type=float, default=0.5, help="Minimum edge weight.")
    parser.add_argument("--max_weight", type=float, default=0.5, help="Maximum edge weight.")
    parser.add_argument("--node_output_path", type=str, default="node.csv", help="Path to save the node CSV file.")
    parser.add_argument("--edge_output_path", type=str, default="edge.csv", help="Path to save the edge CSV file.")
    parser.add_argument("--topology_type", type=str, default="barabasi_albert", help="Type of topology to generate.")
    parser.add_argument("--probability_of_edge", type=float, default=0.2, help="Probability of edge creation for random graphs.")
    parser.add_argument("--k_neighbors", type=int, default=3, help="Number of neighbors for k-nearest neighbor graphs.")
    parser.add_argument("--connection_radius", type=float, default=0.15, help="Connection radius for random geometric graphs.")
    parser.add_argument("--barabasi_m", type=int, default=2, help="Number of edges to attach from a new node to existing nodes for Barabasi-Albert model.")
    args = parser.parse_args()

    print(f"\n--- Generating Graph for Topology: {args.topology_type} ---")

    # generate_sample_input_files に渡すパラメータを準備
    generated_graph = generate_sample_input_files(
        num_nodes=args.num_nodes,
        probability_of_edge=args.probability_of_edge,
        min_degree_guarantee=0,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        node_output_path=args.node_output_path,
        edge_output_path=args.edge_output_path,
        topology_type=args.topology_type,
        k_neighbors=args.k_neighbors,
        connection_radius=args.connection_radius,
        barabasi_m=args.barabasi_m,
    )

    # The rest of the script (diameter calculation, etc.) remains the same...
    output_dir = os.path.dirname(args.node_output_path)
    diameter_endpoints_file_out = os.path.join(output_dir, "diameter_endpoints.csv")
    shortest_paths_file_out = os.path.join(output_dir, "shortest_paths.csv")
    
    diameter_endpoints_to_export = []

    # --- 直径の計算と出力処理 ---
    print("\nAttempting to calculate network diameter and identify endpoints...")
    if generated_graph.number_of_nodes() > 0:
        components = list(nx.connected_components(generated_graph))
        if len(components) == 1:
            print("Graph is connected.")
            try:
                diameter = nx.diameter(generated_graph)
                print(f"Network Diameter: {diameter}")

                all_paths_in_component = calculate_all_pairs_shortest_paths(
                    generated_graph
                )
                max_length_found = -1
                for path_info in all_paths_in_component:
                    if (
                        path_info["length"] is not None
                        and path_info["length"] != "ERROR"
                    ):
                        if path_info["length"] > max_length_found:
                            max_length_found = path_info["length"]

                if max_length_found != -1:
                    identified_endpoints = []
                    for path_info in all_paths_in_component:
                        if path_info["length"] == max_length_found:
                            identified_endpoints.append(
                                {
                                    "component_id": "main_graph",
                                    "node1": path_info["source"],
                                    "node2": path_info["target"],
                                    "diameter": max_length_found,
                                }
                            )
                    diameter_endpoints_to_export.extend(identified_endpoints)
                    print(
                        f"  Identified {len(identified_endpoints)} pairs of endpoints for main graph diameter {max_length_found}:"
                    )
                    for ep in identified_endpoints:
                        print(f"    ({ep['node1']}, {ep['node2']})")
                else:
                    print("  Could not identify diameter endpoints for the main graph.")

            except Exception as e:
                print(
                    f"Error calculating diameter or identifying endpoints for the connected graph: {e}"
                )
                traceback.print_exc()
        else:
            print(
                f"Graph is not connected. Found {len(components)} components. Calculating diameter for each."
            )
            for i, component_nodes in enumerate(components):
                subgraph = generated_graph.subgraph(component_nodes).copy()
                if subgraph.number_of_nodes() > 1:
                    try:
                        comp_diameter = nx.diameter(subgraph)
                        print(
                            f"  Diameter of Component {i+1} (Nodes: {len(component_nodes)}): {comp_diameter}"
                        )

                        all_paths_in_component = calculate_all_pairs_shortest_paths(
                            subgraph
                        )
                        max_length_found_comp = -1
                        for path_info in all_paths_in_component:
                            if (
                                path_info["length"] is not None
                                and path_info["length"] != "ERROR"
                            ):
                                if path_info["length"] > max_length_found_comp:
                                    max_length_found_comp = path_info["length"]

                        if max_length_found_comp != -1:
                            identified_endpoints_comp = []
                            for path_info in all_paths_in_component:
                                if path_info["length"] == max_length_found_comp:
                                    identified_endpoints_comp.append(
                                        {
                                            "component_id": i + 1,
                                            "node1": path_info["source"],
                                            "node2": path_info["target"],
                                            "diameter": max_length_found_comp,
                                        }
                                    )
                            diameter_endpoints_to_export.extend(
                                identified_endpoints_comp
                            )
                            print(
                                f"    Identified {len(identified_endpoints_comp)} pairs of endpoints for Component {i+1} diameter {max_length_found_comp}:"
                            )
                            for ep in identified_endpoints_comp:
                                print(f"      ({ep['node1']}, {ep['node2']})")
                        else:
                            print(
                                f"  Could not identify diameter endpoints for Component {i+1}."
                            )
                    except Exception as e:
                        print(
                            f"  Error calculating diameter or identifying endpoints for Component {i+1}: {e}"
                        )
                        traceback.print_exc()
                else:
                    print(
                        f"  Component {i+1} has only one node, diameter is 0 (or undefined)."
                    )
    else:
        print("Graph is empty, cannot calculate diameter or identify endpoints.")
    # --- 直径の計算と出力処理ここまで ---

    # --- すべてのノードペアの最短経路長をCSVに出力 ---
    all_pairs_paths_data = calculate_all_pairs_shortest_paths(generated_graph)
    if all_pairs_paths_data:
        print(f"\nExporting all-pairs shortest paths to {shortest_paths_file_out}...")
        with open(shortest_paths_file_out, "w", newline="") as spf:
            fieldnames = ["source", "target", "length"]
            writer = csv.DictWriter(spf, fieldnames=fieldnames)
            writer.writeheader()
            rows_to_write = []
            for row in all_pairs_paths_data:
                if row["length"] is None:
                    row["length"] = -1
                rows_to_write.append(row)
            writer.writerows(rows_to_write)
        print(
            f"Successfully exported {len(all_pairs_paths_data)} all-pairs shortest path lengths."
        )
    else:
        print("No all-pairs shortest path data to export.")
    # --- 最短経路長計算と出力処理ここまで ---

    # --- 直径の端点をCSVに出力 ---
    export_diameter_endpoints(diameter_endpoints_to_export, diameter_endpoints_file_out)
    # --- 直径の端点出力処理ここまで ---

    # プロット処理はオーケストレーションスクリプトで制御するため、ここではコメントアウト
    # print("\nAttempting to plot the generated graph...")
    # try:
    #     graph_to_plot = generated_graph
    #     if graph_to_plot.number_of_nodes() > 0:
    #         plt.figure(figsize=(12, 10))
    #         pos = nx.spring_layout(graph_to_plot, k=0.5, iterations=50, seed=42)
    #         nx.draw_networkx_nodes(graph_to_plot, pos, node_size=500, node_color="skyblue", alpha=0.9)
    #         nx.draw_networkx_edges(graph_to_plot, pos, alpha=0.6, edge_color="gray")
    #         node_labels = nx.get_node_attributes(graph_to_plot, "label")
    #         nx.draw_networkx_labels(graph_to_plot, pos, labels=node_labels, font_size=8)
    #         edge_weights = nx.get_edge_attributes(graph_to_plot, "weight")
    #         nx.draw_networkx_edge_labels(graph_to_plot, pos, edge_labels=edge_weights, font_size=7, font_color="red")
    #         plt.title(f"Generated Network Graph\n({args.topology_type}, Nodes: {graph_to_plot.number_of_nodes()}, Edges: {graph_to_plot.number_of_edges()})")
    #         plt.axis("off")
    #         plt.tight_layout()
    #         plt.show()
    #         print("Plot displayed. Close the plot window to exit.")
    #     else:
    #         print("No nodes to plot.")
    # except Exception as e:
    #     print(f"An error occurred during plotting: {e}")
    #     traceback.print_exc()

    print("\n--- All graph generation and analysis completed ---")

