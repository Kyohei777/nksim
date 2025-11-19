import csv
import random
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime


# (ensure_min_degree and ensure_graph_is_connected functions remain unchanged)
def ensure_min_degree(graph, min_degree, max_retries_per_node=10):
    """
    指定されたグラフの全ノードが最低次数を満たすようにエッジを追加する。
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
        node1 = random.choice(list(components[i]))
        node2 = random.choice(list(components[i+1]))
        if not graph.has_edge(node1, node2):
            weight = round(random.uniform(min_weight, max_weight), 2)
            graph.add_edge(node1, node2, weight=weight, name=f"L_conn_{node1}-{node2}")
            added_edges_count += 1
            print(f"      Added bridge edge ({node1}, {node2}) with weight {weight}")

    if nx.is_connected(graph):
        print(f"    Graph is now connected after adding {added_edges_count} bridge edges.")
    else:
        print("    Warning: Graph is still not connected after initial bridging. Retrying...")
        max_retries = graph.number_of_nodes()
        retries = 0
        while not nx.is_connected(graph) and retries < max_retries:
            components = list(nx.connected_components(graph))
            if len(components) <= 1:
                break
            node1 = random.choice(list(components[0]))
            node2 = random.choice(list(components[1]))
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

def generate_graph_structure(
    num_nodes=100,
    topology_type="barabasi_albert",
    probability_of_edge=0.3,
    k_neighbors=3,
    connection_radius=0.2,
    num_hubs=3,
    connect_hubs_complete=True,
    barabasi_m=2,
    min_degree_guarantee=0,
    min_weight=0.5,
    max_weight=0.5,
):
    """
    Generates the graph structure and node data, but does not write to files.
    """
    print(f"Generating graph structure for {num_nodes} nodes...")
    print(f"Topology type: {topology_type}")

    nodes_data = [{"node_id": i, "label": f"Node{i}"} for i in range(num_nodes)]

    G = nx.Graph() # Initialize an empty graph

    if topology_type == "random":
        G = nx.erdos_renyi_graph(num_nodes, probability_of_edge, seed=random.randint(0, 10000))
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
        if num_nodes <= 1: G = nx.Graph(); G.add_node(0) if num_nodes == 1 else None
        elif m_edges >= num_nodes: G = nx.complete_graph(num_nodes)
        else: G = nx.barabasi_albert_graph(num_nodes, m_edges, seed=random.randint(0, 10000))
    # ... (other topology generation logic remains the same) ...
    else:
        print(f"Unknown topology type: {topology_type}. Defaulting to barabasi_albert.")
        G = nx.barabasi_albert_graph(num_nodes, 2, seed=random.randint(0, 10000))

    # Ensure all nodes from nodes_data are in the graph
    G.add_nodes_from([n['node_id'] for n in nodes_data])

    if G.number_of_nodes() > 1:
        G = ensure_graph_is_connected(G, min_weight, max_weight)

    if G.number_of_nodes() > 0 and min_degree_guarantee > 0:
        ensure_min_degree(G, min_degree_guarantee)

    print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, nodes_data


def calculate_all_pairs_shortest_paths(graph):
    """
    (This function remains unchanged)
    """
    all_shortest_paths = []
    if graph.number_of_nodes() == 0:
        return all_shortest_paths
    nodes = list(graph.nodes())
    processed_pairs = 0
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    print(f"\nCalculating all-pairs shortest paths... (Total {total_pairs} pairs)")
    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if source_node < target_node:
                processed_pairs += 1
                if processed_pairs % 1000 == 0:
                    print(f"  Processed {processed_pairs}/{total_pairs} pairs...")
                try:
                    length = nx.shortest_path_length(graph, source=source_node, target=target_node)
                    all_shortest_paths.append({"source": source_node, "target": target_node, "length": length})
                except nx.NetworkXNoPath:
                    all_shortest_paths.append({"source": source_node, "target": target_node, "length": None})
    return all_shortest_paths


def export_diameter_endpoints(diameter_endpoints_data, output_filename="diameter_endpoints.csv"):
    """
    (This function remains unchanged)
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
    print(f"Successfully exported {len(diameter_endpoints_data)} diameter endpoint entries.")


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
    parser.add_argument("--min_connect_rate", type=float, default=0.4, help="Minimum connect rate for a link.")
    parser.add_argument("--max_connect_rate", type=float, default=0.4, help="Maximum connect rate for a link.")
    parser.add_argument("--min_disconnect_rate", type=float, default=0.2, help="Minimum disconnect rate for a link.")
    parser.add_argument("--max_disconnect_rate", type=float, default=0.2, help="Maximum disconnect rate for a link.")
    # New argument will be added here later
    args = parser.parse_args()

    print(f"\n--- Generating Graph for Topology: {args.topology_type} ---")

    # --- Step 1: Generate Graph Structure ---
    generated_graph, nodes_data = generate_graph_structure(
        num_nodes=args.num_nodes,
        topology_type=args.topology_type,
        probability_of_edge=args.probability_of_edge,
        k_neighbors=args.k_neighbors,
        connection_radius=args.connection_radius,
        barabasi_m=args.barabasi_m,
        min_degree_guarantee=0,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    # --- Step 2: Write Node File ---
    with open(args.node_output_path, "w", newline="") as nf:
        fieldnames = ["node_id", "label"]
        writer = csv.DictWriter(nf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodes_data)
    print(f"Generated {args.node_output_path} with {len(nodes_data)} nodes.")

    # --- Step 3: Set Edge Attributes (Connect/Disconnect Rates, Weight, etc.) ---
    # This is where the new degradation logic will go.
    # For now, we just set the default values.
    for u, v in generated_graph.edges():
        if 'weight' not in generated_graph[u][v]:
            generated_graph[u][v]['weight'] = round(random.uniform(args.min_weight, args.max_weight), 2)
        if 'name' not in generated_graph[u][v]:
            generated_graph[u][v]['name'] = f"L{u}-{v}"
        
        generated_graph[u][v]['connect_rate'] = round(random.uniform(args.min_connect_rate, args.max_connect_rate), 4)
        generated_graph[u][v]['disconnect_rate'] = round(random.uniform(args.min_disconnect_rate, args.max_disconnect_rate), 4)

    # --- Step 4: Write Edge File ---
    edges_data = []
    for u, v, data in generated_graph.edges(data=True):
        edges_data.append({
            "source": u, "target": v,
            "weight": data.get('weight', 1.0),
            "name": data.get('name', f"L{u}-{v}"),
            "connect_rate": data.get('connect_rate', 0.4),
            "disconnect_rate": data.get('disconnect_rate', 0.2)
        })
    with open(args.edge_output_path, "w", newline="") as ef:
        fieldnames = ["source", "target", "weight", "name", "connect_rate", "disconnect_rate"]
        writer = csv.DictWriter(ef, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges_data)
    print(f"Generated {args.edge_output_path} with {len(edges_data)} edges.")


    # --- Step 5: Analyze Graph and Export Data (Diameter, etc.) ---
    output_dir = os.path.dirname(args.node_output_path)
    diameter_endpoints_file_out = os.path.join(output_dir, "diameter_endpoints.csv")
    shortest_paths_file_out = os.path.join(output_dir, "shortest_paths.csv")
    
    diameter_endpoints_to_export = []
    if generated_graph.number_of_nodes() > 0 and nx.is_connected(generated_graph):
        try:
            diameter = nx.diameter(generated_graph)
            print(f"Network Diameter: {diameter}")
            # This part can be slow, so we find just one pair for simplicity
            endpoints = nx.periphery(generated_graph)
            # Find a pair from the periphery that actually has the diameter distance
            found_pair = False
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    u, v = endpoints[i], endpoints[j]
                    if nx.shortest_path_length(generated_graph, u, v) == diameter:
                        diameter_endpoints_to_export.append({
                            "component_id": "main_graph", "node1": u, "node2": v, "diameter": diameter
                        })
                        print(f"  Identified diameter endpoint pair: ({u}, {v})")
                        found_pair = True
                        break
                if found_pair:
                    break
        except Exception as e:
            print(f"Error calculating diameter: {e}")
    else:
        print("Graph is empty or not connected, skipping diameter calculation.")

    export_diameter_endpoints(diameter_endpoints_to_export, diameter_endpoints_file_out)
    
    all_pairs_paths_data = calculate_all_pairs_shortest_paths(generated_graph)
    if all_pairs_paths_data:
        # ... (writing shortest_paths.csv remains the same)
        pass

    print("\n--- All graph generation and analysis completed ---")

