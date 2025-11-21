import csv
import random
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from collections import deque

def ensure_min_degree(graph, min_degree, max_retries_per_node=10):
    """
    Ensures all nodes in the graph have at least the minimum specified degree.
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
                    added_count_for_u += 1
                    modified = True
        if not modified:
            break

def ensure_graph_is_connected(graph, min_weight, max_weight):
    """
    Ensures the graph is connected by adding bridges between components if necessary.
    """
    if graph.number_of_nodes() <= 1 or nx.is_connected(graph):
        return graph

    print("    Graph is not connected. Attempting to connect components...")
    components = list(nx.connected_components(graph))
    
    for i in range(len(components) - 1):
        node1 = random.choice(list(components[i]))
        node2 = random.choice(list(components[i+1]))
        if not graph.has_edge(node1, node2):
            weight = round(random.uniform(min_weight, max_weight), 2)
            graph.add_edge(node1, node2, weight=weight, name=f"L_conn_{node1}-{node2}")

    if not nx.is_connected(graph):
         print("    Warning: Graph could not be connected after initial bridging.")
    
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
    seed=None,
):
    """
    Generates the graph structure and node data, but does not write to files.
    """
    print(f"Generating graph structure for {num_nodes} nodes...")
    print(f"Topology type: {topology_type}")

    nodes_data = [{"node_id": i, "label": f"Node{i}"} for i in range(num_nodes)]
    G = nx.Graph()
    
    # Graph generation logic based on topology_type
    if topology_type == "random":
        G = nx.erdos_renyi_graph(num_nodes, probability_of_edge, seed=seed)
    elif topology_type == "grid":
        m = int(num_nodes**0.5)
        n = (num_nodes + m - 1) // m
        G = nx.grid_2d_graph(m, n, seed=seed)
        node_map = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, node_map)
        if G.number_of_nodes() > num_nodes:
            G = G.subgraph(list(range(num_nodes))).copy()
    elif topology_type == "barabasi_albert":
        if num_nodes > barabasi_m:
            G = nx.barabasi_albert_graph(num_nodes, barabasi_m, seed=seed)
        else:
            G = nx.complete_graph(num_nodes)
    elif topology_type == "path":
        G = nx.path_graph(num_nodes)
    elif topology_type == "ring":
        G = nx.cycle_graph(num_nodes)
    elif topology_type == "k_nearest_neighbor":
        k = max(2, k_neighbors // 2 * 2) # Ensure k is an even integer >= 2
        if num_nodes > k:
            G = nx.watts_strogatz_graph(num_nodes, k, 0, seed=seed)
        else:
            G = nx.complete_graph(num_nodes) # fallback for small n
    elif topology_type == "rgg":
        G = nx.random_geometric_graph(num_nodes, connection_radius, seed=seed)
    else:
        print(f"Warning: Unknown or unsupported topology_type '{topology_type}'. Defaulting to barabasi_albert.")
        G = nx.barabasi_albert_graph(num_nodes, 2, seed=seed)

    G.add_nodes_from([n['node_id'] for n in nodes_data])

    if G.number_of_nodes() > 1:
        G = ensure_graph_is_connected(G, min_weight, max_weight)

    if G.number_of_nodes() > 0 and min_degree_guarantee > 0:
        ensure_min_degree(G, min_degree_guarantee)

    print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, nodes_data

def calculate_all_pairs_shortest_paths(graph):
    """
    Calculates shortest path lengths for all pairs of nodes.
    """
    all_shortest_paths = []
    if graph.number_of_nodes() == 0:
        return all_shortest_paths
    nodes = list(graph.nodes())
    
    try:
        # This is much faster if the graph is connected
        path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        for source_node, paths in path_lengths.items():
            for target_node, length in paths.items():
                if source_node < target_node:
                    all_shortest_paths.append({"source": source_node, "target": target_node, "length": length})
    except nx.NetworkXError: # Handles disconnected graphs
        for i, source_node in enumerate(nodes):
            for j, target_node in enumerate(nodes):
                if source_node < target_node:
                    try:
                        length = nx.shortest_path_length(graph, source=source_node, target=target_node)
                        all_shortest_paths.append({"source": source_node, "target": target_node, "length": length})
                    except nx.NetworkXNoPath:
                        all_shortest_paths.append({"source": source_node, "target": target_node, "length": None})
    return all_shortest_paths

def export_diameter_endpoints(diameter_endpoints_data, output_filename="diameter_endpoints.csv"):
    """
    Exports diameter endpoints to a CSV file.
    """
    if not diameter_endpoints_data:
        return
    with open(output_filename, "w", newline="") as defile:
        fieldnames = ["component_id", "node1", "node2", "diameter"]
        writer = csv.DictWriter(defile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diameter_endpoints_data)
    print(f"Successfully exported {len(diameter_endpoints_data)} diameter endpoint entries to {output_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate network graph files for simulation with optional edge quality degradation.")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes in the graph.")
    parser.add_argument("--min_weight", type=float, default=0.5, help="Minimum edge weight.")
    parser.add_argument("--max_weight", type=float, default=0.5, help="Maximum edge weight.")
    parser.add_argument("--node_output_path", type=str, default="node.csv", help="Path to save the node CSV file.")
    parser.add_argument("--edge_output_path", type=str, default="edge.csv", help="Path to save the edge CSV file.")
    parser.add_argument("--topology_type", type=str, default="barabasi_albert", help="Type of topology to generate.")
    parser.add_argument("--barabasi_m", type=int, default=2, help="Number of edges for Barabasi-Albert model.")
    parser.add_argument("--k_neighbors", type=int, default=3, help="Number of nearest neighbors for k-nearest neighbor graph.")
    parser.add_argument("--connection_radius", type=float, default=0.2, help="Connection radius for Random Geometric Graph (RGG).")
    parser.add_argument("--min_connect_rate", type=float, default=0.4, help="Default minimum connect rate.")
    parser.add_argument("--max_connect_rate", type=float, default=0.4, help="Default maximum connect rate.")
    parser.add_argument("--min_disconnect_rate", type=float, default=0.2, help="Default minimum disconnect rate.")
    parser.add_argument("--max_disconnect_rate", type=float, default=0.2, help="Default maximum disconnect rate.")
    parser.add_argument("--degrade-edges", action='store_true', help="Enable edge quality degradation based on distance from diameter path.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generators.")
    args = parser.parse_args()

    # Set the seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    print(f"\n--- Generating Graph for Topology: {args.topology_type} ---")

    # Step 1: Generate Graph Structure
    generated_graph, nodes_data = generate_graph_structure(
        num_nodes=args.num_nodes,
        topology_type=args.topology_type,
        barabasi_m=args.barabasi_m,
        k_neighbors=args.k_neighbors,
        connection_radius=args.connection_radius,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        seed=args.seed,
    )

    # Step 2: Write Node File
    with open(args.node_output_path, "w", newline="") as nf:
        fieldnames = ["node_id", "label"]
        writer = csv.DictWriter(nf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodes_data)
    print(f"Generated {args.node_output_path}")

    # Step 3: Analyze Graph to find Diameter
    diameter_endpoints_to_export = []
    if generated_graph.number_of_nodes() > 0 and nx.is_connected(generated_graph):
        try:
            diameter = nx.diameter(generated_graph)
            periphery_nodes = nx.periphery(generated_graph)
            found_pair = False
            for i in range(len(periphery_nodes)):
                for j in range(i + 1, len(periphery_nodes)):
                    u, v = periphery_nodes[i], periphery_nodes[j]
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

    # Step 4: Set Edge Attributes
    if args.degrade_edges and diameter_endpoints_to_export:
        print("Degrading edge quality based on distance from diameter path...")
        
        quality_levels = {
            0: {'c': 0.1, 'd': 0.9},
            1: {'c': 0.175, 'd': 0.725},
            2: {'c': 0.25, 'd': 0.55},
            3: {'c': 0.325, 'd': 0.375},
            4: {'c': 0.4, 'd': 0.2}
        }
        
        ref_pair = diameter_endpoints_to_export[0]
        src, dst = ref_pair['node1'], ref_pair['node2']
        
        ref_path_nodes = nx.shortest_path(generated_graph, source=src, target=dst)
        print(f"  Reference path for degradation: {ref_path_nodes}")

        distances = {node: -1 for node in generated_graph.nodes()}
        q = deque()
        for node in ref_path_nodes:
            q.append(node)
            distances[node] = 0
        
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v_neighbor in generated_graph.neighbors(u):
                if distances[v_neighbor] == -1:
                    distances[v_neighbor] = distances[u] + 1
                    q.append(v_neighbor)

        for u, v in generated_graph.edges():
            level = min(distances[u], distances[v])
            if level > 4: level = 4
            
            base_q = quality_levels[level]
            connect_rate = base_q['c'] * random.uniform(0.8, 1.2)
            disconnect_rate = base_q['d'] * random.uniform(0.8, 1.2)

            if level == 4 and random.random() < 0.05:
                connect_rate, disconnect_rate = 0.5, 0.1

            generated_graph[u][v]['connect_rate'] = round(connect_rate, 4)
            generated_graph[u][v]['disconnect_rate'] = round(disconnect_rate, 4)
    else:
        if args.degrade_edges:
            print("Warning: --degrade-edges specified, but no diameter path found. Using default rates.")
        print("Using default connect/disconnect rates for all edges.")
        for u, v in generated_graph.edges():
            generated_graph[u][v]['connect_rate'] = round(random.uniform(args.min_connect_rate, args.max_connect_rate), 4)
            generated_graph[u][v]['disconnect_rate'] = round(random.uniform(args.min_disconnect_rate, args.max_disconnect_rate), 4)

    for u, v in generated_graph.edges():
        if 'weight' not in generated_graph[u][v]:
            generated_graph[u][v]['weight'] = round(random.uniform(args.min_weight, args.max_weight), 2)
        if 'name' not in generated_graph[u][v]:
            generated_graph[u][v]['name'] = f"L{u}-{v}"

    # Step 5: Write Edge File
    edges_data = []
    for u, v, data in generated_graph.edges(data=True):
        edges_data.append({
            "source": u, "target": v,
            "weight": data.get('weight', 1.0), "name": data.get('name', f"L{u}-{v}"),
            "connect_rate": data.get('connect_rate', 0.4), "disconnect_rate": data.get('disconnect_rate', 0.2)
        })
    with open(args.edge_output_path, "w", newline="") as ef:
        fieldnames = ["source", "target", "weight", "name", "connect_rate", "disconnect_rate"]
        writer = csv.DictWriter(ef, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges_data)
    print(f"Generated {args.edge_output_path}")

    # Step 6: Export Analysis Data
    output_dir = os.path.dirname(args.node_output_path) if os.path.dirname(args.node_output_path) else '.'
    diameter_endpoints_file_out = os.path.join(output_dir, "diameter_endpoints.csv")
    export_diameter_endpoints(diameter_endpoints_to_export, diameter_endpoints_file_out)
    
    print("\n--- All graph generation and analysis completed ---")
