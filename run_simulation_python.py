import os
import sys
import csv

# Add the current directory to the Python path to import run_custom_sweep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import run_custom_sweep

# --- Configuration ---
NODE_FILE = "node.csv"
EDGE_FILE = "edge.csv"
ENDPOINTS_FILE = "diameter_endpoints.csv"

def main():
    print("--- Starting Python-driven Simulation Run ---")

    # --- Read Source and Destination from CSV ---
    print(f"Reading source and destination from {ENDPOINTS_FILE}...")

    source_node = None
    dest_node = None

    try:
        with open(ENDPOINTS_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            try:
                first_data_row = next(reader)
                # Assuming node1 is the second column (index 1) and node2 is the third (index 2)
                source_node = int(first_data_row[1])
                dest_node = int(first_data_row[2])
            except StopIteration:
                print(f"Error: {ENDPOINTS_FILE} contains only a header or is empty.")
                sys.exit(1)
            except ValueError as e:
                print(f"Error: Could not parse node IDs from {ENDPOINTS_FILE}. Check data format. Error: {e}")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: {ENDPOINTS_FILE} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {ENDPOINTS_FILE}: {e}")
        sys.exit(1)

    print(f"Using Source: {source_node}, Destination: {dest_node}")

    # --- Run the custom sweep simulation ---
    print("Starting custom sweep simulation...")
    # Call the main function of run_custom_sweep.py directly
    run_custom_sweep.main(NODE_FILE, EDGE_FILE, source_node, dest_node)

    print("--- Python-driven Simulation Run Finished ---")

if __name__ == "__main__":
    main()
