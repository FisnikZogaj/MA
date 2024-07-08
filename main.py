import subprocess
import torch
from torch_geometric.data import Data
import os
import pickle
# Running main.py will generate ...


#base = r"C:\Users\zogaj\PycharmProjects\MA\SyntheticGraphs"
#os.makedirs(base, exist_ok=True)
#graph_path = os.path.join(base, "graph.pkl")
#with open(graph_path, 'wb') as file:
#    pickle.dump(graph, file)

def save_data_object(data_object, file_path):
    torch.save(data_object, file_path)

# Create your DataObject (example data)
num_nodes = 100
num_features = 16
x = torch.randn((num_nodes, num_features))
edge_index = torch.randint(0, num_nodes, (2, 200))
y = torch.randint(0, 3, (num_nodes,))
data_object = Data(x=x, edge_index=edge_index, y=y)

# Save the DataObject to a file
data_path = 'data_object.pt'
save_data_object(data_object, data_path)

# Different string parameters for each job
params = ["GCN", "GAT", "SAGE"]

# List to keep track of subprocesses
processes = []

# Launch train.py in parallel with different parameters
for param in params:
    command = [
        'python', 'train.py',  # The script to run
        '--data-path', data_path,  # Path to the data file
        '--param', param  # Additional string parameter
    ]
    # Start the subprocess
    process = subprocess.Popen(command)
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All training jobs have completed.")

