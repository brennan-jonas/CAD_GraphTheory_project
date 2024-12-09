import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# Function to process STL into graph data
def stl_to_graph(stl_mesh):
    vertices = stl_mesh.vectors.reshape(-1, 3)  # Flatten the vertices array (x, y, z)

    # Create adjacency matrix (simple nearest neighbor)
    num_vertices = len(vertices)
    adj_matrix = np.zeros((num_vertices, num_vertices))

    # Create edges based on shared vertices between faces
    for i, triangle in enumerate(stl_mesh.vectors):
        for j in range(3):
            for k in range(j + 1, 3):
                vi = np.argmin(np.linalg.norm(vertices - triangle[j], axis=1))
                vk = np.argmin(np.linalg.norm(vertices - triangle[k], axis=1))
                adj_matrix[vi, vk] = adj_matrix[vk, vi] = 1

    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    # Normalize the adjacency matrix
    adj_matrix = normalize_adjacency(adj_matrix)

    # Node features: using the vertices' coordinates (x, y, z)
    features = torch.tensor(vertices, dtype=torch.float32)

    return features, adj_matrix


# Function for adjacency matrix normalization
def normalize_adjacency(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


# GCN model class
class GCNBlock(nn.Module):
    def __init__(self, in_features, out_features, edge_features=None):
        super(GCNBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.edge_linear = nn.Linear(edge_features, out_features) if edge_features else None

    def forward(self, x, adj, edge_attr=None):
        # Node feature transformation
        x = self.linear(x)
        if edge_attr is not None and self.edge_linear is not None:
            # Incorporating edge features (if provided)
            edge_effect = self.edge_linear(edge_attr)
            x += torch.matmul(adj, edge_effect)
        # Graph convolution operation
        x = torch.matmul(adj, x)
        x = F.relu(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_features=None):
        super(GCN, self).__init__()
        self.gcn1 = GCNBlock(input_dim, hidden_dim, edge_features)
        self.gcn2 = GCNBlock(hidden_dim, output_dim, edge_features)

    def forward(self, x, adj, edge_attr=None):
        x = self.gcn1(x, adj, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, adj, edge_attr)
        return x


# Function to visualize the 3D graph
def visualize_graph(features, adj, title="Graph Visualization", use_pca=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Ensure features is a NumPy array, and detach it from the computation graph if it's a tensor
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Check for NaN or Inf values and handle them
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("Warning: NaN or Inf values found in features. Replacing with zeros.")
        features = np.nan_to_num(features)  # Replace NaN with 0 and Inf with a large number

    if use_pca:
        # Reduce to 3D using PCA if features are higher than 3D
        pca = PCA(n_components=3)
        features = pca.fit_transform(features)

    # Extracting the x, y, z coordinates of the nodes
    x_vals = features[:, 0]
    y_vals = features[:, 1]
    z_vals = features[:, 2]

    # Plot the nodes (scatter plot)
    ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')

    # Plot the edges (based on the adjacency matrix)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if adj[i, j] > 0:  # Only plot an edge if there's a connection
                ax.plot([x_vals[i], x_vals[j]], [y_vals[i], y_vals[j]], [z_vals[i], z_vals[j]], 'k-', lw=0.5)

    ax.set_title(title)
    plt.show()


# Load STL file
stl_path = 'Twisted_Vase_Basic.stl'
your_mesh = mesh.Mesh.from_file(stl_path)

# Process the STL model to get graph data
features, adj = stl_to_graph(your_mesh)

# Example setup
num_nodes = features.shape[0]  # Number of nodes in the graph (vertices in the mesh)
input_dim = 3  # Number of node features (x, y, z)
hidden_dim = 16  # Hidden layer size
output_dim = 3  # Output size (e.g., pocket classification: 3 classes)
edge_features_dim = None  # No edge features in this case

# Random labels for nodes (example, replace with actual labels if available)
labels = torch.randint(0, output_dim, (num_nodes,))

# Define the model
model = GCN(input_dim, hidden_dim, output_dim, edge_features=edge_features_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Placeholder for previous epoch's output to compute differences
prev_output = None

# Training loop (for illustration)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(features, adj)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimizer step

    # Optionally visualize during training
    if epoch % 100 == 0:
        visualize_graph(features, adj, title=f"Graph Before GCN Epoch {epoch + 1}", use_pca=True)

    # Print differences between epochs
    if prev_output is not None:
        diff = torch.abs(outputs - prev_output).mean().item()
        print(f'Epoch {epoch + 1}: Difference between epochs: {diff}')

    # Save current output for comparison in the next epoch
    prev_output = outputs.clone()

# Visualize after GCN processing
outputs = model(features, adj)
visualize_graph(outputs, adj, title="Graph After GCN", use_pca=True)
