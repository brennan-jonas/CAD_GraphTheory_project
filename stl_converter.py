from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the STL file
file_path = 'cube.stl'  # Replace with the actual file path
your_mesh = mesh.Mesh.from_file(file_path)

# Access the vertices
vertices = your_mesh.vectors.reshape(-1, 3)  # Extract vertices (Nx3 array)
print("Vertices:\n", vertices)

# Example: Number of faces and vertices
print("Number of faces:", your_mesh.vectors.shape[0])
print("Total vertices:", vertices.shape[0])

# Optional: Calculate adjacency matrix (e.g., based on proximity or shared edges)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the STL mesh
for vector in your_mesh.vectors:
    print(vector)
    tri = Poly3DCollection([vector])
    tri.set_color('cyan')
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

# Set axis limits
ax.auto_scale_xyz(your_mesh.x.flatten(), your_mesh.y.flatten(), your_mesh.z.flatten())
plt.show()
