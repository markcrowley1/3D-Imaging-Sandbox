"""
Description:
    View images and their corresponding ground truth point clouds
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation

POINTS = 512

def rotate(points: np.ndarray, elevation: float, azimuth: float) -> np.ndarray:
    # Rotate to base position
    r = Rotation.from_euler("xyz", [-90, 90, 0], degrees= True)
    r = np.array(r.as_matrix())
    points = np.matmul(points, r)
    # Rotate to rendering position
    r = Rotation.from_euler("xyz", [0, -elevation, -azimuth],
                            degrees= True)
    r = np.array(r.as_matrix())
    points = np.matmul(points, r)
    return points

def visualise_data(img_file: str, points: np.ndarray):
    # Show rendering of textured 3D mesh
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Rendering")
    imgplot = mpimg.imread(img_file)
    plt.imshow(imgplot)

    # Show corresponding ground truth 3d point cloud
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10)
    ax.set_axis_on()
    ax.set_aspect("equal")
    ax.set_title("Ground Truth Point Cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.view_init(0, 0)

    plt.show()

def main():
    # Paths to data
    rendering = "D:/ShapeNetRendering/ShapeNetRendering/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/rendering/00.png"
    obj_file = "D:/shapenet_base/shapenet_core/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/model_normalized.obj"
    # Sample ground truth point cloud
    mesh = trimesh.load(obj_file, force="mesh")
    sample = trimesh.sample.sample_surface(mesh, POINTS)
    sampled_points = sample[0]
    points = np.array(sampled_points)
    # Compare image and point cloud
    points = rotate(points, 28.4219545298, 169.132810398)
    visualise_data(rendering, points)

if __name__ == "__main__":
    main()