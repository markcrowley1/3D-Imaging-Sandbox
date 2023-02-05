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

def read_metadata(dir: str) -> list[tuple]:
    """Return each img file with elevation and azimuth"""
    metadata = []
    data = open(f"{dir}/rendering_metadata.txt", "r")
    filenames = open(f"{dir}/renderings.txt", "r")
    data = data.readlines()
    filenames = filenames.readlines()

    for i in range(len(filenames)):
        filename = filenames[i].strip()
        line = data[i].strip()
        azimuth = float(line.split()[0])
        elevation = float(line.split()[1])
        metadata.append((filename, elevation, azimuth))

    return metadata

def rotate(points: np.ndarray, rotation: list) -> np.ndarray:
    # Rotate to rendering position
    r = Rotation.from_euler("xyz", rotation, degrees= True)
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
    rendering = "D:/ShapeNetRendering/ShapeNetRendering/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/rendering"
    obj_file = "D:/shapenet_base/shapenet_core/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/model_normalized.obj"

    # Sample ground truth point cloud and rotate to base position
    mesh = trimesh.load(obj_file, force="mesh")
    sample = trimesh.sample.sample_surface(mesh, POINTS)
    sampled_points = sample[0]
    points = np.array(sampled_points)
    base_points = rotate(points, [-90, 90, 0])

    # Rotate to align with rendered image and compare
    meta_data = read_metadata(rendering)
    for data in meta_data:
        rotation = [0, -data[1], -data[2]]
        points = rotate(base_points, rotation)
        visualise_data(f"{rendering}/{data[0]}", points)

if __name__ == "__main__":
    main()