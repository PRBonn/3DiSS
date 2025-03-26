import numpy as np
import open3d as o3d
import os
import click
import tqdm
from joblib import Parallel, delayed
from diss.utils.data_map import color_map

def simulate_lidar_npy(points, labels, num_beams=64, v_fov=(-24.9, 2.0), h_res=0.2):
    """
    Simulates a 64-beam LiDAR point cloud from a dense point cloud.

    Parameters:
    - points: point cloud 3d coordinates.
    - labels: labels for each point.
    - v_fov: tuple of (min, max) vertical field of view in degrees.
    - h_res: horizontal resolution in degrees.

    Returns:
    - lidar_pcd: Open3D PointCloud object representing the simulated LiDAR point cloud.
    """

    # LiDAR specs
    v_fov_min, v_fov_max = v_fov
    v_res = (v_fov_max - v_fov_min) / (num_beams - 1)
    beam_thickness = v_res / 2

    # Calculate horizontal and vertical angles
    horizontal_angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    distances = np.linalg.norm(points[:, :2], axis=1)
    vertical_angles = np.arctan2(points[:, 2], distances) * 180 / np.pi

    # Initialize list to hold the LiDAR points
    lidar_points = []
    lidar_labels = []

    # Loop over each vertical beam
    for beam in range(num_beams):
        # Define the vertical angle range for the current beam
        v_angle_min = v_fov_min + beam * v_res
        v_angle_max = v_angle_min + v_res
        beam_angle = (v_angle_min + v_angle_max) / 2
        # the further we go the thiner is the beam
        beam_scale = 3.5 * (1.0 - ((beam+1) / num_beams)**2)
        v_angle_min = beam_angle - beam_thickness * beam_scale 
        v_angle_max = beam_angle + beam_thickness * beam_scale

        # Filter points within the vertical angle range
        v_mask = (vertical_angles >= v_angle_min) & (vertical_angles < v_angle_max)

        # Further filter points to match the horizontal resolution
        filtered_points = points[v_mask]
        filtered_labels = labels[v_mask]
        filtered_h_angles = horizontal_angles[v_mask]
        filtered_distances = distances[v_mask]
        unique_h_angles = np.arange(-180, 180+h_res, h_res)

        for h_angle in unique_h_angles:
            h_mask = (filtered_h_angles >= h_angle) & (filtered_h_angles < h_angle + h_res)
            h_points = filtered_points[h_mask]
            h_labels = filtered_labels[h_mask]
            h_distances = filtered_distances[h_mask]

            if len(h_points) > 0:
                # Choose the closest point (simulate a single LiDAR return)
                closest = np.argmin(h_distances)
                lidar_points.append(h_points[closest])
                lidar_labels.append(h_labels[closest])

    # Convert the LiDAR points back to Open3D point cloud
    lidar_points = np.array(lidar_points)
    lidar_labels = np.array(lidar_labels)
    dist = np.sqrt(np.sum(lidar_points**2, -1))

    #lidar_points = lidar_points[dist > 3.5]
    #lidar_labels = lidar_labels[dist > 3.5]

    lidar_pcd = np.concatenate((lidar_points, lidar_labels[:,None]),-1)

    return lidar_pcd

def simulate_lidar(dense_pcd, num_beams=64, v_fov=(-24.9, 2.0), h_res=0.5):
    """
    Simulates a 64-beam LiDAR point cloud from a dense point cloud.
    
    Parameters:
    - dense_pcd: Open3D PointCloud object representing the dense point cloud.
    - v_fov: tuple of (min, max) vertical field of view in degrees.
    - h_res: horizontal resolution in degrees.
    
    Returns:
    - lidar_pcd: Open3D PointCloud object representing the simulated LiDAR point cloud.
    """
    
    # LiDAR specs
    v_fov_min, v_fov_max = v_fov
    v_res = (v_fov_max - v_fov_min) / (num_beams - 1)
    beam_thickness = v_res / 2
    
    # Convert dense point cloud to numpy array
    points = np.asarray(dense_pcd.points)
    colors = np.asarray(dense_pcd.colors)
    
    # Calculate horizontal and vertical angles
    horizontal_angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    distances = np.linalg.norm(points[:, :2], axis=1)
    vertical_angles = np.arctan2(points[:, 2], distances) * 180 / np.pi
    
    # Initialize list to hold the LiDAR points
    lidar_points = []
    lidar_colors = []
    
    # Loop over each vertical beam
    for beam in range(num_beams):
        # Define the vertical angle range for the current beam
        v_angle_min = v_fov_min + beam * v_res
        v_angle_max = v_angle_min + v_res
        beam_angle = (v_angle_min + v_angle_max) / 2
        # the further we go the thiner is the beam
        beam_scale = 3.5 * (1.0 - ((beam+1) / num_beams)**2)
        v_angle_min = beam_angle - beam_thickness * beam_scale
        v_angle_max = beam_angle + beam_thickness * beam_scale
        
        # Filter points within the vertical angle range
        v_mask = (vertical_angles >= v_angle_min) & (vertical_angles < v_angle_max)
        
        # Further filter points to match the horizontal resolution
        filtered_points = points[v_mask]
        filtered_colors = colors[v_mask]
        filtered_h_angles = horizontal_angles[v_mask]
        unique_h_angles = np.arange(-180, 180+h_res, h_res)
        
        for h_angle in unique_h_angles:
            h_mask = (filtered_h_angles >= h_angle) & (filtered_h_angles < h_angle + h_res)
            h_points = filtered_points[h_mask]
            h_colors = filtered_colors[h_mask]
            
            if len(h_points) > 0:
                # Choose the closest point (simulate a single LiDAR return)
                closest = np.argmin(np.linalg.norm(h_points, axis=1))
                lidar_points.append(h_points[closest])
                lidar_colors.append(h_colors[closest])
    
    # Convert the LiDAR points back to Open3D point cloud
    lidar_pcd = o3d.geometry.PointCloud()
    #dist = np.sqrt(np.sum(np.array(lidar_points)**2, -1))
    lidar_pcd.points = o3d.utility.Vector3dVector(np.array(lidar_points))
    lidar_pcd.colors = o3d.utility.Vector3dVector(np.array(lidar_colors))
    
    return lidar_pcd

def npy_to_pcd(pcd_file):
    pcd_ = np.load(pcd_file)
    points = pcd_[:,:3]
    points[:,:2] -= points[:,:2].mean(0)
    labels = pcd_[:,-1].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color_array = np.array(list(color_map.values()))
    colors = color_array[labels,::-1]
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.)

    return pcd

def dense_to_lidar(path, path_, pcd_file):
    dense_pcd = np.load(os.path.join(path_, pcd_file))['arr_0']
    points = dense_pcd[:,:3]
    points[:,:2] -= points[:,:2].mean(0)
    labels = dense_pcd[:,-1]

    # Simulate the 64-beam LiDAR point cloud
    lidar_pcd = simulate_lidar_npy(points, labels, v_fov=(-19.9,7.0), h_res=0.42)
    np.savez_compressed(os.path.join(path, 'lidar_proj_x0', pcd_file), lidar_pcd)

@click.command()
@click.option('--path',
              '-p',
              type=str,
              default=None)
@click.option('--x0', '-x', is_flag=True, help='test mode')
@click.option('--real', '-r', is_flag=True, help='test mode')
def main(path, x0, real):
    os.makedirs(os.path.join(path, 'lidar_proj_x0'), exist_ok=True)
    if x0:
        path_ = os.path.join(path, 'x0')
    elif real:
        path_ = os.path.join(path, 'real_data')
    print(path_)

    # Load the dense point cloud
    result = Parallel(n_jobs=20)(delayed(dense_to_lidar)(path, path_, pcd_file) for pcd_file in tqdm.tqdm(os.listdir(path_)))

if __name__ == "__main__":
    main()
