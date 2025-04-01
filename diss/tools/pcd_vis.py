import open3d as o3d
import click
import os
import multiprocessing
from diss.tools.render_lidar import simulate_lidar
from diss.utils.data_map import color_map
import numpy as np
from tqdm import tqdm
from random import shuffle

def visualize_pcd(pcd, window_name):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    def close_callback(vis):
        vis.close()

    vis.register_key_callback(ord("Q"), close_callback)

    while vis.poll_events():
        vis.update_renderer()
    vis.destroy_window()

def load_pcd(pcd_file, xyz_range):
    if pcd_file.endswith('.npz'):
        points = np.load(pcd_file)['arr_0']
    elif pcd_file.endswith('.npy'):
        points = np.load(pcd_file)

    grid_fov = (points[:,0] > xyz_range[0][0]) & (points[:,0] < xyz_range[0][1]) &\
                (points[:,1] > xyz_range[1][0]) & (points[:,1] < xyz_range[1][1]) &\
                (points[:,2] > xyz_range[2][0]) & (points[:,2] < xyz_range[2][1])

    return points[grid_fov]

def npy_to_pcd(pcd_file):
    if pcd_file.endswith('.ply'):
        return o3d.io.read_point_cloud(pcd_file) 

    xyz_range = [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]]
    pcd_ = load_pcd(pcd_file, xyz_range)
    points = np.round(pcd_[:,:3]/0.1) * 0.1
    labels = pcd_[:,-1].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if 'diff_x0' in pcd_file:
        color_array = np.array(list(color_map.values()))
        colors = color_array[labels,::-1]
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.)

    return pcd

@click.command()
### Add your options here
@click.option('--path',
              '-p',
              type=str,
              default=None)
def main(path):
    pcd_list = os.listdir(os.path.join(path,'diff_x0'))
    shuffle(pcd_list)
    pcd_range = [25.6, 25.6, 2.2]
    pcd_res = 1.
    for pcd_file in tqdm(pcd_list):
        pcd = npy_to_pcd(os.path.join(path, 'diff_x0', pcd_file))
        pcd.estimate_normals()

        if 'single_scan' in path:
            pcd_lidar = npy_to_pcd(os.path.join(path, 'cond', pcd_file))
            pcd_lidar.estimate_normals()

        #o3d.visualization.draw([x0_pcd, x0_snr_avg_pcd])
        process1 = multiprocessing.Process(target=visualize_pcd, args=(pcd, "PCD"))
        if 'single_scan' in path:
            process2 = multiprocessing.Process(target=visualize_pcd, args=(pcd_lidar, "PCD LiDAR"))
        
        # Start the processes
        process1.start()
        if 'single_scan' in path:
            process2.start()
        
        # Wait for both processes to complete
        process1.join()
        if 'single_scan' in path:
            process2.join()


if __name__ == "__main__":
    main()
