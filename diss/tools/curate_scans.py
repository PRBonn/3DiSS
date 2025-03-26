from tqdm import tqdm
from diss.utils.data_map import color_map
import click
import numpy as np
import open3d as o3d
import os
import shutil
from pynput import keyboard
from natsort import natsorted
from tqdm import tqdm

save_request = False
window_closed = False

def on_press(key):
    global save_requested, window_closed
    try:
        if key.char.lower() == 'x':
            save_requested = True
        elif key.char.lower() == 'q':
            window_closed = True
    except AttributeError:
        pass

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
    points = pcd_[:,:3]
    labels = pcd_[:,-1].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color_array = np.array(list(color_map.values()))
    colors = color_array[labels,::-1]
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.)

    return pcd

@click.command()
### Add your options here
@click.option('--input_path',
              '-i',
              type=str,
              default=None)
@click.option('--output_path',
              '-o',
              type=str,
              default=None)
def main(input_path, output_path):
    global save_requested, window_closed
    # Ensure target directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get a list of point cloud files
    point_cloud_files = [f for f in os.listdir(input_path) if f.endswith('.npy') or f.endswith('.npz') or f.endswith('.ply')]
    point_cloud_files = natsorted(point_cloud_files)

    if not point_cloud_files:
        print("No point cloud files found in the specified directory.")
        exit(1)
    
    # Global state to handle key presses and point cloud index
    current_index = 0
    save_requested = False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    for pcd_file in tqdm(point_cloud_files[6000:]):
        if os.path.isfile(f'{output_path}/{pcd_file}'):
            continue
        file_path = os.path.join(input_path, pcd_file)
        print(f"Displaying: {file_path}")

        # Load and display the point cloud
        pcd = npy_to_pcd(file_path)
        pcd.estimate_normals()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        window_closed = False
        print("Press 'X' to save this point cloud or close the window to move to the next.")

        while not window_closed:
            vis.poll_events()
            vis.update_renderer()

            if save_requested:
                save_requested = False
                window_closed = True
                target_path = os.path.join(output_path, os.path.basename(file_path))
                shutil.copy(file_path, target_path)
                print(f"Saved: {target_path}")

        vis.clear_geometries()
        current_index += 1

    vis.destroy_window()
    listener.stop()
    print("Finished displaying all point clouds.")

if __name__ == "__main__":
    main()
