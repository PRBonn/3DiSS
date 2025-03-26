import numpy as np
import MinkowskiEngine as ME
import torch
import os

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def feats_to_coord(p_feats, resolution):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def points_to_tensor(grid_coords, grid_feats, resolution, train_step):
    # in the first iteration pytorch uses more memory to figure out how to proper deal with the data for ME (sparse tensors)
    # the memory allocated is way bigger than the actual needed so we limit the amount of points for the first iteration
    if train_step == 0:
        grid_coords = [ c[:100] for c in grid_coords ]
        grid_feats = [ f[:100] for f in grid_feats ]

    # add batch index
    batched_coords = ME.utils.batched_coordinates(list(grid_coords), dtype=torch.float32, device=torch.device('cuda'))
    batched_feats = ME.utils.batched_coordinates(list(grid_feats), dtype=torch.float32, device=torch.device('cuda'))[:,1:]

    x_occupancy = ME.SparseTensor(
        features=batched_feats,
        coordinates=batched_coords,
        device=torch.device('cuda'),
    )

    torch.cuda.empty_cache()

    return x_occupancy

def pcd_to_fov(points, xyz_range, resolution):
    grid_fov = (points[:,0] > xyz_range[0][0]) & (points[:,0] < xyz_range[0][1]) &\
                (points[:,1] > xyz_range[1][0]) & (points[:,1] < xyz_range[1][1]) &\
                (points[:,2] > xyz_range[2][0]) & (points[:,2] < xyz_range[2][1])

    x_fov = torch.tensor(points[grid_fov])

    x_fov[:,:3] = (x_fov[:,:3] / resolution).trunc()
    _, mapping = ME.utils.sparse_quantize(coordinates=x_fov[:,:3], return_index=True)
    x_fov = x_fov[mapping]
    x_fov[:,:3] -= x_fov[:,:3].min(0).values

    return x_fov[:,:3], torch.tensor(points[grid_fov][mapping])

def point_set_to_sparse(points, scan, resolution, filename, xyz_range):
    points_coords, points_feats = pcd_to_fov(points, xyz_range, resolution)
    if scan is not None:
        scan_coords, scan_feats = pcd_to_fov(scan, xyz_range, resolution)
    else:
        scan_coords, scan_feats = None, None

    return [points_coords,
            points_feats,
            scan_coords,
            scan_feats,
            filename,
            ]

class SparseCollation:
    def __init__(self):
        return

    def __call__(self, data):
        batch = list(zip(*data))

        return {'coords': batch[0], 'feats': batch[1], 'cond': [batch[2], batch[3]], 'filename': batch[4]}
