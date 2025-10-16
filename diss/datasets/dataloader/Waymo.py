import torch
from torch.utils.data import Dataset
from diss.utils.pcd_transforms import *
from diss.utils.data_map import learning_map
from diss.utils.collations import point_set_to_sparse, load_poses
from natsort import natsorted
import json
import os
import numpy as np
import yaml

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class WaymoSet(Dataset):
    def __init__(self, data_dir, split, resolution, xyz_range):
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        if split == 'train':
            with open(f'{self.data_dir}/waymo_split/official_train_w_dynamic_w_ego_motion_gt_30m_good_voxel.json', 'r') as f:
                self.split = json.load(f)
        if split == 'validation' or split == 'test':
            with open(f'{self.data_dir}/waymo_split/official_val_w_dynamic_w_ego_motion_gt_30m_good_voxel.json', 'r') as f:
                self.split = json.load(f)


        self.xyz_range = xyz_range

        self.cache_maps = {}

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.nr_data = len(self.points_datapath)
        os.makedirs(os.path.join(self.data_dir, f'assets/diss/gt_waymo/'), exist_ok=True)

        #print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def read_pcd_map(self, seq_map):
        pcd_raw = torch.load(f'{self.data_dir}/waymo_scube/sequences/{seq_map}.pcd.vs01.pth')
        pcd_pts = pcd_raw['points'].numpy()
        pcd_sem = pcd_raw['semantics'].numpy()
        pcd_to_world = pcd_raw['pc_to_world'].numpy()
        p_full = np.concatenate((pcd_pts, pcd_sem[:,None]), -1)

        return p_full, pcd_to_world

    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []

        for seq in self.split:
            with open(f'{self.data_dir}/waymo_data/segment-{seq}_with_camera_labels.tfrecord/calib.txt', 'r') as f:
                calib_raw = f.readlines()[0]
            extrinsics = np.array(calib_raw.split(' ')).astype(float).reshape((4,4))

            with open(f'{self.data_dir}/waymo_data/segment-{seq}_with_camera_labels.tfrecord/poses.txt', 'r') as f:
                poses_raw = f.readlines()
 
            _, pcd_to_world = self.read_pcd_map(seq)

            for pose_raw in poses_raw:
                pose = np.array(pose_raw.split(' ')).astype(float).reshape((4,4))
                pose_lidar = (np.linalg.inv(pcd_to_world) @ pose) @ extrinsics
                self.points_datapath.append(seq)
                self.seq_poses.append(pose_lidar)

        # we do the validation just over few samples for faster training
        if '08' in self.split:
            self.points_datapath = self.points_datapath[:12]
            self.seq_poses = self.seq_poses[:12]

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        theta = torch.FloatTensor(1,1).uniform_(0, 2*np.pi).item()
        scale_factor = torch.FloatTensor(1,1).uniform_(0.95, 1.05).item()
        rot_mat = np.array([[np.cos(theta),
                            -np.sin(theta), 0],
                            [np.sin(theta),
                            np.cos(theta), 0], [0, 0, 1]])
        points[:,:,:3] = np.dot(points[:,:,:3], rot_mat) * scale_factor


        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        if os.path.exists(os.path.join(self.data_dir, f'assets/diss/gt_waymo/{index}.npy')):
            p_cache = np.load(os.path.join(self.data_dir, f'assets/diss/gt_waymo/{index}.npy'))
            p_set = p_cache[:,:3]
            l_set = p_cache[:,-1,None]
        else:
            seq_num = self.points_datapath[index]
            pose = self.seq_poses[index]
    
            # load map for the sequence corresponding to the sampled scan and transform it to be centered at the scan pose
            #p_map = self.cache_maps[seq_num]
            p_map, _ = self.read_pcd_map(seq_num)
            trans = pose[:-1,-1]

            # crop an area of 51.2m circunference around the pose from the map
            dist_full = np.sum((p_map[:,:3] - trans)**2, -1)**.5
            p_full = p_map[dist_full < 51.2]
    
            l_set = p_full[:,-1]
            
            # apply the inverse pose to be centered on zero again
            p_full_ = np.concatenate((p_full[:,:3], np.ones((len(p_full),1))), axis=-1)
            p_full[:,:3] = (p_full_ @ np.linalg.inv(pose).T)[:,:3]
    
            # remove some noise from the bottom of the scans
            p_set = p_full[:,:3]
            p_set = p_set[p_full[:,2] > -4.]
            l_set = l_set[p_full[:,2] > -4.]
    
            labeled_idx = l_set != 0
            p_set = p_set[labeled_idx]
            l_set = l_set[labeled_idx, None]

            # cache this cropped point cloud to avoid redoing all the transformations
            np.save(os.path.join(self.data_dir, f'assets/diss/gt_waymo/{index}.npy'), np.concatenate((p_set, l_set), axis=-1))
    
        p_set = self.transforms(p_set)

        return point_set_to_sparse(
            np.concatenate((p_set, l_set), axis=-1),
            None,
            self.resolution,
            self.points_datapath[index],
            self.xyz_range,
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
