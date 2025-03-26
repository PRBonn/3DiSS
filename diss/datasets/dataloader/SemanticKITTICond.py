import torch
from torch.utils.data import Dataset
from diss.utils.pcd_transforms import *
from diss.utils.data_map import learning_map
from diss.utils.collations import point_set_to_sparse, load_poses
from natsort import natsorted
import os
import numpy as np
import yaml

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class CondKITTISet(Dataset):
    def __init__(self, data_dir, split, resolution, xyz_range):
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self.split = split
        self.xyz_range = xyz_range

        self.cache_maps = {}

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.nr_data = len(self.points_datapath)
        os.makedirs(os.path.join(self.data_dir, f'assets/diss/gt/'), exist_ok=True)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []

        for seq in self.split:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            poses = load_poses(os.path.join(point_seq_path, 'calib.txt'), os.path.join(point_seq_path[:-3], f'pin_slam_poses/{seq}.txt'))
            p_full = np.load(f'{point_seq_path}/sem_map.npy') if self.split != 'test' else np.array([[1,0,0],[0,1,0],[0,0,1]])
            self.cache_maps[seq] = p_full
 
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.seq_poses.append(poses[file_num])

        # we do the validation just over few samples for faster training
        if '08' in self.split:
            self.points_datapath = self.points_datapath[:12]
            self.seq_poses = self.seq_poses[:12]

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        if os.path.exists(os.path.join(self.data_dir, f'assets/diss/gt/{index}.npy')):
            p_cache = np.load(os.path.join(self.data_dir, f'assets/diss/gt/{index}.npy'))
            p_set = p_cache[:,:3]
            l_set = p_cache[:,-1,None]
        else:
            seq_num = self.points_datapath[index].split('/')[-3]
            fname = self.points_datapath[index].split('/')[-1].split('.')[0]
            pose = self.seq_poses[index]
    
            # load map for the sequence corresponding to the sampled scan and transform it to be centered at the scan pose
            p_map = self.cache_maps[seq_num]
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
    
            # remove moving points (avoid the "ghost" artifacts when aggregating moving objects)
            static_idx = l_set < 252
            p_set = p_set[static_idx]
            l_set = l_set[static_idx]
    
            labeled_idx = l_set != 0
            p_set = p_set[labeled_idx]
            l_set = l_set[labeled_idx, None]

            # cache this cropped point cloud to avoid redoing all the transformations
            np.save(os.path.join(self.data_dir, f'assets/diss/gt/{index}.npy'), np.concatenate((p_set, l_set), axis=-1))
    
        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1,4))[:,:3]
        p_set = np.concatenate((p_set, p_part), axis=0)

        p_set = self.transforms(p_set)
        p_part = p_set[-len(p_part):]
        p_set = p_set[:-len(p_part)]

        return point_set_to_sparse(
            np.concatenate((p_set, l_set), axis=-1),
            p_part,
            self.resolution,
            self.points_datapath[index],
            self.xyz_range,
        )

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################
