import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

import torch
import torch.utils.data as data

class H36MDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=False):
        super(H36MDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug

        self._h36m_anno_dir = config.h36m_anno_dir
        self.used_joint_indexes = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
        self._h36m_files = self._get_h36m_files()

        self.h36m_motion_input_length =  config.motion.h36m_input_length
        self.h36m_motion_target_length =  config.motion.h36m_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_files)

    def _get_h36m_files(self):

        # create list
        seq_names = []

        if self._split_name == 'train' :
            seq_names += np.loadtxt(
                os.path.join(self._h36m_anno_dir.replace('h36m', ''), "h36m_train.txt"), dtype=str
                ).tolist()
        else :
            seq_names += np.loadtxt(
                os.path.join(self._h36m_anno_dir.replace('h36m', ''), "h36m_test.txt"), dtype=str
                ).tolist()

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._h36m_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                file_list.append(subject)

        h36m_files = []
        for path in file_list:
            info = open(path, 'r').readlines()
            pose_info = []
            for line in info:
                line = line.strip().split(',')
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            pose_info = np.array(pose_info)
            T = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(T, 32, 3, 3)
            xyz_info = rotmat2xyz_torch(pose_info)
            xyz_info = xyz_info[:, self.used_joint_indexes, :]
            h36m_files.append(xyz_info)
        return h36m_files

    def _collect_all(self):
        # Keep align with HisRep dataloader
        self.h36m_seqs = []
        self.data_idx = []
        idx = 0
        for h36m_motion_poses in self._h36m_files:
            N = len(h36m_motion_poses)
            if N < self.h36m_motion_target_length + self.h36m_motion_input_length:
                continue

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_motion_poses = h36m_motion_poses[sampled_index]

            T = h36m_motion_poses.shape[0]
            h36m_motion_poses = h36m_motion_poses.reshape(T, -1)

            self.h36m_seqs.append(h36m_motion_poses)
            valid_frames = np.arange(0, T - self.h36m_motion_input_length - self.h36m_motion_target_length + 1, self.shift_step)

            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        motion = self.h36m_seqs[idx][frame_indexes]
        if self.data_aug:
            if torch.rand(1)[0] > .5:
                idx = [i for i in range(motion.size(0)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        h36m_motion_input = motion[:self.h36m_motion_input_length] / 1000 # meter
        h36m_motion_target = motion[self.h36m_motion_input_length:] / 1000 # meter

        h36m_motion_input = h36m_motion_input.float()
        h36m_motion_target = h36m_motion_target.float()
        return h36m_motion_input, h36m_motion_target

