import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

import torch
import torch.utils.data as data

class H36MEval(data.Dataset):
    def __init__(self, config, split_name, paired=True):
        super(H36MEval, self).__init__()
        self._split_name = split_name
        self._h36m_anno_dir = config.h36m_anno_dir
        self._actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

        self.h36m_motion_input_length =  config.motion.h36m_input_length
        self.h36m_motion_target_length =  config.motion.h36m_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._h36m_files = self._get_h36m_files()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_files)

    def _get_h36m_files(self):

        # create list
        seq_names = []

        seq_names += open(
            os.path.join(self._h36m_anno_dir.replace('h36m', ''), "h36m_test.txt"), 'r'
            ).readlines()

        self.h36m_seqs = []
        self.data_idx = []
        idx = 0
        for subject in seq_names:
            subject = subject.strip()
            for act in self._actions:
                filename0 = '{0}/{1}/{2}_{3}.txt'.format(self._h36m_anno_dir, subject, act, 1)
                filename1 = '{0}/{1}/{2}_{3}.txt'.format(self._h36m_anno_dir, subject, act, 2)
                poses0 = self._preprocess(filename0)
                poses1 = self._preprocess(filename1)

                self.h36m_seqs.append(poses0)
                self.h36m_seqs.append(poses1)

                num_frames0 = poses0.shape[0]
                num_frames1 = poses1.shape[0]

                fs_sel1, fs_sel2 = find_indices_256(num_frames0, num_frames1,
                                   self.h36m_motion_input_length + self.h36m_motion_target_length,
                                   input_n=self.h36m_motion_input_length)
                #fs_sel1, fs_sel2 = find_indices_srnn(num_frames0, num_frames1,
                #                   self.h36m_motion_input_length + self.h36m_motion_target_length,
                #                   input_n=self.h36m_motion_input_length)
                valid_frames0 = fs_sel1[:, 0]
                tmp_data_idx_1 = [idx] * len(valid_frames0)
                tmp_data_idx_2 = list(valid_frames0)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                valid_frames1 = fs_sel2[:, 0]
                tmp_data_idx_1 = [idx + 1] * len(valid_frames1)
                tmp_data_idx_2 = list(valid_frames1)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                idx += 2

    def _preprocess(self, filename):
        info = open(filename, 'r').readlines()
        pose_info = []
        for line in info:
            line = line.strip().split(',')
            if len(line) > 0:
                pose_info.append(np.array([float(x) for x in line]))
        pose_info = np.array(pose_info)
        pose_info = pose_info.reshape(-1, 33, 3)
        pose_info[:, :2] = 0
        N = pose_info.shape[0]
        pose_info = pose_info.reshape(-1, 3)
        pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(N, 33, 3, 3)[:, 1:]
        pose_info = rotmat2xyz_torch(pose_info)

        sample_rate = 2
        sampled_index = np.arange(0, N, sample_rate)
        h36m_motion_poses = pose_info[sampled_index]

        T = h36m_motion_poses.shape[0]
        h36m_motion_poses = h36m_motion_poses.reshape(T, 32, 3)
        return h36m_motion_poses

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        motion = self.h36m_seqs[idx][frame_indexes]

        h36m_motion_input = motion[:self.h36m_motion_input_length] / 1000.
        h36m_motion_target = motion[self.h36m_motion_input_length:] / 1000.

        h36m_motion_input = h36m_motion_input.float()
        h36m_motion_target = h36m_motion_target.float()
        return h36m_motion_input, h36m_motion_target

