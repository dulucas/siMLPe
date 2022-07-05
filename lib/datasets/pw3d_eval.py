import os
import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.angle_to_joint import ang2joint

import torch
import torch.utils.data as data

class PW3DEval(data.Dataset):
    def __init__(self, config, split_name, paired=True):
        super(PW3DEval, self).__init__()
        self._split_name = split_name
        self._pw3d_anno_dir = config.pw3d_anno_dir
        self._root_dir = config.root_dir

        self._pw3d_file_names = self._get_pw3d_names()

        self.pw3d_motion_input_length =  config.motion.pw3d_input_length
        self.pw3d_motion_target_length =  config.motion.pw3d_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step

        self._load_skeleton()
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._pw3d_file_names)

    def _get_pw3d_names(self):

        # create list
        seq_names = []
        assert self._split_name == 'test'

        seq_names = glob.glob(self._pw3d_anno_dir + 'test/*')

        return seq_names

    def _load_skeleton(self):

        skeleton_info = np.load(
                os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()[:, :22]
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            self.parent[i] = parents[i]

    def _collect_all(self):
        self.pw3d_seqs = []
        self.data_idx = []
        idx = 0
        sample_rate = int(60 // 25)
        for pw3d_seq_name in tqdm(self._pw3d_file_names):
            pw3d_info = pkl.load(open(pw3d_seq_name, 'rb'), encoding='latin1')
            pw3d_motion_poses = pw3d_info['poses_60Hz']
            for i in range(len(pw3d_motion_poses)):
                N = len(pw3d_motion_poses[i])

                sampled_index = np.arange(0, N, sample_rate)
                motion_poses = pw3d_motion_poses[i][sampled_index]

                T = motion_poses.shape[0]
                motion_poses = motion_poses.reshape(T, -1, 3)
                motion_poses = motion_poses[:, :-2]
                motion_poses = R.from_rotvec(motion_poses.reshape(-1, 3)).as_rotvec()
                motion_poses = motion_poses.reshape(T, 22, 3)
                motion_poses[:, 0] = 0

                p3d0_tmp = self.p3d0.repeat([motion_poses.shape[0], 1, 1])
                motion_poses = ang2joint(p3d0_tmp, torch.tensor(motion_poses).float(), self.parent)
                motion_poses = motion_poses.reshape(-1, 22, 3)[:, 4:22].reshape(T, 54)

                self.pw3d_seqs.append(motion_poses)
                valid_frames = np.arange(0, T - self.pw3d_motion_input_length - self.pw3d_motion_target_length + 1, self.shift_step)

                self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.pw3d_motion_input_length + self.pw3d_motion_target_length)
        motion = self.pw3d_seqs[idx][frame_indexes]
        pw3d_motion_input = motion[:self.pw3d_motion_input_length]
        pw3d_motion_target = motion[self.pw3d_motion_input_length:]
        return pw3d_motion_input, pw3d_motion_target

