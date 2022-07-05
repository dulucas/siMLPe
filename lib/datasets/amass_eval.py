import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.angle_to_joint import ang2joint

import torch
import torch.utils.data as data

class AMASSEval(data.Dataset):
    def __init__(self, config, split_name, paired=True):
        super(AMASSEval, self).__init__()
        self._split_name = split_name
        self._amass_anno_dir = config.amass_anno_dir
        self._root_dir = config.root_dir

        self._amass_file_names = self._get_amass_names()

        self.amass_motion_input_length =  config.motion.amass_input_length
        self.amass_motion_target_length =  config.motion.amass_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step

        self._load_skeleton()
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._amass_file_names)

    def _get_amass_names(self):

        # create list
        seq_names = []
        assert self._split_name == 'test'

        seq_names += open(
            os.path.join(self._amass_anno_dir, "amass_test.txt"), 'r'
            ).readlines()

        file_list = []
        for dataset in seq_names:
            dataset = dataset.strip()
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        #file_list = file_list[:10]
        return file_list

    def _load_skeleton(self):

        skeleton_info = np.load(
                os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _collect_all(self):
        # Keep align with HisRep dataloader
        self.amass_seqs = []
        self.data_idx = []
        idx = 0
        for amass_seq_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_seq_name)
            amass_motion_poses = amass_info['poses'] # 156 joints(all joints of SMPL)
            N = len(amass_motion_poses)
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            frame_rate = amass_info['mocap_framerate']
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            amass_motion_poses = amass_motion_poses[sampled_index]

            T = amass_motion_poses.shape[0]
            amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
            amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
            amass_motion_poses[:, 0] = 0

            p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
            amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), self.parent)
            amass_motion_poses = amass_motion_poses.reshape(-1, 52, 3)[:, 4:22].reshape(T, 54)

            self.amass_seqs.append(amass_motion_poses)
            valid_frames = np.arange(0, T - self.amass_motion_input_length - self.amass_motion_target_length + 1, self.shift_step)

            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.amass_motion_input_length + self.amass_motion_target_length)
        motion = self.amass_seqs[idx][frame_indexes]
        amass_motion_input = motion[:self.amass_motion_input_length]
        amass_motion_target = motion[self.amass_motion_input_length:]
        return amass_motion_input, amass_motion_target

