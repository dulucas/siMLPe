import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.angle_to_joint import ang2joint

import torch
import torch.utils.data as data

class AMASSDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=True):
        super(AMASSDataset, self).__init__()
        self._split_name = split_name
        self._data_aug = data_aug
        self._root_dir = config.root_dir

        self._amass_anno_dir = config.amass_anno_dir

        self._amass_file_names = self._get_amass_names()
        self.amass_motion_input_length =  config.motion.amass_input_length
        self.amass_motion_target_length =  config.motion.amass_target_length

        self.motion_dim = config.motion.dim
        self._load_skeleton()
        self._all_amass_motion_poses = self._load_all()
        self._file_length = len(self._all_amass_motion_poses)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_amass_motion_poses)

    def _get_amass_names(self):

        # create list
        seq_names = []

        if self._split_name == 'train' :
            seq_names += np.loadtxt(
                os.path.join(self._amass_anno_dir.replace('amass', ''), "amass_train.txt"), dtype=str
                ).tolist()
        else :
            seq_names += np.loadtxt(
                os.path.join(self._amass_anno_dir.replace('amass', ''), "amass_test.txt"), dtype=str
                ).tolist()

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return file_list

    def _preprocess(self, amass_motion_feats):
        if amass_motion_feats is None:
            return None
        amass_seq_len = amass_motion_feats.shape[0]

        if self.amass_motion_input_length + self.amass_motion_target_length < amass_seq_len:
            start = np.random.randint(amass_seq_len - self.amass_motion_input_length  - self.amass_motion_target_length + 1)
            end = start + self.amass_motion_input_length
        else:
            return None
        amass_motion_input = torch.zeros((self.amass_motion_input_length, amass_motion_feats.shape[1]))
        amass_motion_input[:end-start] = amass_motion_feats[start:end]

        amass_motion_target = torch.zeros((self.amass_motion_target_length, amass_motion_feats.shape[1]))
        amass_motion_target[:self.amass_motion_target_length] = amass_motion_feats[end:end+self.amass_motion_target_length]

        amass_motion = torch.cat([amass_motion_input, amass_motion_target], axis=0)

        return amass_motion

    def _load_skeleton(self):

        skeleton_info = np.load(
                os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _load_all(self):
        all_amass_motion_poses = []
        for amass_motion_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_motion_name)
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
            amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), self.parent).reshape(-1, 52, 3)[:, 4:22].reshape(T, -1)

            all_amass_motion_poses.append(amass_motion_poses)
        return all_amass_motion_poses

    def __getitem__(self, index):
        amass_motion_poses = self._all_amass_motion_poses[index]
        amass_motion = self._preprocess(amass_motion_poses)
        if amass_motion is None:
            while amass_motion is None:
                index = np.random.randint(self._file_length)
                amass_motion_poses = self._all_amass_motion_poses[index]
                amass_motion = self._preprocess(amass_motion_poses)

        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(amass_motion.size(0)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                amass_motion = amass_motion[idx]

        amass_motion_input = amass_motion[:self.amass_motion_input_length].float()
        amass_motion_target = amass_motion[-self.amass_motion_target_length:].float()
        return amass_motion_input, amass_motion_target

