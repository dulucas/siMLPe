import argparse
import os, sys
import json
import numpy as np

from config import config
from model import siMLPe as Model
from datasets.amass import AMASSDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(config.seed)
writer = SummaryWriter()

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.amass_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 100000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,n,c = amass_motion_input.shape
        amass_motion_input_ = amass_motion_input.clone()
        amass_motion_input_ = torch.matmul(dct_m, amass_motion_input_.cuda())
    else:
        amass_motion_input_ = amass_motion_input.clone()

    motion_pred = model(amass_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m, motion_pred)

    if config.deriv_output:
        offset = amass_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.amass_target_length] + offset

    b,n,c = amass_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,18,3).reshape(-1,3)
    amass_motion_target = amass_motion_target.cuda().reshape(b,n,18,3).reshape(-1,3)
    loss = torch.mean(torch.norm(motion_pred - amass_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,18,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = amass_motion_target.reshape(b,n,18,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model = Model(config)
model.train()
model.cuda()

config.motion.amass_target_length = config.motion.amass_target_length_train
dataset = AMASSDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (amass_motion_input, amass_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
