import os
import random
from os.path import join as pjoin

import torch

torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse, save_arg_parse
from data.t2m_dataset import MotionDataset
from utils.paramUtil import t2m_kinematic_chain

from utils.plot_script import plot_3d_motion

os.environ["OMP_NUM_THREADS"] = "1"


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i, joint in enumerate(data):
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, t2m_kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    opt = arg_parse(True)
    opt.name = "test2"
    opt.max_epoch = 2
    opt.milestones = [75_000, 125_000]
    opt.warm_up_iter = 1000
    opt.log_every = 75
    opt.save_latest = 15000
    save_arg_parse(opt)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.data_root = './dataset/mootion2/'
    opt.joints_num = 24
    dim_pose = 3 + 24 * 6
    radius = 4
    fps = 30

    motions = sorted(os.listdir(opt.data_root))
    random.seed(12321)
    random.shuffle(motions)
    random.seed()
    train_split = motions[:-5000]
    val_split = motions[-5000:]

    net = RVQVAE(opt,
                 dim_pose,
                 opt.nb_code,
                 opt.code_dim,
                 opt.code_dim,
                 opt.down_t,
                 opt.stride_t,
                 opt.width,
                 opt.depth,
                 opt.dilation_growth_rate,
                 opt.vq_act,
                 opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)

    print('Total parameters of all models: {}M'.format(pc_vq / 1000_000))

    ckpt = torch.load("./checkpoints/t2m/test/model/latest.tar", map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    net.load_state_dict(ckpt[model_key])

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    train_dataset = MotionDataset(opt, train_split, "./dataset/mootion2_train.pk")
    val_dataset = MotionDataset(opt, val_split, "./dataset/mootion2_val.pk")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)
    trainer.train(train_loader, val_loader, plot_t2m)

## train_vq.py --dataset_name kit --batch_size 512 --name VQVAE_dp2 --gpu_id 3
## train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp2_b256 --gpu_id 2
## train_vq.py --dataset_name kit --batch_size 1024 --name VQVAE_dp2_b1024 --gpu_id 1
## python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp1_b256 --gpu_id 2
