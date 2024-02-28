import os
import torch
import random

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.get_opt import get_opt
from utils.fixseed import fixseed

from data.t2m_dataset import Text2MotionDataset


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                      dim_pose,
                      vq_opt.nb_code,
                      vq_opt.code_dim,
                      vq_opt.output_emb_width,
                      vq_opt.down_t,
                      vq_opt.stride_t,
                      vq_opt.width,
                      vq_opt.depth,
                      vq_opt.dilation_growth_rate,
                      vq_opt.vq_act,
                      vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'latest.tar'),
                      map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    opt.vq_name = "test"
    opt.name = "test_base"
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/base/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.data_root = './dataset/mootion/'
    opt.joints_num = 24
    dim_pose = 3 + 24 * 6
    radius = 4
    fps = 30

    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code

    t2m_transformer = MaskTransformer(code_dim=vq_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=opt)

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    motions = sorted(os.listdir(opt.data_root))
    random.seed(12321)
    random.shuffle(motions)
    random.seed()
    train_split = motions[:-5000]
    val_split = motions[-5000:]

    train_dataset = Text2MotionDataset(opt, train_split)
    val_dataset = Text2MotionDataset(opt, val_split)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    trainer = MaskTransformerTrainer(opt, t2m_transformer, vq_model)

    trainer.train(train_loader, val_loader)
