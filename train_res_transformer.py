import os
import json
import torch
import random
from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import ResidualTransformer
from models.mask_transformer.transformer_trainer import ResidualTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions
from tqdm import tqdm

from utils.get_opt import get_opt
from utils.fixseed import fixseed

from data.t2m_dataset import Text2MotionDataset, collate_fn


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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'E01I0020000.tar'),
                      map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.to(opt.device)
    return vq_model, vq_opt


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    opt.vq_name = "test2_ft"
    opt.name = "test_res35_ft"
    opt.share_weight = True
    opt.cond_drop_prob = 0.2

    ### new added ###
    # opt.max_epoch = 50
    # opt.milestones = [75_000]
    # opt.warm_up_iter = 1000
    # opt.log_every = 75
    # opt.save_latest = 15000

    opt.milestones = [15_000]
    opt.warm_up_iter = 200
    opt.log_every = 75
    opt.save_latest = 5000

    parser.save()

    fixseed(opt.seed)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log/lm/', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.data_root = './dataset/mootion2/'
    opt.joints_num = 24
    dim_pose = 3 + 24 * 6
    radius = 4
    fps = 30
    finetune = True

    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='text',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)

    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())

    print(res_transformer)
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    if not finetune:
        motions = os.listdir(opt.data_root)
        val_cnt = 5000
    else:
        motions = []
        for motion in tqdm(os.listdir(opt.data_root)):
            if motion.startswith('C'):
                motions.append(motion)
            elif motion.startswith('M'):
                with open(pjoin(opt.data_root, motion), 'r') as f:
                    data = json.load(f)
                    if "_" not in data['mid']:
                        motions.append(motion)
        print(len([m for m in motions if m.startswith('C')]), len([m for m in motions if m.startswith('M')]))
        val_cnt = 1000
    motions = sorted(motions)

    random.seed(12321)
    random.shuffle(motions)
    random.seed()
    train_split = motions[:-val_cnt]
    val_split = motions[-val_cnt:]

    train_dataset = Text2MotionDataset(opt, train_split)
    val_dataset = Text2MotionDataset(opt, val_split)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, collate_fn=collate_fn,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, collate_fn=collate_fn,
                            shuffle=True, drop_last=True)

    ckpt = torch.load("./checkpoints/t2m/test_res35/model/E33I0150000.tar", map_location="cpu")
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model)

    trainer.train(train_loader, val_loader)
