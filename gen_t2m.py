import os
from os.path import join as pjoin

import torch
import utils.rotation_conversions as geometry

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from utils.my_smpl import MySMPL

from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
import json
import binascii

clip_version = 'ViT-B/32'
device = torch.device("cuda")


def load_vq_model():
    vq_opt = get_opt("./checkpoints/t2m/test/opt.txt", device=device)
    vq_opt.dim_pose = 3 + 24 * 6
    vq_model = RVQVAE(vq_opt,
                      vq_opt.dim_pose,
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
    ckpt = torch.load("./checkpoints/t2m/test/model/latest.tar", map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    vq_model.eval()
    vq_model.to(device)
    return vq_model, vq_opt


def load_trans_model(vq_opt):
    model_opt = get_opt("./checkpoints/t2m/test_base/opt.txt", device=device)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load("./checkpoints/t2m/test_base/model/latest.tar", map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {model_opt.name} from epoch {ckpt["ep"]}!')
    t2m_transformer.eval()
    t2m_transformer.to(device)
    return t2m_transformer, model_opt


def load_res_model(vq_opt):
    res_opt = get_opt("./checkpoints/t2m/test_res/opt.txt", device=device)
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='text',
                                          latent_dim=res_opt.latent_dim,
                                          ff_size=res_opt.ff_size,
                                          num_layers=res_opt.n_layers,
                                          num_heads=res_opt.n_heads,
                                          dropout=res_opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=res_opt.cond_drop_prob,
                                          # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                          share_weight=res_opt.share_weight,
                                          clip_version=clip_version,
                                          opt=res_opt)

    ckpt = torch.load("./checkpoints/t2m/test_res/model/latest.tar", map_location='cpu')
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    res_transformer.eval()
    res_transformer.to(device)
    return res_transformer, res_opt


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)
    torch.autograd.set_detect_anomaly(True)

    result_dir = pjoin('./generation', "text2motion")
    os.makedirs(result_dir, exist_ok=True)

    vq_model, vq_opt = load_vq_model()
    t2m_transformer, model_opt = load_trans_model(vq_opt)
    res_transformer, res_opt = load_res_model(vq_opt)
    smpl = MySMPL("checkpoints/smpl", gender="neutral", ext="pkl").to(device)

    prompt_list = ["A person is running on a treadmill."]
    length_list = [opt.max_motion_length]

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(device).long()

    m_length = token_lens * 4
    captions = prompt_list
    kinematic_chain = t2m_kinematic_chain

    with torch.no_grad():
        mids = t2m_transformer.generate(captions, token_lens,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=opt.temperature,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample)
        mids = res_transformer.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motion = vq_model.forward_decoder(mids)
        b, s, c = pred_motion.shape
        model_x = pred_motion[:, :, :3].clone()
        model_x[:, :, [0, 2]] = torch.cumsum(model_x[:, :, [0, 2]], dim=1)
        model_x = model_x.reshape(b * s, -1)
        model_q = geometry.rotation_6d_to_axis_angle(pred_motion[:, :, 3:].reshape(b, s, -1, 6)).reshape(b * s, -1)
        model_xp = smpl(global_orient=model_q[:, :3], body_pose=model_q[:, 3:], transl=model_x).joints

        root_positions = model_x.reshape(b, s, 3).cpu().numpy()
        rotations = model_q.reshape(b, s, 24, 3).cpu().numpy()
        joints = model_xp[:, :22].reshape(b, s, 22, 3).cpu().numpy()

    for k, (caption, _root_positions, _rotations, _joints) in enumerate(
            zip(captions, root_positions, rotations, joints)):
        _m_length = m_length[k].item()

        _root_positions = _root_positions[:_m_length]
        _rotations = _rotations[:_m_length]
        _joints = _joints[:_m_length]
        _m_length = _root_positions.shape[0]

        print("---->Sample %d: %s %d" % (k, caption, _m_length))
        plot_3d_motion(pjoin(result_dir, f"{k}.mp4"), kinematic_chain, _joints, title=caption, fps=30)
        with open(pjoin(result_dir, f"{k}.json"), "w") as f:
            json.dump({"root_positions": binascii.b2a_base64(
                _root_positions.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                       "rotations": binascii.b2a_base64(_rotations.flatten().astype(np.float32).tobytes()).decode(
                           "utf-8"),
                       "dtype": "float32",
                       "fps": 30,
                       "mode": "axis_angle",
                       "n_frames": _m_length,
                       "n_joints": 24}, f, indent=4)
