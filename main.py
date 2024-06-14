from setproctitle import setproctitle

setproctitle('momask')
import dotenv

dotenv.load_dotenv()
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

import torch
import utils.rotation_conversions as geometry

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed

import numpy as np
import binascii
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from utils.my_smpl import MySMPL
from numba import jit

clip_version = 'ViT-B/32'
device = torch.device("cuda")


def load_vq_model():
    vq_opt = get_opt("./checkpoints/t2m/test2_ft/opt.txt", device=device)
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
    ckpt = torch.load("./checkpoints/t2m/test2_ft/model/E01I0020000.tar", map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    vq_model.eval()
    vq_model.to(device)
    return vq_model, vq_opt


def load_trans_model(vq_opt):
    model_opt = get_opt("./checkpoints/t2m/test_large25_ft/opt.txt", device=device)
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
    # ckpt = torch.load("./checkpoints/t2m/test_large/model/E120I0540000.tar", map_location='cpu')
    # ckpt = torch.load("./checkpoints/t2m/test_large2/model/E36I0165000.tar", map_location='cpu')
    # ckpt = torch.load("./checkpoints/t2m/test_large_ft/model/E63I0020000.tar", map_location='cpu')
    ckpt = torch.load("./checkpoints/t2m/test_large25_ft/model/E63I0020000.tar", map_location='cpu')
    # ckpt = torch.load("./checkpoints/t2m/test_large_ft2/model/E63I0020000.tar", map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {model_opt.name} from epoch {ckpt["ep"]}!')
    t2m_transformer.eval()
    t2m_transformer.to(device)
    return t2m_transformer, model_opt


def load_res_model(vq_opt):
    res_opt = get_opt("./checkpoints/t2m/test_res35_ft/opt.txt", device=device)
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

    # ckpt = torch.load("./checkpoints/t2m/test_res_ft2/model/E79I0025000.tar", map_location='cpu')
    # ckpt = torch.load("./checkpoints/t2m/test_res_ft/model/E79I0025000.tar", map_location='cpu')
    ckpt = torch.load("./checkpoints/t2m/test_res35_ft/model/E79I0025000.tar", map_location='cpu')
    # ckpt = torch.load("./checkpoints/t2m/test_res3/model/latest.tar", map_location='cpu')
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    res_transformer.eval()
    res_transformer.to(device)
    return res_transformer, res_opt


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

data = {}


@app.on_event('startup')
def init_data():
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    parser = EvalT2MOptions()
    opt = parser.parse()
    sys.argv = old_argv
    fixseed(opt.seed)
    torch.autograd.set_detect_anomaly(True)
    vq_model, vq_opt = load_vq_model()
    t2m_transformer, model_opt = load_trans_model(vq_opt)
    res_transformer, res_opt = load_res_model(vq_opt)
    data["vq_model"] = vq_model
    data["t2m_transformer"] = t2m_transformer
    data["res_transformer"] = res_transformer
    data['smpl'] = MySMPL("checkpoints/smpl", gender="neutral", ext="pkl").to(device)
    data["opt"] = opt
    return data


@jit(nopython=True)
def accumulate_clip_y(y, leg, v):
    y[0] = min(max(y[0], leg[0]), 5.0)
    for i in range(1, len(y)):
        y[i] = min(max(y[i - 1] + v[i - 1], leg[i]), 5.0)


def fix_motion(root_positions, rotations):
    joints = data['smpl'](global_orient=rotations[:, 0],
                          body_pose=rotations[:, 1:].reshape(-1, 23 * 3),
                          transl=root_positions).joints
    min_y = joints[..., 1].min(1).values
    minmin_y = min_y.min()
    root_positions[:, 1] -= minmin_y
    min_y -= minmin_y
    leg_y = torch.clip(root_positions[:, 1] - min_y, 0.0, 5.0)
    vy = root_positions[1:, 1] - root_positions[:-1, 1]
    root_positions = root_positions.cpu().numpy()
    accumulate_clip_y(root_positions[:, 1], leg_y.cpu().numpy(), vy.cpu().numpy())
    return root_positions.astype(np.float32), rotations.cpu().numpy().astype(np.float32)


PROMPT = """You are a linguistician with expertise in expressing human motions. You translate user input into a short and precise English phrase to describe a human motion, like "a person is doing some action.", by following these rules:

(1) Always output in English. Whatever language the user input is in, you must translate the user input into an English phrase and only focus on the motion part. E.g., you translate "我在游泳" to "A person is swimming."

(2) Always describe a motion. Even if user input has no explicit meaning of a motion, you still translate it into a motion description with a linguistic technique called "Synaesthesia." Synaesthesia corresponds with the human brain's thinking, where a description leads to an association with a different, unrelated scenario. Take human actions as an example: "The wind blowing and leaves swaying" can be linked to "A person is swaying their arms dancing."; "A towering mountain" can be associated with "A person is standing silently."; and "Colorful clouds" can be connected to "A person is jumping around.".

(3) Best association. Connect a best-associated motion to the user input if it is a person's name or a concept related to a person. For example, "Michael Jackson" can result in "A person is dancing moonwalk steps.".

(4) Describe the motion in detail when there is enough info. For example, for the user input, "A person opening their arms in a welcoming gesture.", you can output "A person is standing, raising and waving both arms."

(5) When the input includes terminology, try to retain them as much as possible in the output.

"""


async def translation(prompt):
    for i in range(5):
        try:
            ret = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "system",
                           "content": PROMPT
                           },
                          {"role": "user", "content": prompt}],
                timeout=10,
                request_timeout=10,
            )
            prompt = ret["choices"][0]["message"]["content"]
            print(f"After translation: " + prompt + " -- Usage: " +
                  str(ret["usage"].to_dict()).replace("\n", ""))
            return prompt
        except Exception:
            pass
    print("Error 501, translation failed")
    return prompt


@app.get("/angle/")
async def angle(prompt: str, length: int = -1, temp: float = 1.0):
    prompt = await translation(prompt)
    vq_model, t2m_transformer, res_transformer, opt = data["vq_model"], data["t2m_transformer"], data[
        "res_transformer"], data["opt"]
    if length == -1:
        length = opt.max_motion_length
    else:
        length = np.clip(length, 64, opt.max_motion_length).item()
    temp = np.clip(temp, 0.1, 10.0).item()
    captions = [prompt]
    length_list = [length]

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(device).long()

    m_length = token_lens * 4

    with torch.no_grad():
        mids = t2m_transformer.generate(captions, token_lens,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=temp,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample)
        mids = res_transformer.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motion = vq_model.forward_decoder(mids)
        b, s, c = pred_motion.shape
        model_x = pred_motion[:, :, :3].clone()
        model_x[:, :, [0, 2]] = torch.cumsum(model_x[:, :, [0, 2]], dim=1)
        root_positions = model_x[0, :m_length[0]]

        model_m = geometry.rotation_6d_to_matrix(pred_motion[:, :, 3:].reshape(b, s, -1, 6))
        for i in range(1, s):
            model_m[:, i, 0] @= model_m[:, i - 1, 0]
        model_q = geometry.matrix_to_axis_angle(model_m)
        rotations = model_q[0, :m_length[0]]
        root_positions, rotations = fix_motion(root_positions, rotations)

    return {"root_positions": binascii.b2a_base64(
        root_positions.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode(
                "utf-8"),
            "dtype": "float32",
            "fps": 30,
            "mode": "axis_angle",
            "n_frames": root_positions.shape[0],
            "n_joints": 24}


@app.get("/")
async def healthy():
    return "OK"
