from torch.utils import data
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
import binascii
import re
import torch
import os
import utils.rotation_conversions as geometry
import pickle as pk


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def mirror_text(text, pattern):
    def replace(match):
        word = match.group(0)
        if word == "left":
            return "right"
        elif word == "right":
            return "left"
        return word

    return pattern.sub(replace, text)


def mirror_motion(rotations, root_positions):
    rotations, root_positions = rotations.copy(), root_positions.copy()
    mirror_chain = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    rotations[:, :, 0] = rotations[:, mirror_chain, 0]
    rotations[:, :, 1] = -rotations[:, mirror_chain, 1]
    rotations[:, :, 2] = -rotations[:, mirror_chain, 2]
    root_positions[:, 0] *= -1
    return rotations, root_positions


def motion_to_147(rotations, root_positions):
    velocity = root_positions.copy()
    with torch.no_grad():
        # rotation_6d = geometry.axis_angle_to_rotation_6d(
        #     torch.tensor(rotations, dtype=torch.float32)).reshape(-1, 24 * 6).numpy()
        matrix = geometry.axis_angle_to_matrix(
            torch.tensor(rotations, dtype=torch.float32)).reshape(-1, 24, 3, 3)
        root_matrix = matrix[1:, 0] @ matrix[:-1, 0].inverse()
        matrix[1:, 0] = root_matrix
        rotation_6d = geometry.matrix_to_rotation_6d(matrix).reshape(-1, 24 * 6).numpy()
    velocity[1:, [0, 2]] = root_positions[1:, [0, 2]] - root_positions[:-1, [0, 2]]
    velocity[0, [0, 2]] = 0
    rotations_147 = np.concatenate([velocity, rotation_6d], axis=1, dtype=np.float32)
    return rotations_147


class MotionDataset(data.Dataset):
    def __init__(self, opt, splits, cache_file):
        self.opt = opt
        self.splits = np.array(splits, dtype=object)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                data = pk.load(f)
            self.cumsum = data["cumsum"]
            self.lengths = data["lengths"]
        else:
            self.lengths = []
            for name in tqdm(splits):
                with open(opt.data_root + name, encoding='utf-8') as f:
                    data = json.load(f)
                self.lengths.append(data["n_frames"] - opt.window_size)

            self.cumsum = np.cumsum([0] + self.lengths)
            self.lengths = np.int32(self.lengths)

            with open(cache_file, "wb") as f:
                pk.dump({
                    "cumsum": self.cumsum,
                    "lengths": self.lengths}, f)

        print("Total number of motions {}, snippets {}".format(len(self.lengths), self.cumsum[-1]))

    def inv_transform(self, data):
        return data

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        with open(self.opt.data_root + self.splits[motion_id], encoding='utf-8') as f:
            data = json.load(f)
        rotations = np.frombuffer(binascii.a2b_base64(data["rotations"]),
                                  dtype=data["dtype"]).reshape(-1, 24, 3)[idx:idx + self.opt.window_size]
        root_positions = np.frombuffer(binascii.a2b_base64(data["root_positions"]),
                                       dtype=data["dtype"]).reshape(-1, 3)[idx:idx + self.opt.window_size]
        if random.random() > 0.5:
            rotations, root_positions = mirror_motion(rotations, root_positions)
        motion147 = motion_to_147(rotations, root_positions)
        return motion147


def collate_fn(batch):
    # 分别提取数据和标签
    text = [item[0] for item in batch]
    motion147 = [item[1] for item in batch]
    m_length = [item[2] for item in batch]

    # 填充数据使它们具有相同的长度
    motion147 = pad_sequence(motion147, batch_first=True, padding_value=0)

    # 将标签转换为张量
    m_length = torch.tensor(m_length, dtype=torch.int64)

    return text, motion147, m_length


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, splits):
        self.opt = opt
        self.max_motion_length = opt.max_motion_length
        self.splits = np.array(splits, dtype=object)
        self.pattern = re.compile(r'\bleft\b|\bright\b')

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, item):
        with open(self.opt.data_root + self.splits[item], encoding='utf-8') as f:
            data = json.load(f)
        rotations = np.frombuffer(binascii.a2b_base64(data["rotations"]),
                                  dtype=data["dtype"]).reshape(-1, 24, 3)
        root_positions = np.frombuffer(binascii.a2b_base64(data["root_positions"]),
                                       dtype=data["dtype"]).reshape(-1, 3)
        if data['n_frames'] > self.max_motion_length:
            m_length = self.max_motion_length
        else:
            if random.random() > 0.67:
                m_length = (data['n_frames'] // self.opt.unit_length - 1) * self.opt.unit_length
            else:
                m_length = (data['n_frames'] // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, data['n_frames'] - m_length)
        rotations = rotations[idx:idx + m_length]
        root_positions = root_positions[idx:idx + m_length]
        desc = random.choice(data['desc'])
        action = data.get('action')
        if random.random() > 0.5:
            rotations, root_positions = mirror_motion(rotations, root_positions)
            desc = mirror_text(desc, self.pattern)
        motion147 = motion_to_147(rotations, root_positions)
        if action:
            text = action + " || " + desc
        else:
            text = desc
        return text, torch.from_numpy(motion147), m_length
