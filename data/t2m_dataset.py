from os.path import join as pjoin
from torch.utils import data
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
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
        rotation_6d = geometry.axis_angle_to_rotation_6d(
            torch.tensor(rotations, dtype=torch.float32)).reshape(-1, 24 * 6).numpy()
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


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, splits):
        self.opt = opt
        self.max_length = 30
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length

        data_dict = {}

        new_name_list = []
        length_list = []
        for name in tqdm(splits):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if len(motion) >= 200:
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 30): int(to_tag * 30)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
