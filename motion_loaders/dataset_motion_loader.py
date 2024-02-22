from data.t2m_dataset import Text2MotionDatasetEval, collate_fn  # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt


def get_dataset_motion_loader(opt_path, batch_size, splits, device):
    opt = get_opt(opt_path, device)
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    dataset = Text2MotionDatasetEval(opt, splits, w_vectorizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                            collate_fn=collate_fn, shuffle=True)
    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
