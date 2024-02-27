import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F
import utils.rotation_conversions as geometry
from utils.my_smpl import MySMPL
import shutil

import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss

import os
import sys
import json


def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        self.smpl = MySMPL("checkpoints/smpl", gender="neutral", ext="pkl").to(self.device)

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions)

        # full reconstruction loss
        loss_rec = self.l1_criterion(pred_motion, motions)

        # velocity loss
        model_out_v = pred_motion[:, 1:] - pred_motion[:, :-1]
        model_out_v[:, :, [0, 2]] = pred_motion[:, 1:, [0, 2]]
        target_v = motions[:, 1:] - motions[:, :-1]
        target_v[:, :, [0, 2]] = motions[:, 1:, [0, 2]]
        loss_v = self.l1_criterion(model_out_v, target_v)

        # FK loss
        b, s, c = pred_motion.shape

        model_x = pred_motion[:, :, :3].clone()
        model_x[:, :, [0, 2]] = torch.cumsum(model_x[:, :, [0, 2]], dim=1)
        model_x = model_x.reshape(b * s, -1)
        model_q = geometry.rotation_6d_to_axis_angle(pred_motion[:, :, 3:].reshape(b, s, -1, 6)).reshape(b * s, -1)

        target_x = motions[:, :, :3].clone()
        target_x[:, :, [0, 2]] = torch.cumsum(target_x[:, :, [0, 2]], dim=1)
        target_x = target_x.reshape(b * s, -1)
        target_q = geometry.rotation_6d_to_axis_angle(motions[:, :, 3:].reshape(b, s, -1, 6)).reshape(b * s, -1)

        model_xp = self.smpl(global_orient=model_q[:, :3], body_pose=model_q[:, 3:], transl=model_x).joints
        target_xp = self.smpl(global_orient=target_q[:, :3], body_pose=target_q[:, 3:], transl=target_x).joints

        loss_explicit = self.l1_criterion(model_xp, target_xp)

        self.motions = target_xp[:, :22].reshape(b, s, 22, 3)
        self.pred_motion = model_xp[:, :22].reshape(b, s, 22, 3)

        loss = 0.636 * loss_rec + 2.964 * loss_v + 0.646 * loss_explicit + 0.02 * loss_commit

        return loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity

    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it, val_loss):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'val_loss': val_loss
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def update_logs(self, logs, loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity):
        logs['loss'] += loss.item()
        logs['loss_rec'] += loss_rec.item()
        logs['loss_v'] += loss_v.item()
        logs['loss_explicit'] += loss_explicit.item()
        logs['loss_commit'] += loss_commit.item()
        logs['perplexity'] += perplexity.item()
        logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

    def train(self, train_loader, val_loader, plot_eval=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99),
                                        weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones,
                                                              gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())
        self.vq_model.train()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                self.update_logs(logs, loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity)
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    print('Validation time:')
                    self.vq_model.eval()
                    val_logs = defaultdict(def_value, OrderedDict())
                    with torch.no_grad():
                        for i, batch_data in enumerate(val_loader):
                            loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity = self.forward(batch_data)
                            self.update_logs(val_logs, loss, loss_rec, loss_v, loss_explicit, loss_commit, perplexity)
                    val_mean_loss = OrderedDict()
                    for tag, value in val_logs.items():
                        val_mean_loss[tag] = value / len(val_loader)
                        self.logger.add_scalar('Val/%s' % tag, val_mean_loss[tag], it)

                    print(
                        'Validation Loss: %.5f, Reconstruction: %.5f, Velocity: %.5f, Explicit: %.5f, Commit: %.5f, Perplexity: %.5f'
                        % (val_mean_loss['loss'], val_mean_loss['loss_rec'], val_mean_loss['loss_v'],
                           val_mean_loss['loss_explicit'],
                           val_mean_loss['loss_commit'], val_mean_loss['perplexity']))
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, val_mean_loss)
                    shutil.copyfile(pjoin(self.opt.model_dir, 'latest.tar'),
                                    pjoin(self.opt.model_dir, 'E%02dI%07d.tar' % (epoch, it)))
                    with open(pjoin(self.opt.model_dir, 'val_loss.jsonl'), "a") as f:
                        f.write(json.dumps({'E%02dI%07d' % (epoch, it): val_mean_loss}) + "\n")
                    data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                    save_dir = pjoin(self.opt.eval_dir, 'E%02dI%07d' % (epoch, it))
                    os.makedirs(save_dir, exist_ok=True)
                    plot_eval(data, save_dir)
                    self.vq_model.train()

            epoch += 1


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s" % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()

            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss
