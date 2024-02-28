import torch
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from models.mask_transformer.tools import *

import shutil
import json


def def_value():
    return 0.0


class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it, val_loss):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'val_loss': val_loss
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'],
                                                                             strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])  # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler'])  # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr,
                                               weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        self.t2m_transformer.train()
        while epoch < self.opt.max_epoch:
            self.vq_model.eval()
            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 80:
                    print('Validation time:')
                    self.vq_model.eval()
                    self.t2m_transformer.eval()
                    val_logs = defaultdict(def_value, OrderedDict())
                    with torch.no_grad():
                        for i, batch_data in enumerate(val_loader):
                            loss, acc = self.forward(batch_data)
                            val_logs['loss'] += loss.item()
                            val_logs["acc"] += acc
                            val_logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']
                    val_mean_loss = OrderedDict()
                    for tag, value in val_logs.items():
                        val_mean_loss[tag] = value / len(val_loader)
                        self.logger.add_scalar('Val/%s' % tag, val_mean_loss[tag], it)
                    print(f"Validation loss:{val_mean_loss['loss']:.5f}, Accuracy:{val_mean_loss['acc']:.5f}")
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, val_mean_loss)
                    shutil.copyfile(pjoin(self.opt.model_dir, 'latest.tar'),
                                    pjoin(self.opt.model_dir, 'E%02dI%07d.tar' % (epoch, it)))
                    with open(pjoin(self.opt.model_dir, 'val_loss.jsonl'), "a") as f:
                        f.write(json.dumps({'E%02dI%07d' % (epoch, it): val_mean_loss}) + "\n")
                    self.t2m_transformer.train()
            epoch += 1


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            # self.l1_criterion = torch.nn.SmoothL1Loss()

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it, val_loss):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'val_loss': val_loss
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'],
                                                                             strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer'])  # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler'])  # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr,
                                               weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        self.res_transformer.train()
        while epoch < self.opt.max_epoch:
            self.vq_model.eval()
            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 80:
                    print('Validation time:')
                    self.vq_model.eval()
                    self.res_transformer.eval()
                    val_logs = defaultdict(def_value, OrderedDict())
                    with torch.no_grad():
                        for i, batch_data in enumerate(val_loader):
                            loss, acc = self.forward(batch_data)
                            val_logs['loss'] += loss.item()
                            val_logs["acc"] += acc
                            val_logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']
                    val_mean_loss = OrderedDict()
                    for tag, value in val_logs.items():
                        val_mean_loss[tag] = value / len(val_loader)
                        self.logger.add_scalar('Val/%s' % tag, val_mean_loss[tag], it)
                    print(f"Validation loss:{val_mean_loss['loss']:.5f}, Accuracy:{val_mean_loss['acc']:.5f}")

                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, val_mean_loss)
                    shutil.copyfile(pjoin(self.opt.model_dir, 'latest.tar'),
                                    pjoin(self.opt.model_dir, 'E%02dI%07d.tar' % (epoch, it)))
                    with open(pjoin(self.opt.model_dir, 'val_loss.jsonl'), "a") as f:
                        f.write(json.dumps({'E%02dI%07d' % (epoch, it): val_mean_loss}) + "\n")
                    self.res_transformer.train()
            epoch += 1
