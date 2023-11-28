import dataclasses
from contextlib import nullcontext

import torch, torch.nn as nn
import numpy as np
# from tqdm import trange, tqdm
from tqdm.autonotebook import trange, tqdm

import util


@dataclasses.dataclass
class TrainConfig:
    epochs: int
    batchs_per_epoch: int
    batch_size: int

    min_lr: float
    max_lr: float

    device: str
    mixed: bool

    patience: int
    warmup_iters: int

    grad_acc_steps: int
    grad_clip: float


def gen_train_config(device, batches_per_epoch, epochs, lr=1e-3, bs=64):
    warmup_iters=int(epochs*batches_per_epoch * 0.05)
    mx = device == 'cuda'
    return TrainConfig(epochs=epochs, batchs_per_epoch=batches_per_epoch, 
                       batch_size=bs, min_lr=lr/10, max_lr=lr, device=device, 
                       mixed=mx, patience=2, warmup_iters=warmup_iters, 
                       grad_clip=1.0, grad_acc_steps=1)


def update_status_str(epoch, tl, vl, ta, va):
    return 'epoch {:02d} TL {:.4f} TA {:.4f} VL {:.4f} VA {:.4f}'.format(
            epoch, tl, ta, vl, va)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.max_iters = cfg.epochs * cfg.batchs_per_epoch
        self.ema_loss = None
        self.ema_acc = None

        self.on_epoch_end = None
        self.on_best_model = None

        self.bls, self.accs = [], []
        self.metrics = []

    def get_lr(self, epoch, n_batch):
        cfg = self.cfg
        t = epoch * cfg.batchs_per_epoch + n_batch
        if t < cfg.warmup_iters:
            return cfg.max_lr * t / cfg.warmup_iters
        r = (t - cfg.warmup_iters) / self.max_iters
        return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + np.cos(r*np.pi))

    def train(self, model, dl_train, dl_val, n_eval_iters=None):
        cfg = self.cfg
        mixed = cfg.mixed and cfg.device == 'cuda'

        if n_eval_iters is None:
            n_eval_iters = len(dl_val)

        opt = model.configure_optimizers(cfg.device)

        self.ema_loss = util.EMA()
        self.ema_acc = util.EMA()
        scaler = torch.cuda.amp.GradScaler(enabled=mixed)
        total_step = 0
        best_vl = None
        iters_without_improv = 0

        metrics = util.estimate_metrics(model, dl_train, dl_val, n_eval_iters)
        tl, vl = metrics['train_loss'], metrics['val_loss']
        ta, va = metrics['train_acc'], metrics['val_acc']
        best_vl = vl

        if mixed:
            ctx = torch.autocast(device_type=cfg.device, dtype=torch.bfloat16)
        else:
            ctx = nullcontext()

        self.bls, self.accs = [], []
        self.metrics = [metrics]

        model.train()
        for epoch in (te := trange(cfg.epochs, leave=True)):
            te.set_description(update_status_str(epoch, tl, vl, ta, va))

            tqdm_ = tqdm(enumerate(dl_train), total=cfg.batchs_per_epoch)
            for n_batch, ((x_im, x_ctx), target) in tqdm_:
                x_im, x_ctx = x_im.to(cfg.device), x_ctx.to(cfg.device)
                target = target.to(cfg.device)
                lr = self.get_lr(epoch, n_batch)
                for g in opt.param_groups:
                    g['lr'] = lr

                with ctx:
                    logits, loss = model(x_im, x_ctx, target)
                    loss = loss / cfg.grad_acc_steps

                scaler.scale(loss).backward()

                total_step += 1
                if total_step % cfg.grad_acc_steps != 0:
                    continue

                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                self.ema_loss.update(loss.item() * cfg.grad_acc_steps)
                with torch.no_grad():
                    acc = torch.mean(logits.argmax(-1) == target, 
                                     dtype=torch.float32)
                    self.ema_acc.update(acc.item())

                self.bls.append(self.ema_loss.value)
                self.accs.append(self.ema_acc.value)
                tqdm_.set_description('epoch {:02d} loss {:.4f} acc {:.4f}'.format(
                    epoch, self.ema_loss.value, self.ema_acc.value))

            metrics = util.estimate_metrics(model, dl_train, dl_val, n_eval_iters)
            tl, vl = metrics['train_loss'], metrics['val_loss']
            ta, va = metrics.get('train_acc', None), metrics.get('val_acc', None)
            self.metrics.append(metrics)

            te.set_description(update_status_str(epoch, tl, vl, ta, va))

            if best_vl is None or vl < best_vl:
                best_vl = vl
                iters_without_improv = 0
                if self.on_best_model is not None:
                    self.on_best_model(self, model, epoch)
            else:
                iters_without_improv += 1

            if self.on_epoch_end:
                self.on_epoch_end(self, model, epoch)

            if iters_without_improv > cfg.patience:
                print('** val loss not improving, early stopping...')
                break

        
        print('** best VL {:.4f}'.format(best_vl))
