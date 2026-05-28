import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from PSRNet import PSRNet, init_weights
from lib.utils import LFDataset


class Logger(object):
    def __init__(self, filename="exp.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()


def prepare_dir(path: str, name: str):
    if os.path.exists(path) and (not os.path.isdir(path)):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{path}_{name}_{ts}"
    os.makedirs(path, exist_ok=True)
    return path


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data.clamp_(-grad_clip, grad_clip)


def scalarize(x, default=0.0):
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().float().mean().cpu().item())
    if isinstance(x, (float, int)):
        return float(x)
    return float(default)


def focal_loss(pred, mask, gamma=2.0, alpha=0.25):
    pred_sig = torch.sigmoid(pred)
    pt = (1 - pred_sig) * mask + pred_sig * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()


def hybrid_e_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred_sig = torch.sigmoid(pred)
    mpred = pred_sig.mean(dim=(2, 3), keepdim=True)
    phiFM = pred_sig - mpred
    mmask = mask.mean(dim=(2, 3), keepdim=True)
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)
    return (bce + eloss + wiou).mean()


def soft_morphological_gradient(x: torch.Tensor, kernel_size: int = 3):
    pad = kernel_size // 2
    dil = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_consistency_loss(contour_logits: torch.Tensor, pred_logits: torch.Tensor, coarse_logits: torch.Tensor, kernel_size: int = 3):
    contour_prob = torch.sigmoid(contour_logits)
    pred_bd = soft_morphological_gradient(torch.sigmoid(pred_logits).detach(), kernel_size=kernel_size)
    coarse_bd = soft_morphological_gradient(torch.sigmoid(coarse_logits).detach(), kernel_size=kernel_size)
    loss_pred = F.l1_loss(contour_prob, pred_bd)
    loss_coarse = F.l1_loss(contour_prob, coarse_bd)
    return loss_pred + 0.5 * loss_coarse


def fs_to_bs3hw(fs: torch.Tensor, num_slices: int = None):
    if fs.dim() == 5:
        return fs
    if fs.dim() == 4:
        if fs.size(1) == 3:
            S, C, H, W = fs.shape
            assert C == 3
            return fs.unsqueeze(0)
        B, C, H, W = fs.shape
        assert C % 3 == 0
        S = C // 3
        if num_slices is not None:
            assert S == num_slices
        return fs.contiguous().view(B, S, 3, H, W)
    raise ValueError(f"Unsupported fs shape: {fs.shape}")


def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (p.ndim == 1) or name.endswith('.bias') or ('norm' in lname) or ('bn' in lname):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': float(weight_decay)}, {'params': no_decay, 'weight_decay': 0.0}]


def apply_backbone_warmup_requires_grad(model: torch.nn.Module, epoch: int, warm_epochs: int, freeze_bn: bool = True):
    total_params = 0
    trainable_params = 0
    warm_on = (warm_epochs is not None) and (int(warm_epochs) > 0) and (epoch < int(warm_epochs))
    if warm_on:
        for p in model.parameters():
            p.requires_grad = True
        backbone_prefixes = ("backbone_rgb.", "backbone_fs.", "backbone_depth.")
        for name, p in model.named_parameters():
            total_params += p.numel()
            if name.startswith(backbone_prefixes):
                p.requires_grad = False
            if p.requires_grad:
                trainable_params += p.numel()
        if hasattr(model, "backbone_rgb"):
            model.backbone_rgb.eval()
        if hasattr(model, "backbone_fs"):
            model.backbone_fs.eval()
        if hasattr(model, "backbone_depth"):
            model.backbone_depth.eval()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    m.eval()
    else:
        for p in model.parameters():
            total_params += p.numel()
            p.requires_grad = True
            trainable_params += p.numel()
    return warm_on, trainable_params, total_params



@torch.no_grad()
def evaluate(args, model, datasets, device):
    model.eval()
    maes = []
    for dataset in datasets:
        test_loader = DataLoader(
            LFDataset(location=os.path.join(args.eval_data_location, dataset) + '/', crop=False, train=False, image_size=args.image_size),
            batch_size=1, shuffle=False, num_workers=args.num_worker
        )
        mae_sum = 0.0
        for allfocus, fs, depth, gt, names in test_loader:
            rgb = allfocus.to(device)
            depth = depth.to(device)
            gt = gt.to(device)
            fs = fs.to(device)
            fs = fs_to_bs3hw(fs, num_slices=args.num_slices)
            pred, contour_pred, coarse, aux = model(fs, rgb, depth, gt=None, return_aux=True)
            pred_prob = torch.sigmoid(pred)
            mae_sum += torch.abs(pred_prob - gt).mean().item()
        maes.append(mae_sum / max(len(test_loader), 1))
    return float(np.mean(maes))


def train(args, model, train_loader, device, optimizer, scheduler, writer):
    best_mae = 1e9
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        warm_on, n_tr, n_all = apply_backbone_warmup_requires_grad(
            model,
            epoch=epoch,
            warm_epochs=args.backbone_warm_epochs,
            freeze_bn=args.backbone_warm_freeze_bn
        )
        if writer is not None:
            writer.add_scalar("train/backbone_warm_on", 1.0 if warm_on else 0.0, epoch)
            writer.add_scalar("train/trainable_ratio", float(n_tr) / float(max(n_all, 1)), epoch)

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        edge_w = args.edge_loss_weight_after if epoch >= args.edge_weight_after_epoch else args.edge_loss_weight_before

        for it, (allfocus, fs, depth, gt, contour, names) in enumerate(train_loader):
            rgb = allfocus.to(device)
            depth = depth.to(device)
            gt = gt.to(device)
            contour = contour.to(device)
            fs = fs.to(device)
            fs = fs_to_bs3hw(fs, num_slices=args.num_slices)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred, contour_pred, coarse, aux = model(fs, rgb, depth, gt=gt, return_aux=True)
                loss_sal = hybrid_e_loss(pred, gt)
                loss_edge = focal_loss(contour_pred, contour, gamma=args.focal_gamma, alpha=args.focal_alpha)
                loss_coarse = hybrid_e_loss(coarse, gt)
                loss_route = aux['route_loss']
                edge_elastica_loss = aux['edge_aux']['edge_elastica_loss']
                loss_bc = boundary_consistency_loss(contour_pred, pred, coarse, kernel_size=args.boundary_kernel)

                loss = (
                    args.sal_loss_weight * loss_sal +
                    edge_w * loss_edge +
                    args.coarse_loss_weight * loss_coarse +
                    args.route_loss_weight * loss_route +
                    args.edge_elastica_loss_weight * edge_elastica_loss +
                    args.boundary_consistency_weight * loss_bc
                )
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            running += loss.item()

            if (it + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (it + 1) % args.print_freq == 0:
                if writer is not None:
                    loss_show = running * args.accum_steps / args.print_freq
                    lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/loss", loss_show, global_step)
                running = 0.0

            global_step += 1

        if len(train_loader) % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        mae = evaluate(args, model, args.eval_dataset, device)
        print(f"[Epoch {epoch}] Val MAE = {mae:.6f}")

        if writer is not None:
            writer.add_scalar("val/mae", mae, epoch)

        if mae < best_mae:
            best_mae = mae
            save_path = os.path.join(args.model_path, "best.pth")
            torch.save(model.state_dict(), save_path)

        if epoch >= args.save_after and (epoch % args.save_every == 0):
            p = os.path.join(args.model_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), p)


def parse_args():
    p = argparse.ArgumentParser("Train PSRNet with Euler-Elastica Boundary Transformer")
    p.add_argument("--model_path", type=str, default="models/PSRNet_EEBT")
    p.add_argument("--log_path", type=str, default="log/PSRNet_EEBT")
    p.add_argument("--pretrained_model", type=str, default="./pre_trained/swin_tiny_patch4_window7_224.pth")
    p.add_argument("--cuda", type=str, default="0")
    p.add_argument("--train_data_location", type=str, default="./data/train/DUTLF-FS/")
    p.add_argument("--eval_data_location", type=str, default="./data/test/")
    p.add_argument("--eval_dataset", nargs="+", default=["DUTLF-FS"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_worker", type=int, default=0)
    p.add_argument("--num_slices", type=int, default=12)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--region_temperature", type=float, default=0.7)
    p.add_argument("--hyper_blocks", type=int, default=1)
    p.add_argument("--hyper_heads", type=int, default=4)
    p.add_argument("--hyper_mlp_ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--boundary_blocks", type=int, default=2)
    p.add_argument("--boundary_heads", type=int, default=4)
    p.add_argument("--boundary_token_hw", type=int, default=14)
    p.add_argument("--evolution_steps", type=int, default=3)
    p.add_argument("--evolution_hidden_ratio", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--backbone_warm_epochs", type=int, default=20)
    p.add_argument("--backbone_warm_freeze_bn", action="store_false", default=True)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--use_tensorboard", action="store_true", default=False)
    p.add_argument("--sal_loss_weight", type=float, default=1.0)
    p.add_argument("--coarse_loss_weight", type=float, default=1.0)
    p.add_argument("--route_loss_weight", type=float, default=0.3)
    p.add_argument("--edge_weight_after_epoch", type=int, default=140)
    p.add_argument("--edge_loss_weight_before", type=float, default=1.0)
    p.add_argument("--edge_loss_weight_after", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--edge_elastica_loss_weight", type=float, default=0.20)
    p.add_argument("--boundary_consistency_weight", type=float, default=0.20)
    p.add_argument("--boundary_kernel", type=int, default=3)
    p.add_argument("--save_after", type=int, default=0)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    args.model_path = prepare_dir(args.model_path, "model")
    args.log_path = prepare_dir(args.log_path, "tb")
    log_file = os.path.join(args.model_path, "exp.txt")
    sys.stdout = Logger(log_file)

    writer = SummaryWriter(args.log_path) if args.use_tensorboard else None
    train_set = LFDataset(location=args.train_data_location, image_size=args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True, drop_last=True)

    model = PSRNet(
        backbone_type="swin",
        num_slices=args.num_slices,
        topk=args.topk,
        region_temperature=args.region_temperature,
        hyper_blocks=args.hyper_blocks,
        hyper_heads=args.hyper_heads,
        hyper_mlp_ratio=args.hyper_mlp_ratio,
        dropout=args.dropout,
        boundary_blocks=args.boundary_blocks,
        boundary_heads=args.boundary_heads,
        boundary_token_hw=args.boundary_token_hw,
        evolution_steps=args.evolution_steps,
        evolution_hidden_ratio=args.evolution_hidden_ratio,
    )
    model.apply(init_weights)
    model.load_pretrained(args.pretrained_model)
    model.to(device)

    optimizer = torch.optim.AdamW(build_adamw_param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    train(args, model, train_loader, device, optimizer, scheduler, writer)
    if writer is not None:
        writer.close()
