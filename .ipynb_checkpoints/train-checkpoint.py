from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import wandb
import time

from lightly.loss.ntx_ent_loss import NTXentLoss
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets.ObjectFolder2 import ObjectFolder2Dataset, get_default_transform
from models.dgcnn import DGCNN, ResNet
from util import IOStream, AverageMeter

def setup_directories(exp_name):
    os.makedirs(f'checkpoints/{exp_name}/models', exist_ok=True)
    os.makedirs(f'checkpoints/{exp_name}/features', exist_ok=True)

def train(args, io):
    wandb.init(project="CrossPoint-OF2", name=args.exp_name)
    setup_directories(args.exp_name)

    device = torch.device("cuda" if args.cuda else "cpu")

    transform = get_default_transform()
    dataset = ObjectFolder2Dataset(root_dir=args.data_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Model setup
    point_model = DGCNN(args).to(device)
    img_model = ResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True), feat_dim=256).to(device)

    if args.resume:
        point_model.load_state_dict(torch.load(args.model_path))
        io.cprint("Resumed model from checkpoint.")

    wandb.watch(point_model, log='all')

    optimizer = optim.Adam(list(point_model.parameters()) + list(img_model.parameters()), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = NTXentLoss(temperature=0.1).to(device)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        point_model.train()
        img_model.train()

        loss_meter = AverageMeter()

        for i, (pc, img) in enumerate(train_loader):
            pc, img = pc.to(device), img.to(device)
            pc = pc.transpose(2, 1)

            optimizer.zero_grad()
            _, pc_feat, _ = point_model(pc)
            img_feat = img_model(img)

            loss = criterion(pc_feat, img_feat)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), pc.size(0))

            if i % args.print_freq == 0:
                io.cprint(f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss_meter.avg:.4f}")

        wandb.log({"Train Loss": loss_meter.avg, "Epoch": epoch})

        # Save model checkpoints
        torch.save(point_model.state_dict(), f"checkpoints/{args.exp_name}/models/point_model_epoch{epoch}.pth")
        torch.save(img_model.state_dict(), f"checkpoints/{args.exp_name}/models/img_model_epoch{epoch}.pth")

        point_model.eval()
        img_model.eval()
        all_pc_features, all_img_features = [], []
        with torch.no_grad():
            for pc, img in train_loader:
                pc, img = pc.to(device), img.to(device)
                pc = pc.transpose(2, 1)
                _, pc_feat, _ = point_model(pc)
                img_feat = img_model(img)
                all_pc_features.append(pc_feat.cpu())
                all_img_features.append(img_feat.cpu())
        all_pc_features = torch.cat(all_pc_features)
        all_img_features = torch.cat(all_img_features)
        torch.save(all_pc_features, f'checkpoints/{args.exp_name}/features/pc_features_epoch{epoch}.pt')
        torch.save(all_img_features, f'checkpoints/{args.exp_name}/features/img_features_epoch{epoch}.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='of2_crosspoint')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--print_freq', type=int, default=20)
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(42)
    if args.cuda:
        torch.cuda.manual_seed_all(42)

    io = IOStream(f'checkpoints/{args.exp_name}/log.txt')
    io.cprint(str(args))

    train(args, io)

if __name__ == "__main__":
    main()