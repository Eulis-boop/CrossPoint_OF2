from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import wandb
import time
from lightly.loss.ntx_ent_loss import NTXentLoss
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets.ObjectFolder2 import ObjectFolder2Dataset, get_default_transform
from models.dgcnn import DGCNN, ResNet
from util import IOStream, AverageMeter

from modelnet40_svm import ModelNet40SVM  # Assuming you have this file for downstream evaluation

def setup_directories(exp_name):
    os.makedirs(f'checkpoints/{exp_name}/models', exist_ok=True)

def evaluate_linear_svm(model, device):
    model.eval()
    train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
    test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=False)

    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for data, label in loader:
                data = data.permute(0, 2, 1).to(device)
                feat = model(data)[2].detach().cpu().numpy()
                features.extend(feat)
                labels.extend(label.numpy())
        return np.array(features), np.array(labels)

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    clf = SVC(C=0.1, kernel='linear')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc

def train(args, io):
    wandb.init(project="CrossPoint-OF2", name=args.exp_name)
    setup_directories(args.exp_name)

    device = torch.device("cuda" if args.cuda else "cpu")

    transform = get_default_transform()
    dataset = ObjectFolder2Dataset(root_dir=args.data_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Model setup
    point_model = DGCNN(args).to(device)
    img_model = ResNet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True), feat_dim=2048).to(device)

    if args.resume:
        point_model.load_state_dict(torch.load(args.model_path))
        io.cprint("Resumed model from checkpoint.")

    wandb.watch(point_model, log='all')

    optimizer = optim.Adam(list(point_model.parameters()) + list(img_model.parameters()), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = NTXentLoss(temperature=0.1).to(device)

    best_acc = 0
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

        acc = evaluate_linear_svm(point_model, device)
        wandb.log({"Linear SVM Accuracy": acc})
        io.cprint(f"Linear Evaluation Accuracy: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            io.cprint("Saving best model.")
            torch.save(point_model.state_dict(), f"checkpoints/{args.exp_name}/models/best_point_model.pth")
            torch.save(img_model.state_dict(), f"checkpoints/{args.exp_name}/models/best_img_model.pth")

        # Save last model every epoch
        torch.save(point_model.state_dict(), f"checkpoints/{args.exp_name}/models/last_point_model.pth")
        torch.save(img_model.state_dict(), f"checkpoints/{args.exp_name}/models/last_img_model.pth")

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
