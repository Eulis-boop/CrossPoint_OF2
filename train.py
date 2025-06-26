from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

#CHANGED: Import our custom dataset (ObjectFolder2) instead of ShapeNetRender
from datasets.ObjectFolder2 import ObjectFolder2, get_default_transform
from models.dgcnn import DGCNN, ResNet
from util import IOStream, AverageMeter

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def train(args, io):
    wandb.init(project="CrossPoint", name=args.exp_name)

    #CHANGED: Use our transform function
    transform = get_default_transform()

    #CHANGED: Replace ShapeNetRender with ObjectFolder2
    train_loader = DataLoader(
        ObjectFolder2(
            root_dir = args.data_path, #Will be passed via command line
            transform = transform
        ),
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 4
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    #Initialize models
    point_model = DGCNN(args).to(device)
    img_model = ResNet(resnet50(), feat_dim=2048).to(device)

    wandb.watch(point_model)

    if args.resume:
        point_model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded!")

    parameters = List(point_model.parameters()) + list(img_model.parameters())

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=aargs.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature=0.1).to(device)

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()

        #Training
        train_losses = AverageMeter()
        train_imid_losses = AverageMeter()
        train_cmid_losses = AverageMeter()

        point_model.train()
        img_model.train()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')

        #CHANGED: Data loading now returns (point_cloud, img) pairs
        for i, (point_clouds, imgs) in enumerate(train_loader):
            point_clouds, imgs = point_clouds.to(device), imgs.to(device)
            batch_size = point_clouds.size()[0]

            opt.zero_grad()

            #Process point clouds
            point_clouds = point_clouds.transpose(2, 1).contiguous()
            _, point_feats, _ = point_model(point_clouds)

            #Process images
            img_feats = img_model(imgs)

            #CHANGED: Simplified loss calculation (no dual point clouds)
            loss_cmid = criterion(point_feats, img_feats)
            total_loss = loss_cmid #Using only cross-modal loss initially

            total_loss.backward()
            opt.step()

            train_losses.update(total_loss.item(), batch_size)
            train_cmid_losses.update(loss_cmid.item(), batch_size)

            if i % args.print_freq == 0:
                print(f'Epoch ({epoch}), Batch({i}/{len(train_loader)}), loss: {train_losses.avg:.6f}, cmid loss: {train_cmid_losses.avg:.6f}')

            wandb_log['Train Loss'] = train_losses.avg
            wandb_log['Train CMID Loss'] = train_cmid_losses.avg

            outstr = f'Train {epoch}, loss: {train_losses.avg:.6f}'
            io.cprint(outstr)

            #Testing (keeping ModelNet40 for compatibility)
            train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
            test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024, batch_size=128, shuffle=True)

            feats_train = []
            labels_train = []
            point_model.eval()

            for i, (data, label) in enumerate(train_val_loader):
                labels = list(map(lambda x: x[0],label.numpy().tolist()))
                data = data.permute(0, 2, 1).to(device)
                with torch.no_grad():
                    feats = point_model(data)[2]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_train.append(feat)
                labels_train += labels

            feats_train = np.array(feats_train)
            labels_train = np.array(labels_train)

            feats_test = []
            labels_test = []

            for i, (data, label) in enumerate(test_val_loader):
                labels = list(map(lambda x: x[0],label.numpy().tolist()))
                data = data.permute(0, 2, 1).to(device)
                with torch.no_grad():
                    feats = point_model(data)[2]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_test.append(feat)
                labels_test += labels

            feats_test = np.array(feats_test)
            labels_test = np.array(labels_test)

            model_tl = SVC(C = 0.1, kernel ='linear')
            model_tl.fit(feats_train, labels_train)
            test_accuracy = model_tl.score(feats_test, labels_test)
            wandb_log['Linear Accuracy'] = test_accuracy
            print(f"Linear Accuracy : {test_accuracy}")

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                print('==> Saving Best Model...')
                save_file = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth')
                torch.save(point_model.state_dict(), save_file)

                save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/', 'img_model_best.pth')
                torch.save(img_model.state_dict(), save_img_model_file)

            wandb.log(wandb_log)

        print('==> Saving Last Model...')
        torch.save(point_model.state_dict(), os.path.join(f'checkpoints/{args.exp_name}/models/', 'ckpt_epoch_last.pth'))
        torch.save(img_model.state_dict(), os.path.join(f'checkpoints/{args._exp_name}/models/', 'img_model_last.pth'))

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Point Cloud Recognition')
        #Add new argument for data path
        parser.add_argument('--data_path', type=str, required=True, 
                            help='Path to processed ObjectFolder2 data')
        #All the original arguments
        parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                            help='Name of the experiment')
        parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                            choices=['dgcnn', 'dgcnn_seg'],
                            help='Model to use, [pointnet, dgcnn]')
        parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                            help='Size of batch)')
        parser.add_argument('--test_batch_size', type=int, default=16,
                            metavar='batch_size',
                            help='Size of batch)')
        parser.add_argument('--epochs', type=int, default=250, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001, 0.1 if using sgd)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--no_cuda', type=bool, default=False,
                            help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--eval', type=bool,  default=False,
                            help='evaluate the model')
        parser.add_argument('--num_points', type=int, default=2048,
                            help='num of points to use')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout rate')
        parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                            help='Dimension of embeddings')
        parser.add_argument('--k', type=int, default=20, metavar='N',
                            help='Num of nearest neighbors to use')
        parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
        parser.add_argument('--model_path', type=str, default='', metavar='N',
                            help='Pretrained model path')
        parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')

        args = parser.parse_args()
        _init_()

        io = IOStream('checkpoints/' + args.exp_name + '/run.log')
        io.cprint(str(args))

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        if args.cuda:
            io.cprint(f'Using GPU: {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
            torch.cuda.manual_seed(args.seed)
        else:
            io.cprint('Using CPU')

        if not args.eval:
            train(args, io)