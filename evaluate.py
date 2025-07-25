import os
import argparse
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets.ObjectFolder2 import ObjectFolder2Dataset, get_default_transform
from models.dgcnn import DGCNN, ResNet

@torch.no_grad()
def extract_features(dataloader, point_model, image_model, device):
    point_model.eval()
    image_model.eval()

    point_feats, img_feats, labels = [], [], []

    for pc, img, label in dataloader:
        pc = pc.to(device).transpose(2, 1)
        img = img.to(device)

        _, pc_emb, _ = point_model(pc)
        img_emb = image_model(img)

        point_feats.append(pc_emb.cpu().numpy())
        img_feats.append(img_emb.cpu().numpy())
        labels.extend(label)

    return (
        np.vstack(point_feats),
        np.vstack(img_feats),
        np.array(labels)
    )

def main(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Dataset
    transform = get_default_transform()
    dataset = ObjectFolder2Dataset(
        root_dir=args.data_path,
        transform=transform,
        return_label=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Models
    from torchvision.models import resnet18
    point_model = DGCNN(args).to(device)
    image_model = ResNet(resnet18(pretrained=True), feat_dim=256).to(device)

    point_model.load_state_dict(torch.load(args.point_model_path))
    image_model.load_state_dict(torch.load(args.image_model_path))

    # Feature extraction
    print("Extracting features...")
    pc_feats, img_feats, labels = extract_features(dataloader, point_model, image_model, device)

    # Classification
    print("Training SVM on point cloud embeddings...")
    clf_pc = SVC(kernel='linear', C=1.0)
    clf_pc.fit(pc_feats, labels)
    acc_pc = clf_pc.score(pc_feats, labels)
    print(f"Accuracy (SVM on point cloud features): {acc_pc:.4f}")

    print("Training SVM on image embeddings...")
    clf_img = SVC(kernel='linear', C=1.0)
    clf_img.fit(img_feats, labels)
    acc_img = clf_img.score(img_feats, labels)
    print(f"Accuracy (SVM on image features): {acc_img:.4f}")

    # t-SNE visualization
    if args.tsne_plot:
        print("Generating t-SNE plot...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(pc_feats)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=6)
        plt.title("t-SNE of Point Cloud Embeddings")
        plt.savefig(os.path.join(args.output_dir, "tsne_pointclouds.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--point_model_path', type=str, required=True)
    parser.add_argument('--image_model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--tsne_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
