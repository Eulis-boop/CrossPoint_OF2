import os
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
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

    pc_feats_list, img_feats_list, labels_list = [], [], []

    for pc, img, label in dataloader:
        pc = pc.to(device).transpose(2, 1)
        img = img.to(device)

        _, pc_emb, _ = point_model(pc)
        img_emb = image_model(img)

        pc_feats_list.append(pc_emb.cpu().numpy())
        img_feats_list.append(img_emb.cpu().numpy())
        labels_list.extend(label.numpy())

    pc_feats  = np.vstack(pc_feats_list)
    img_feats = np.vstack(img_feats_list)
    labels    = np.array(labels_list)
    return pc_feats, img_feats, labels

def main(args):
    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Dataset + DataLoader
    transform = get_default_transform()
    dataset = ObjectFolder2Dataset(
        root_dir    = args.data_path,
        transform   = transform,
        return_label= True
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4
    )

    # Model instantiation
    point_model = DGCNN(args).to(device)
    from torchvision.models import resnet18
    image_model = ResNet(resnet18(pretrained=True), feat_dim=256).to(device)

    # Load checkpoints
    ckpt_dir = os.path.join("checkpoints", args.exp_name, "models")
    point_ckpt = os.path.join(ckpt_dir, args.point_model_path)
    image_ckpt = os.path.join(ckpt_dir, args.image_model_path)

    point_model.load_state_dict(torch.load(point_ckpt, map_location=device))
    image_model.load_state_dict(torch.load(image_ckpt, map_location=device))

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract embeddings
    print("Extracting embeddings...")
    pc_feats, img_feats, labels = extract_features(dataloader, point_model, image_model, device)

    # Save raw embeddings and labels
    np.save(os.path.join(args.output_dir, "pc_embeddings.npy"),  pc_feats)
    np.save(os.path.join(args.output_dir, "img_embeddings.npy"), img_feats)
    np.save(os.path.join(args.output_dir, "labels.npy"),         labels)

    # Train/test split for SVM evaluation
    print("Evaluating with linear SVM on point cloud embeddings...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        pc_feats, labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels
    )
    clf_pc = SVC(kernel='linear', C=1.0)
    clf_pc.fit(X_tr, y_tr)
    y_pred = clf_pc.predict(X_te)
    acc_pc = accuracy_score(y_te, y_pred)
    print(f"SVM Point Cloud Accuracy: {acc_pc:.4f}")

    print("Evaluating with linear SVM on image embeddings...")
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        img_feats, labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels
    )
    clf_img = SVC(kernel='linear', C=1.0)
    clf_img.fit(X_tr2, y_tr2)
    y_pred2 = clf_img.predict(X_te2)
    acc_img = accuracy_score(y_te2, y_pred2)
    print(f"SVM Image Accuracy:      {acc_img:.4f}")

    # Save SVM results
    with open(os.path.join(args.output_dir, "svm_results.txt"), "w") as f:
        f.write(f"SVM Point Cloud Accuracy: {acc_pc:.4f}\n")
        f.write(f"SVM Image Accuracy:       {acc_img:.4f}\n")

    # t-SNE visualization (point clouds)
    if args.tsne_plot:
        print("Generating t-SNE plot...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(pc_feats)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, cmap='tab20', s=10
        )
        plt.legend(
            *scatter.legend_elements(),
            title="Classes", loc="best", fontsize=6
        )
        plt.title("t-SNE of Point Cloud Embeddings")
        plt.savefig(os.path.join(args.output_dir, "tsne_pointclouds.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',            type=str, required=True,
                        help="Experiment name (matches train.py --exp_name)")
    parser.add_argument('--data_path',           type=str, required=True,
                        help="Path to processed ObjectFolder2 data")
    parser.add_argument('--point_model_path',    type=str, required=True,
                        help="Name of point model checkpoint (e.g. best_point_model.pth)")
    parser.add_argument('--image_model_path',    type=str, required=True,
                        help="Name of image model checkpoint (e.g. best_img_model.pth)")
    parser.add_argument('--batch_size',          type=int,   default=32)
    parser.add_argument('--cuda',                action='store_true')
    parser.add_argument('--output_dir',          type=str,   default='results')
    parser.add_argument('--tsne_plot',           action='store_true',
                        help="Whether to generate t-SNE visualization")
    # these args are required to initialize DGCNN
    parser.add_argument('--emb_dims',            type=int,   default=1024)
    parser.add_argument('--k',                   type=int,   default=20)
    parser.add_argument('--dropout',             type=float, default=0.5)
    args = parser.parse_args()

    main(args)
