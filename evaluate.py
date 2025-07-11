import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datasets.ObjectFolder2 import ObjectFolder2, get_default_transform
from models.dgcnn import DGCNN
from torch.utils.data import DataLoader

def evaluate(args):
    #1. Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN(args).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{args.exp_name}/models/best_model.pth"))
    model.eval()

    #2. Load data
    transform = get_default_transform()
    val_dataset = ObjectFolder2(
        root_dir = args.data_path,
        transform = transform,
        split = 'val' #!
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #3. Extract features
    features, labels = [], []
    with torch.no_grad():
        for point_cloud, img, label in val_loader:
            point_cloud = point_cloud.to(device).transpose(2, 1)
            _, _, feat = model(point_cloud)
            features.append(feat.cpu().numpy())
            labels.append(label.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    #4. Linear Evaluation
    train_size = int(0.8 * len(features))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features[:train_size], labels[:train_size])
    preds = knn.predict(features[train_size:])
    acc = accuracy_score(labels[train_size:], preds)
    print(f"KNN Accuracy: {acc:.4f}")

    #5. t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(features[:500]) #Subsample for speed

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels[:500], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE of Point Cloud Embeddings")
    plt.savefig(f"checkpoints/{args.exp_name}/tsne.png")
    plt.close()

    #6. Cross-modal Retrieval (if we have paired data)
    if hasattr(args, 'cross_modal'):
		evaluate_cross_modal(model, val_loader, device)

def evaluate_cross_modal(model, dataloader, device):
	"""Evaluate point cloud -> image retrieval"""
	pc_features, img_features = [], []

    #Load the image model (ResNet)
    img_model = ResNet(resnet50(), feat_dim=2048).to(device)
    img_model.load_state_dict(torch.load(f"checkpoints/{args.exp_name}/models/img_model_best.pth"))
    img_model.eval()
    
    with torch.no_grad():
		for point_cloud, img, _ in dataloader:
            #
			point_cloud = point_cloud.to(device).transpose(2, 1)
			_, _, pc_feat = model(point_cloud)
			pc_features.append(pc_feat.cpu())

			#
            img_feat = img_model(img.to(device))
    		img_features.append(img_feat.cpu())

    #Convert to tensors
    pc_features = torch.cat(pc_features)
    img_features = torch.cat(img_features)

    #Calculate recovery metrics
    def calculate_metrics(query_feats, target_feats, top_k=(1, 5, 10)):
        """Calculate Recall@k y Median Rank"""
        sim_matrix = torch.mm(query_feats, target_feats.t()) #Similarity matrix

        results = {}
        for k in top_k:
            -, indices = sim_matrix.topk(k, dim=1)
            correct = torch.zeros(len(query_feats))
            for i in range(len(query_feats)):
                correct[i] = i in indices[i]
            results[f"Recall@{k}"] = correct.mean().item()

        #Calculate Median Rank
        -, indices = sim_matrix.sort(descending=True)
        ranks = torch.where(indices == torch.arange(len(query_feats)).unsqueeze(1))[1]
        results["MedianRank"] = torch.median(ranks.float()).item()

        return results

    #Evaluate both directions (PC→Img and Img→PC)
    print("\nCross Recovery Metrics:")

    #PC → Image
    metrics = calculate_metrics(pc_features, img_features)
    print("Point Cloud → Image:")
    for k, v in metrics.item():
        print(f"{k}: {v:.4f}")

    #Image → PC
    metrics = calculate_metrics(img_features, pc_features)
    print("\nImage → Point Cloud:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, required=True)
	parser.add_argument('--data_path', type=str, required=True)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--k', type=int, default=20)
	parser.add_argument('--emb_dims', type=int, default=1024)
	args = parser.parse_args()

	evaluate(args)
