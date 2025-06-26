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
    if hasa