# CrossPoint-OF2: Self-Supervised Representation Learning on ObjectFolder 2.0
This project adapts the CrossPoint framework for self-supervised learning on the **ObjectFolder 2.0** dataset, leveraging 3D points clouds and their corresponding 2D views to learn multi-modal representations.

---

## Objective
Adapt the **CrossPoint** self-supervised learning framework to train on the **ObjectFolder 2.0** dataset, using 3D point clouds and their associated 2D RGB views, and evaluate the learned representations using a **linear classifier**.

---

## Repository Structure
CrossPoint_OF2/
- datasets/
- - ObjectFolder2.py # Custom dataset loader for ObjectFolder2
- models/
- - dgcnn.py # DGCNN and ResNet architectures
- scripts/
- - train_trial.slurm # SLURM job script for training
- - evaluate_trial.slurm # SLURM job script for evaluation
- train.py # Self-supervised training script
- evaluate.py # Linear classifier training + t-SNE visualization
- util.py # Utility functions (logging, metrics)
- requirements.txt # Required packages
- README.md # Project documentation
- checkpoints/ # Saved models and logs

---

## Installation & Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
torch>=1.12
torchvision
numpy
pillow
scikit_learn
matplotlib
tqdm
wandb==0.15.9
lightly==1.3.0
```
> Make sure you also have `Rust` and `cargo` installed to avoid errors during installation of `lightly`.

---

## How to Run
### Train the model (self-supervised)
```bash
python train.py \
  --data_path /path/to/processed/ObjectFolder2 \
  --exp_name my_experiment \
  --epochs 100 \
  --batch_size 32 \
  --cuda
```

### Evaluate with Linear SVM & t-SNE
```bash
python evaluate.py \
  --data_path /path/to/processed/ObjectFolder2 \
  --exp_name my_experiment \
  --cuda
```

---

## Outputs
- Trained representations (point cloud & image encoders)
- Linear classifier accuracy (on ObjectFolder2)
- t-SNE visualizations (colored by object class)

```Linear classifier accuracy (SVM): %```

<p align="center">
    <em>(Pending...)</em>
    <img src="docs/tsne_example.png" alt="t-SNE visualization of embeddings" width="500"/>
</p>

---

## Dataset: ObjectFolder 2.0
- Contains ~1000 real-world 3D objects.
- Modalities: RGB images, 3D point clouds, audio, and touch.
- We use the **point cloud** and **RGB** modalities only.
- Dataset must be preprocessed into `.npy`for point clouds and `.png`for images.

---

## Credits
- DGCNN implementation from: https://github.com/WangYueFt/dgcnn
- Original CrossPoint paper: https://arxiv.org/pdf/2203.00680
- CrossPoint repo: https://github.com/MohamedAfham/CrossPoint
- ObjectFolder 2.0 dataset: https://objectfolder.stanford.edu

---

## Authors
- Eunice Saraí Castillo Turrubiartes
- Ghazal Rouhafzay