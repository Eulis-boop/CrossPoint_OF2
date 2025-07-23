import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ObjectFolder2(Dataset):
	def __init__(self, root_dir, transform=None, n_points=2048):
		"""
		Args:
		     root_dir (str): Path to processed data (contains pointclouds/ and RGB_images/).
		     transform (callable, optional): Transform to apply to RGB images.
		     n_points (int): Number of points in each point cloud (must match preprocess.py).
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.n_points = n_points
		self.samples = self._load_samples() #list of (point_cloud_path, image_path)

	def _load_samples(self):
		samples = []
		pc_dir = os.path.join(self.root_dir, "pointclouds") #preprocessed .npy files
		img_dir = os.path.join(self.root_dir, "RGB_images") #original or resized .png files

		#match point clouds to their corresponding images
		for pc_file in os.listdir(pc_dir):
			if pc_file.endswith(".npy"):
				obj_id = pc_file.split(".")[0]
				img_file = f"{obj_id}.png"
				img_path = os.path.join(img_dir, img_file)
				if os.path.exists(img_path):
					samples.append((
						os.path.join(pc_dir, pc_file),
						img_path
					))
		return samples

	def __getitem__(self, idx):
		#load point cloud (shape: [2048, 3])
		point_cloud = np.load(self.samples[idx][0])

		#load image and apply transforms
		img = Image.open(self.samples[idx][1]).convert("RGB")
		if self.transform:
			img = self.transform(img)

		return point_cloud, img

	def __len__(self):
		return len(self.samples)

#transforms for RGB images (same as CrossPoint)
def get_default_transform():
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

#example usage:
if __name__ == "__main__":
	transform = get_default_transform()
	dataset = ObjectFolder2(
		root_dir = "/path/to/ObjectFolder_data/processed",
		transform = transform
	)
	print(f"Dataset size: {len(dataset)}")
	pc, img = dataset[0]
	print(f"Point cloud shape: {pc.shape}")
	print(f"Image shape: {img.shape}")
