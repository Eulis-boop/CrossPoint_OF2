import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ObjectFolder2Dataset(Dataset):
	def __init__(self, root_dir, transform=None, n_points=2048, return_label=False):
		"""
		Args:
		     root_dir (str): Path to processed data (contains pointclouds/ and RGB_images/).
		     transform (callable, optional): Transform to apply to RGB images.
		     n_points (int): Number of points in each point cloud (must match preprocess.py).
             return_label (bool): Whether to return object labels (for classification/t-SNE).
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.n_points = n_points
        self.reutrn_label = return_label
		self.samples = self._load_samples() #list of (point_cloud_path, image_path, label)

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
					if self.return_label:
                        class_name = obj_id.split("_")[0]
                        samples.append((
    						os.path.join(pc_dir, pc_file),
    						img_path,
                            class_name
    					))
                    else:
                        samples.append((
                            os.path.join(pc_dir, pc_file),
                            img_path
                        ))
		return samples

	def __getitem__(self, idx):
		if self.return_label:
            pc_path, img_path, label = self.samples[idx]
        else:
            pc_path, img_path = self.samples[idx]

        #load point cloud (shape: [2048, 3])
		point_cloud = torch.from_numpy(np.load(pc_path)).float()

		#load and transform image
		img = Image.open(img_path).convert("RGB")
		if self.transform:
			img = self.transform(img)
            
        if self.return_label:
            return point_cloud, img, label
        else:
    		return point_cloud, img

	def __len__(self):
		return len(self.samples)

#Default image transform (same as CrossPoint)
def get_default_transform():
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

#Example usage:
if __name__ == "__main__":
	transform = get_default_transform()
	dataset = ObjectFolder2Dataset(
		root_dir = "/scratch/sarai/ObjectFolder_data/processed",
		transform = transform,
        return_label = True
	)
	print(f"Dataset size: {len(dataset)}")
	pc, img, label = dataset[0]
	print(f"Point cloud shape: {pc.shape}")
	print(f"Image shape: {img.shape}")
    print(f"Label: {label}")