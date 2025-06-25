import os
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

def preprocess_point_clouds(obj_root_dir, output_dir, n_points=2048):
	"""Convert .obj files to .npy point clouds."""
	os.makedirs(output_dir, exist_ok=True)

	for obj_folder in tqdm(os.listdir(obj_root_dir)):
		if obj_folder.startswith("ObjectFolder"):
			for obj_id in os.listdir(os.path.join(obj_root_dir, obj_folder)):
				obj_path = os.path.join(obj_root_dir, obj_folder, obj_id, "model.obj")
				if os.path.exists(obj_path):
					mesh = trimesh.load(obj_path):
					points = mesh.sample(n_points) #shape: (2048, 3)
					np.save(
						os.path.join(output_dir, f"{obj_id}.npy"),
						points.astype(np.float32)

def preprocess_images(img_root_dir, output_dir, size=(224, 224)):
	"""Resize RGB images to 224x224 and save."""
	os.makedirs(output_dir, exist_ok=True)

	for obj_folder in tqdm(os.listdir(img_root_dir)):
		if obj_folder.startswith("ObjectFolder"):
			for obj_id in os.listdir(os.path.join(img_root_dir, obj_folder)):
				img_path = os.path.join(
					img_root_dir, obj_folder, obj_id,
					"textures", ".png"
				)
				if os.path.exists(img_path):
					img = Image.open(img_path).convert("RGB")
					img = img.resize(size)
					img.save(os.path.join(output_dir, f"{obj_id}.png"))

if __name__ == "__main__":
	#Paths
	OBJECTFOLDER_DATA_DIR = "/path/to/ObjectFolder_data"  # Parent of ObjectFolder1-100/, etc.
	OUTPUT_DIR = os.path.join(OBJECTFOLDER_DATA_DIR, "processed")

	# Step 1: Preprocess point clouds
	print("Preprocessing point clouds...")
	preprocess_point_clouds(OBJECTFOLDER_DATA_DIR, os.path.join(OUTPUT_DIR, "pointclouds"))

	# Step 2: Preprocess images
	print("Preprocessing images...")
	preprocess_images(OBJECTFOLDER_DATA_DIR, os.path.join(OUTPUT_DIR, "RGB_images"))

	print("Done! Preprocessed data saved to:", OUTPUT_DIR)
