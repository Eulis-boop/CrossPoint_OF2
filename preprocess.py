import os
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

def is_image(filename):
    """Check if a file is an image based on its extension."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def preprocess_objectfolder(
    root_dir,
    output_pc_dir,
    output_img_dir,
    n_points=2048,
    image_size=(224, 224)
):
    """Convert .obj files to point clouds and extract RGB images."""
    os.makedirs(output_pc_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    skipped = 0
    total = 0

    print(f"Starting preprocessing from: {root_dir}\n")

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("ObjectFolder"):
            for obj_id in os.listdir(folder_path):
                obj_path = os.path.join(folder_path, obj_id)
                model_path = os.path.join(obj_path, "model.obj")
                textures_path = os.path.join(obj_path, "textures")
        
                #Skip if model or textures folder is missing
                if not os.path.exists(model_path) or not os.path.isdir(textures_path):
                    skipped += 1
                    continue
        
                try:
                    #Convert .obj to point cloud and save
                    mesh = trimesh.load(model_path, force='mesh')
                    points = mesh.sample(n_points)
                    np.save(os.path.join(output_pc_dir, f"{obj_id}.npy"), points.astype(np.float32))
        
                    #Load the first available image in textures folder
                    texture_files = [f for f in os.listdir(textures_path) if is_image(f)]
                    if not texture_files:
                        skipped += 1
                        continue
        
                    img_path = os.path.join(textures_path, texture_files[0])
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(image_size)
                    img.save(os.path.join(output_img_dir, f"{obj_id}.png"))
        
                    total += 1
                except Exception as e:
                    print(f"[Warning] Failed processing object {obj_id}: {e}")
                    skipped += 1

    print(f"\n Preprocessing complete.")
    print(f" Successfully processed objects: {total}")
    print(f" Skipped objects (missing files or errors): {skipped}")

if __name__ == "__main__":
    #Paths (update with the actual path)
    ROOT_DIR = "/scratch/sarai/ObjectFolder_data"
    OUTPUT_DIR = os.path.join(ROOT_DIR, "processed")
    PC_DIR = os.path.join(OUTPUT_DIR, "pointclouds")
    IMG_DIR = os.path.join(OUTPUT_DIR, "RGB_images")

    print("Preprocessing ObjectFolder 2.0 dataset...\n")
    preprocess_objectfolder(
        root_dir=ROOT_DIR,
        output_pc_dir=PC_DIR,
        output_img_dir=IMG_DIR
    )
