import os
import glob
import subprocess

input_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/frames"
output_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/binary_masks"

os.makedirs(output_dir, exist_ok=True)

image_extensions = ["*.jpg", "*.png", "*.jpeg"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

print(f"Found {len(image_paths)} images.")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    mask_filename = f"mask_{base_filename}.npy"
    mask_path = os.path.join(output_dir, mask_filename)
    
    # Skip if mask already exists
    if os.path.exists(mask_path):
        print(f"⏭️ Skipping {filename} — output already exists.")
        continue

    print(f"➡️ Processing: {img_path}")
    result = subprocess.run([
        "python3", "text2mask.py",
        "--input", img_path,
        "--binary_mask", mask_path
    ])

    if result.returncode != 0:
        print(f"❌ Failed on: {img_path}")
    else:
        print(f"✅ Successfully created mask for: {filename}")