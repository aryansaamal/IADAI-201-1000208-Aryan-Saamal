import os
import cv2
import albumentations as A
from tqdm import tqdm

# === PATHS ===
# Path to your original dataset
input_dir = r"E:\XII IBCP\AI\ML SA\Dataset"      # 🔁 Replace with your actual dataset path
# Path where processed images will be saved
output_dir = r"C:\path\to\Processed_Dataset"  # 🔁 Replace if needed
img_size = 224  # Target image size (224x224)

# === AUGMENTATION PIPELINE ===
transform = A.Compose([
    A.Resize(img_size, img_size),  # Ensure uniform size
    A.HorizontalFlip(p=0.5),       # Flip horizontally (50% chance)
    A.VerticalFlip(p=0.3),         # Flip vertically (30% chance)
    A.Rotate(limit=30, p=0.5),     # Rotate within ±30 degrees
    A.RandomBrightnessContrast(p=0.5),  # Randomly change brightness/contrast
    A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5)  # Zoom-like crop
])

# === CREATE OUTPUT FOLDERS ===
for category in os.listdir(input_dir):
    category_path = os.path.join(output_dir, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

# === PROCESS EACH IMAGE ===
for category in os.listdir(input_dir):
    input_category_path = os.path.join(input_dir, category)
    output_category_path = os.path.join(output_dir, category)

    for img_name in tqdm(os.listdir(input_category_path), desc=f"Processing {category}"):
        try:
            img_path = os.path.join(input_category_path, img_name)
            image = cv2.imread(img_path)

            # Skip invalid/unreadable files
            if image is None:
                continue

            # Apply transformations
            transformed = transform(image=image)
            processed_img = transformed["image"]

            # Save preprocessed image
            output_img_path = os.path.join(output_category_path, img_name)
            cv2.imwrite(output_img_path, processed_img)

        except Exception as e:
            print(f"Error processing {img_name} in {category}: {e}")

print("\n✅ Preprocessing complete! All resized and augmented images are saved in:", output_dir)
