# Lung-Cancer-Diagnosis-ML-DL-pipeline
Hybrid approach to find the lung cancer using CT and MRI Scan

## DataSet Used : The IQ-OTHNCCD lung cancer dataset
Link: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset?resource=download

### Mount drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
### Load Dataset:
```python
import os
dataset_path = "/content/drive/MyDrive/Dataset-Lungcancer/The IQ-OTHNCCD lung cancer dataset"
if os.path.exists(dataset_path):
    print("Dataset path exists.")
else:
    print("Dataset path NOT found. Check the path.")
```
## Preprocessing
```python
import os
import numpy as np
import cv2
import glob
import concurrent.futures
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects, disk

# Define Paths
dataset_path = "/content/drive/MyDrive/Dataset-Lungcancer/The IQ-OTHNCCD lung cancer dataset"
save_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data"
os.makedirs(save_path, exist_ok=True)
```

```python
categories = ["Bengin cases", "Malignant cases", "Normal cases"]
img_size = 256

# Image Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Denoising
    denoised_img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Segmentation
    thresh = threshold_otsu(denoised_img)
    binary_mask = denoised_img > thresh
    binary_mask = binary_closing(binary_mask, disk(2))
    binary_mask = remove_small_objects(binary_mask, min_size=500)
    segmented_img = denoised_img * binary_mask

    # Resizing
    resized_img = cv2.resize(segmented_img, (img_size, img_size))
    return resized_img, None  # Return image and label
```

```python
# Load Data in Parallel
data, labels = [], []

def process_category(category, class_num):
    images, lbls = [], []
    path = os.path.join(dataset_path, category)
    image_files = glob.glob(os.path.join(path, "*.png"))  # Adjust extension if needed

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(preprocess_image, image_files)

    for img, lbl in results:
        if img is not None:
            images.append(img)
            lbls.append(class_num)

    return images, lbls
```

```python
# Run Parallel Processing
for idx, category in enumerate(categories):
    cat_data, cat_labels = process_category(category, idx)
    data.extend(cat_data)
    labels.extend(cat_labels)

# Convert to NumPy Arrays
data = np.array(data, dtype=np.float32).reshape(-1, img_size, img_size, 1) / 255.0
labels = np.array(labels, dtype=np.int32)

# Save Processed Data to Drive
np.save(os.path.join(save_path, "lung_cancer_data.npy"), data)
np.save(os.path.join(save_path, "lung_cancer_labels.npy"), labels)

print("Dataset Loaded and Saved Successfully!")
```


---

🔵 **Result:**
- Dataset Loaded and Saved Successfully!

---

## Preprocessing: Contrast Enhancement and Data Augmentation

```python
import os
import numpy as np
import cv2
import glob
import concurrent.futures
from skimage.exposure import equalize_hist
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import img_as_ubyte

def enhance_contrast(image):
    """Apply histogram equalization for contrast enhancement."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def augment_image(image):
    """Apply data augmentation: rotation, flipping, brightness adjustment."""
    augmented_images = []

    # Rotate images by 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        augmented_images.append(rotate(image, angle, mode='wrap'))

    # Flip images horizontally and vertically
    augmented_images.append(np.fliplr(image))
    augmented_images.append(np.flipud(image))

    # Adjust brightness
    brightness_factor = 1.5  # Increase brightness
    bright_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    augmented_images.append(bright_image)

    return [img_as_ubyte(img) for img in augmented_images]

def process_image(image):
    """Enhance contrast and apply augmentation."""
    enhanced_image = enhance_contrast(image)
    augmented_images = augment_image(enhanced_image)
    return [enhanced_image] + augmented_images

def main():
    save_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data"
    os.makedirs(save_path, exist_ok=True)

    # Load saved dataset
    data = np.load(os.path.join(save_path, "lung_cancer_data.npy"))
    labels = np.load(os.path.join(save_path, "lung_cancer_labels.npy"))

    processed_data = []
    processed_labels = []

    # Get image shape from the first image
    img_shape = data.shape[1:3]

    for img, label in zip(data, labels):
        img = (img * 255).astype(np.uint8)  # Convert back to 8-bit image
        processed_images = process_image(img)
        processed_data.extend(processed_images)
        processed_labels.extend([label] * len(processed_images))

    # Convert to NumPy arrays and save
    processed_data = np.array(processed_data, dtype=np.float32).reshape(-1, img_shape[0], img_shape[1], 1) / 255.0
    processed_labels = np.array(processed_labels, dtype=np.int32)

    np.save(os.path.join(save_path, "enhanced_lung_cancer_data.npy"), processed_data)
    np.save(os.path.join(save_path, "enhanced_lung_cancer_labels.npy"), processed_labels)

    print("Contrast Enhancement and Data Augmentation Completed Successfully!")

if __name__ == "__main__":
    main()
```

---

🔵 **Result:**
- Contrast Enhancement and Data Augmentation Completed Successfully!

---

## Count the files in each category
```python
import os
import glob
# Define dataset path
dataset_path = "/content/drive/MyDrive/Dataset-Lungcancer/The IQ-OTHNCCD lung cancer dataset"

# Function to check files in a category
def check_images(category):
    category_path = os.path.join(dataset_path, category)
    image_files = glob.glob(os.path.join(category_path, "*.*"))  # Match any file type
    print(f"Found {len(image_files)} files in {category}")
    print("Sample files:", image_files[:5])  # Print first 5 file names
    return image_files

# Check all categories
benign_images = check_images("Bengin cases")
malignant_images = check_images("Malignant cases")
normal_images = check_images("Normal cases")
```

---

🔵 **Result:**
- Found 120 files in Bengin cases
Sample files: ['/content/drive/MyDrive/Dataset-Lungcancer....
- Found 561 files in Malignant cases
Sample files: ['/content/drive/MyDrive/Dataset-Lungcancer....
- Found 416 files in Normal cases
Sample files: ['/content/drive/MyDrive/Dataset-Lungcancer....

---

```python
import glob

image_files = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
    image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))

print(f"🖼️ Found {len(image_files)} images.")
```
---

🔵 **Result:**
- Found 1097 images.

---

## Feature Extraction using EfficientNet-B7
```python
import os
import cv2
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tqdm import tqdm  # For progress tracking

#Load Pretrained EfficientNet-B7
base_model = EfficientNetB7(weights="imagenet", include_top=False)
model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

#Define dataset path and categories
dataset_path = "/content/drive/MyDrive/Dataset-Lungcancer/The IQ-OTHNCCD lung cancer dataset"
categories = ["Bengin cases", "Malignant cases", "Normal cases"]
img_size = 600  # EfficientNetB7 input size

#Initialize storage for features and labels
features = []
labels = []

#Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load image
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (img_size, img_size))  # Resize to 600x600 (EfficientNet-B7 input size)
    img = preprocess_input(img)  # Normalize using EfficientNet preprocessing
    return img

#Extract features for each category
for label, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    image_files = glob.glob(os.path.join(category_path, "*.*"))  # Supports .png, .jpg, etc.

    print(f"Processing {len(image_files)} images from {category}...")

    for image_path in tqdm(image_files):
        img = preprocess_image(image_path)
        if img is not None:
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            feature_vector = model.predict(img)  # Extract features
            features.append(feature_vector.flatten())  # Store flattened feature vector
            labels.append(label)  # Store corresponding label

#Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print("Feature extraction completed!")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

plt.figure(figsize=(12, 6))
importances = lgb_model.feature_importances_
plt.bar(range(len(importances)), importances, color="teal")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in LightGBM")
plt.show()

#Save extracted features for later use
np.save("/content/drive/MyDrive/Processed_Lung_Cancer_Data/EfficientNetB7_features.npy", features)
np.save("/content/drive/MyDrive/Processed_Lung_Cancer_Data/EfficientNetB7_labels.npy", labels)
np.save("/content/drive/MyDrive/Processed_Lung_Cancer_Data/lung_cancer_labels.npy", labels)
```

---

🔵 **Result:**
- Processing 120 images from Bengin cases...
  0%|          | 0/120 [00:00<?, ?it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 14s 14s/step
  1%|          | 1/120 [00:20<40:42, 20.53s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step
  2%|▏         | 2/120 [00:20<17:00,  8.65s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 116ms/step
  2%|▎         | 3/120 [00:21<09:20,  4.79s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step
  3%|▎         | 4/120 [00:21<05:45,  2.98s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step
  4%|▍         | 5/120 [00:21<03:45,  1.96s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step
  5%|▌         | 6/120 [00:21<02:33,  1.35s/it]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step
  6%|▌         | 7/120 [00:21<01:49,  1.03it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step
  7%|▋         | 8/120 [00:21<01:19,  1.42it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 98ms/step
  8%|▊         | 9/120 [00:22<01:00,  1.83it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step
  8%|▊         | 10/120 [00:22<00:48,  2.27it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step
  9%|▉         | 11/120 [00:22<00:37,  2.87it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step
 10%|█         | 12/120 [00:22<00:30,  3.51it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step
 11%|█         | 13/120 [00:22<00:27,  3.88it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step
 12%|█▏        | 14/120 [00:22<00:25,  4.16it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step
 12%|█▎        | 15/120 [00:23<00:23,  4.38it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step
 13%|█▎        | 16/120 [00:23<00:22,  4.55it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 88ms/step
 14%|█▍        | 17/120 [00:23<00:21,  4.71it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step
 15%|█▌        | 18/120 [00:23<00:19,  5.33it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step
 16%|█▌        | 19/120 [00:23<00:19,  5.24it/s]1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step

---




