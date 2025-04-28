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
Found 561 files in Malignant cases
Sample files: ['/content/drive/MyDrive/Dataset-Lungcancer....
Found 416 files in Normal cases
Sample files: ['/content/drive/MyDrive/Dataset-Lungcancer....

---




