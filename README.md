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

```yml
---

🔵 **Explanation:**
- First code block = your Python code (`python`)
- Then write a heading like "Output:"
- Second code block = your result/output (`bash` or no language)

---
