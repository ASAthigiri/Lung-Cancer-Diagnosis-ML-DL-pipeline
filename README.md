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

```python
import glob

image_files = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
    image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))

print(f"🖼️ Found {len(image_files)} images.")
```

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

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load extracted features and labels
features = np.load("/content/drive/MyDrive/Processed_Lung_Cancer_Data/EfficientNetB7_features.npy")
labels = np.load("/content/drive/MyDrive/Processed_Lung_Cancer_Data/EfficientNetB7_labels.npy")

# Split into train & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict on test set
y_pred = classifier.predict(X_test)

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Feature Extraction Accuracy using EfficientNetB7: {accuracy * 100:.2f}%")

# Detailed Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Bengin cases", "Malignant cases", "Normal cases"]))
```

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
categories = ["Bengin cases", "Malignant cases", "Normal cases"]
precision = [1.00, 0.97, 0.79]
recall = [0.24, 1.00, 0.98]
f1_score = [0.39, 0.99, 0.87]
accuracy = 89.09  # Feature Extraction Accuracy

# -------- Plot 1: Accuracy Bar Chart --------
plt.figure(figsize=(6, 4))
plt.bar(['EfficientNetB7 Feature Extraction'], [accuracy], color='skyblue')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Feature Extraction Accuracy')
plt.show()

# -------- Plot 2: Grouped Bar Chart for Precision, Recall, F1-score --------
x = np.arange(len(categories))  # Label locations
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, precision, width, label='Precision', color='blue')
ax.bar(x, recall, width, label='Recall', color='orange')
ax.bar(x + width, f1_score, width, label='F1-Score', color='green')

ax.set_xlabel('Categories')
ax.set_ylabel('Scores')
ax.set_title('Classification Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.ylim(0, 1.1)
plt.show()

# -------- Plot 3: Line Chart for Trends --------
plt.figure(figsize=(8, 5))
plt.plot(categories, precision, marker='o', linestyle='-', label='Precision', color='blue')
plt.plot(categories, recall, marker='o', linestyle='-', label='Recall', color='orange')
plt.plot(categories, f1_score, marker='o', linestyle='-', label='F1-Score', color='green')

plt.xlabel('Categories')
plt.ylabel('Scores')
plt.title('Trends of Classification Metrics')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True)
plt.show()

# -------- Plot 4: Heatmap of Classification Report --------
df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score
}, index=categories)

plt.figure(figsize=(7, 5))
sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()
```
## Dataset training - LightGBM

```python
import os
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Define paths for extracted features & labels
feature_data_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data/EfficientNetB7_features.npy"
label_data_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data/lung_cancer_labels.npy"

#Check if dataset files exist
if not os.path.exists(feature_data_path) or not os.path.exists(label_data_path):
    raise FileNotFoundError("Feature or Label file is missing! Please run feature extraction first.")

#Load extracted features & labels
features = np.load(feature_data_path)
labels = np.load(label_data_path)

#Ensure Labels are 1D
print(f"Original Labels Shape: {labels.shape}")
labels = labels.reshape(-1)  # Flatten to 1D if needed
print(f"Reshaped Labels Shape: {labels.shape}")

#Print unique labels to check for class imbalance
unique_labels, label_counts = np.unique(labels, return_counts=True)
print(f"Unique Labels: {unique_labels}")
print(f"Label Counts: {label_counts}")

#Split dataset into Train & Test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

#Check dataset sizes after splitting
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Train labels size: {y_train.shape[0]}, Test labels size: {y_test.shape[0]}")

#Train LightGBM Classifier with Optimized Parameters
lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="multiclass",
    num_class=len(np.unique(labels)),  # Auto-detect number of classes
    metric="multi_logloss",
    num_leaves=128,
    learning_rate=0.02,
    n_estimators=500,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.2,
    class_weight="balanced",  # Helps with class imbalance
    random_state=42
)

#Train Model
print(" Training LightGBM Model...")
lgb_model.fit(X_train, y_train)

#Make Predictions
y_train_pred = lgb_model.predict(X_train)
y_test_pred = lgb_model.predict(X_test)

#Ensure Predictions Shape Matches Test Labels
assert y_test_pred.shape[0] == y_test.shape[0], f"Mismatch in predictions: y_test={y_test.shape[0]}, y_test_pred={y_test_pred.shape[0]}"

#Evaluate Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

#Save the Trained Model
model_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data/Lung_Cancer_LGBM_Model.pkl"
joblib.dump(lgb_model, model_path)
print(f"Model saved at {model_path}!")

#Load the model later for predictions
loaded_model = joblib.load(model_path)
y_pred = loaded_model.predict(X_test)

#Feature Importance Visualization
plt.figure(figsize=(12, 6))
importances = lgb_model.feature_importances_
plt.bar(range(len(importances)), importances, color="teal")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in LightGBM")
plt.show()
```
## Model Evaluation & Accuracy Metrics
```python
import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

#Load trained model
lgb_model = joblib.load("/content/drive/MyDrive/Lung_Cancer_LGBM_Model.pkl")      -- test image

#Load EfficientNet-B7 for feature extraction
base_model = EfficientNetB7(weights="imagenet", include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

#Categories
categories = ["Benign", "Malignant", "Normal"]
img_size = 600  # EfficientNetB7 input size

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    #Extract feature vector
    feature_vector = feature_extractor.predict(img)
    return feature_vector.flatten()

def predict_lung_cancer_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(" Error: Folder not found!")
        return

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            features = extract_features(image_path)
            if features is None:
                print(f" Error: Unable to process {filename}")
                continue

            prediction = lgb_model.predict([features])[0]
            predicted_label = categories[prediction]

            print(f" Image: {filename} | Prediction: {predicted_label}")

#Test with a folder containing images
test_folder = "/content/drive/MyDrive/Test Case"
predict_lung_cancer_from_folder(test_folder)
```

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Predict test set
y_pred = lgb_model.predict(X_test)

#Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n LightGBM Model Accuracy: {accuracy * 100:.2f}%")

#Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

#Confusion Matrix
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

## Highlight Cancer-Affected Regions (Grad-CAM)

```python
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow.keras.backend as K
import joblib

#  Load the Pre-trained LightGBM Model
lgb_model_path = "/content/drive/MyDrive/Processed_Lung_Cancer_Data/Lung_Cancer_LGBM_Model.pkl"
lgb_model = joblib.load(lgb_model_path)
print(f"Loaded Model from {lgb_model_path}")

# Load Pre-trained EfficientNetB7 for Feature Extraction
base_model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(600, 600, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
print(" EfficientNetB7 Loaded Successfully!")

# Grad-CAM Function (Improved)
def grad_cam(image_path):
    # Load & Preprocess Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600, 600))  # EfficientNetB7 requires 600x600
    img = preprocess_input(img)
    img_tensor = np.expand_dims(img, axis=0)

    # Convert to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Feature Extraction for LightGBM
    extracted_features = feature_extractor.predict(img_tensor).flatten().reshape(1, -1)

    # Check Model Feature Dimension
    if extracted_features.shape[1] != lgb_model.booster_.num_feature():
        raise ValueError(
            f"Feature mismatch: Model expects {lgb_model.booster_.num_feature()} features, but got {extracted_features.shape[1]}"
        )

    # Make Prediction using LightGBM
    prediction_probabilities = lgb_model.predict_proba(extracted_features)[0]
    predicted_class = np.argmax(prediction_probabilities)
    confidence = prediction_probabilities[predicted_class] * 100  # Convert to percentage

    # Grad-CAM Implementation
    last_conv_layer = base_model.get_layer("block7a_project_conv")  # More relevant for lung region
    grad_model = Model([base_model.input], [last_conv_layer.output, base_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_score = predictions[:, predicted_class]  # Focus on predicted class activation

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = np.mean(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Ensure only positive activations
    heatmap /= np.max(heatmap)  # Normalize

    # Smooth Heatmap using Gaussian Blur
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=2)

    # Overlay Heatmap on Original Image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Display Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Predicted: Class {predicted_class} ({confidence:.2f}%)")

    plt.show()

# Test on a Sample Image
test_image_path = "/content/drive/MyDrive/Test Case/Testcase1.jpg"
grad_cam(test_image_path)
```
## Performance Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

#Given Confusion Matrix
cm = np.array([[23, 0, 1],
               [0, 113, 0],
               [1, 0, 82]])

#Class labels
classes = ['Class 0', 'Class 1', 'Class 2']

#Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
```

```python
#Classification Report
y_true = np.array([0]*24 + [1]*113 + [2]*83)  # Ground Truth
y_pred = np.array([0]*23 + [2] + [1]*113 + [2]*82 + [0])  # Predicted

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))
```
```python
# ROC Curve (One-vs-All Approach for Multi-Class)
plt.figure(figsize=(6,5))
for i in range(3):
    y_true_binary = (y_true == i).astype(int)
    y_score = (y_pred == i).astype(int)

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```
```python
# Precision-Recall Curve
plt.figure(figsize=(6,5))
for i in range(3):
    y_true_binary = (y_true == i).astype(int)
    y_score = (y_pred == i).astype(int)

    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
    plt.plot(recall, precision, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
```
## Log Loss & ROC Curve data

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, log_loss

# Compute Log Loss
logloss = log_loss(y_test, lgb_model.predict_proba(X_test))

# Compute ROC Curve and AUC
y_probs = lgb_model.predict_proba(X_test)
if y_probs.shape[1] == 2:  # Binary classification
    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
else:  # Multi-class classification (One-vs-Rest)
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Create figure
plt.figure(figsize=(12, 5))

# Plot Log Loss as a single bar
plt.subplot(1, 2, 1)
sns.barplot(x=["Log Loss"], y=[logloss], palette="coolwarm", width=0.4)
plt.ylabel("Log Loss")
plt.ylim(0, logloss + 0.1)  # Ensures the bar is visible
plt.title(f"📉 Log Loss: {logloss:.4f}")

# Plot ROC Curve
plt.subplot(1, 2, 2)
if y_probs.shape[1] == 2:  # Binary classification
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
else:  # Multi-class classification
    for i in range(y_probs.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.4f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="lower right")

# Show the plots
plt.tight_layout()
plt.show()
```
## Cross-Validation Mean Accuracy & Cross-Validation Std Dev
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring="accuracy")
print(f" Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
print(f" Cross-Validation Std Dev: {cv_scores.std():.4f}")
```
## Accuracy and Model performance camparison

```python
import matplotlib.pyplot as plt
import numpy as np

# Model names and their accuracies
models = ['CNN', 'VGG-16', 'ResNet-50', 'VGG-19', 'LightGBM']
accuracies = [94.88, 94.07, 95.40, 92.59, 99.09]
misclass_rates = [100 - acc for acc in accuracies]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Accuracy bars
acc_bars = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='skyblue')
# Misclassification bars
misclass_bars = ax.bar(x + width/2, misclass_rates, width, label='Misclassification Rate (%)', color='lightcoral')

# Label formatting
ax.set_ylabel('Percentage (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(acc_bars)
add_labels(misclass_bars)

plt.tight_layout()
plt.show()
```
```python
import matplotlib.pyplot as plt
import numpy as np

# Models and their performance metrics
models = ['CNN', 'VGG16', 'ResNet50', 'VGG19', 'LightGBM']
specificity = [0.93, 0.96, 0.97, 0.92, 0.99]
precision = [0.84, 0.81, 0.89, 0.79, 0.98]
recall = [0.82, 0.79, 0.88, 0.76, 0.97]
accuracy = [0.9488, 0.9407, 0.9540, 0.9259, 0.9909]

# Bar width and positions
bar_width = 0.2
index = np.arange(len(models))

# Plotting the grouped bar chart
plt.figure(figsize=(10, 5))
plt.bar(index, specificity, bar_width, label='Specificity', color='royalblue')
plt.bar(index + bar_width, precision, bar_width, label='Precision', color='firebrick')
plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='forestgreen')
plt.bar(index + 3 * bar_width, accuracy, bar_width, label='Accuracy', color='mediumpurple')

# Axis settings
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Performance Metrics by Model')
plt.xticks(index + 1.5 * bar_width, models)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5)

plt.show()
```



