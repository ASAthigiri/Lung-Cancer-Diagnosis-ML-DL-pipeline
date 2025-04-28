# Lung-Cancer-Diagnosis-ML-DL-pipeline
Hybrid approach to find the lung cancer using CT and MRI Scan

## DataSet Used : The IQ-OTHNCCD lung cancer dataset
Link: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset?resource=download

Mount drive:
```from google.colab import drive
drive.mount('/content/drive')```

Load Dataset:
```import os
dataset_path = "/content/drive/MyDrive/Dataset-Lungcancer/The IQ-OTHNCCD lung cancer dataset"
if os.path.exists(dataset_path):
    print("Dataset path exists.")
else:
    print("Dataset path NOT found. Check the path.")
