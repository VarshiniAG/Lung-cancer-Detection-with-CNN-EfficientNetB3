#🫁 Lung Cancer Detection using CNN with EfficientNetB3

#A Deep Learning approach for early lung cancer diagnosis with Streamlit deployment via ngrok

⭐ Introduction

Lung cancer is one of the most fatal diseases worldwide, and early detection dramatically improves patient survival rates. This project develops a deep learning model using EfficientNetB3, a highly optimized CNN architecture known for achieving maximum accuracy with minimal computational cost.

EfficientNetB3 provides 40–50% fewer parameters compared to traditional CNNs while delivering state-of-the-art medical imaging performance, making it ideal for lung cancer classification tasks.

A Streamlit web application is built to allow real-time predictions, and since Google Colab does not allow external ports, the app is deployed using ngrok, which exposes the Streamlit server through a secure public URL.

🚀 Why EfficientNetB3? (Strong Justification)

EfficientNetB3 is chosen because:

✔ Better accuracy vs traditional CNNs

It scales depth, width, and resolution using compound scaling, improving diagnostic precision.

✔ Lightweight and fast

It achieves high performance even on limited GPU environments (like Colab).

✔ Best suited for medical imaging

EfficientNet has been widely used in radiology tasks including CT scans, X-rays, MRI detection, etc.

✔ Superior feature extraction

It captures fine lesions and abnormalities in lung tissues better than simple CNNs.

Because of these advantages, EfficientNetB3 significantly improves cancer detection performance over classical CNN models.

🔬 Methodology

Dataset loading & exploration

Image preprocessing (resize, normalize, augmentation)

Baseline CNN model training

EfficientNetB3 Transfer Learning

Load pretrained weights

Freeze convolution base

Add custom fully connected layers

Fine-tune upper layers

Training & model evaluation

Grad-CAM visualization

Streamlit app integration

ngrok deployment (to access Streamlit outside Colab)

🧠 EfficientNetB3 Architecture Overview

Based on MBConv blocks

Uses Swish activation (better than ReLU)

Employs compound coefficient scaling

Achieves an excellent balance of:

Accuracy

Speed

Parameter efficiency

Custom classification head added:

GlobalAveragePooling2D
Dropout(0.4)
Dense(256, activation='relu')
Dense(1, activation='sigmoid')  # for binary classes

📊 Evaluation Metrics

Accuracy

Loss curves

Confusion Matrix

Classification Report

ROC Curve & AUC

Grad-CAM Lung Lesion Heatmaps

🌐 Streamlit Deployment using ngrok (Google Colab)

Google Colab cannot expose web apps directly.
So we use ngrok to create a public link.

✅ Step 1: Install dependencies
!pip install streamlit pyngrok

✅ Step 2: Create Streamlit App
%%writefile app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("lung_model.h5")

st.title("🫁 Lung Cancer Detection - EfficientNetB3")

uploaded = st.file_uploader("Upload Lung CT Scan", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((300,300))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("⚠ Lung Cancer Detected")
    else:
        st.success("✔ Healthy Lung detected")

✅ Step 3: Start Streamlit in background
!streamlit run app.py &>/content/logs.txt &

✅ Step 4: Create ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8501)
public_url


You will get a link like:

https://1234abcd.ngrok.io


This is your public Streamlit app URL, accessible from any browser.

🏗 System Architecture
Dataset → Preprocessing → EfficientNetB3 → Model Training
         ↓
     Streamlit App
         ↓
     ngrok Tunnel
         ↓
 Users access the app publicly

🛠 Technologies Used

TensorFlow / Keras

EfficientNetB3

Python

NumPy, Pandas

Streamlit

Ngrok

Google Colab

▶️ How to Run
1️⃣ Clone repo
git clone https://github.com/yourname/LungCancerEfficientNet.git
cd LungCancerEfficientNet

2️⃣ Train model in Colab

Open the .ipynb file and run all cells.

3️⃣ Run Streamlit app

Use the ngrok steps above.

🔮 Future Enhancements

Convert to ONNX or TFLite for mobile deployment

Deploy on HuggingFace Spaces

Use 3D CT scans with 3D CNN

Add ensemble of EfficientNet variants

Real-time inference dashboard

🎉 Conclusion

Using EfficientNetB3, this project achieves high accuracy in lung cancer detection while remaining computationally efficient. Its Streamlit + ngrok deployment makes it easy to use anywhere — no local installation required.

This solution brings AI-powered cancer detection closer to real-world medical applications.
