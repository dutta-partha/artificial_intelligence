import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os
from glob import glob
import random

st.set_page_config(page_title="Liver Fibrosis Dashboard")

script_dir = os.path.dirname(__file__)

# Sidebar menu
section = st.sidebar.radio("Navigation", [
    "Project Overview",
    "Data Summary",
    "Training Metrics",
    "Evaluation Report",
    "Live Inference"
])

# Load cached training/evaluation stats
@st.cache_data

def load_training_stats():
    try:
        return pd.read_json(os.path.join(script_dir, "training_stats.json"))
    except:
        return None

@st.cache_data

def load_eval_report():
    try:
        with open(os.path.join(script_dir, "eval_report.json")) as f:
            return json.load(f)
    except:
        return None

training_df = load_training_stats()
eval_report = load_eval_report()

# Section: Project Overview
if section == "Project Overview":
    st.title("Liver Fibrosis Classification Dashboard")
    st.markdown("""
    This Streamlit app presents a complete pipeline for classifying liver fibrosis
    stages (F0 to F4) using ultrasound images.

    **Features:**
    - Data distribution and sample images
    - Training and validation performance tracking
    - Evaluation metrics and confusion matrix
    - Upload and test your own ultrasound image
    """)

# Section: Data Summary
elif section == "Data Summary":
    st.title("Dataset Summary")
    if training_df is not None:
        st.subheader("Class Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Label", data=training_df, ax=ax1, order=sorted(training_df['Label'].unique()))
        st.pyplot(fig1)
    else:
        st.warning("Training data statistics not available.")

# Section: Training Metrics
elif section == "Training Metrics":
    st.title("Training Progress")
    if training_df is not None:
        st.subheader("Loss over Epochs")
        st.line_chart(training_df[['train_loss', 'val_loss']])

        st.subheader("Accuracy over Epochs")
        st.line_chart(training_df[['train_acc', 'val_acc']])
    else:
        st.warning("Training metrics not found.")

# Section: Evaluation Report
elif section == "Evaluation Report":
    st.title("Model Evaluation")
    if eval_report is not None:
        st.markdown(f"### Test Accuracy: **{eval_report['accuracy']*100:.2f}%**")

        st.subheader("Classification Report")
        st.json(eval_report['classification_report'])

        st.subheader("Confusion Matrix")
        cm = np.array(eval_report['confusion_matrix'])
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=eval_report['classes'], yticklabels=eval_report['classes'])
        st.pyplot(fig2)
    else:
        st.warning("Evaluation report not available.")

# Section: Live Inference
elif section == "Live Inference":
    st.title("Try It Out: Upload or Select an Image")

    class_labels = ['F0', 'F1', 'F2', 'F3', 'F4']

    def load_model():
        model = models.densenet121(weights=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_labels))
        )
        model.load_state_dict(torch.load(os.path.join(script_dir, "model.pt"), map_location="cpu", weights_only=True))
        model.eval()
        return model

    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(image.convert('RGB')).unsqueeze(0)

    model = load_model()

    selected_image = None

    # Enhanced grid layout with image selection using button under each image
    st.subheader("🔍 Select from Preloaded Images by click the button below. The prediction will appear at the end of this page.")
    preloaded_images = []
    preloaded_captions = []
    img_dir = os.path.join(script_dir, "dataset/validation")
    for cls in class_labels:
        files = (glob(f"{img_dir}/{cls}/*.jpg") + glob(f"{img_dir}/{cls}/*.png"))
        for idx, file in enumerate(files):
            preloaded_images.append(file)
            preloaded_captions.append(f"{cls} - {idx+1}")

    selected_image = None
    grid_cols = 5
    grid_rows = len(preloaded_images) // grid_cols
    for row in range(grid_rows):
        cols = st.columns(grid_cols)
        for i in range(grid_cols):
            idx = row * grid_cols + i
            if idx < len(preloaded_images):
                with cols[i]:
                    #st.image(preloaded_images[idx], caption=preloaded_captions[idx], width=120)
                    st.image(preloaded_images[idx],  width=120)
                    if st.button(preloaded_captions[idx]):
                        selected_image = preloaded_images[idx]
                        st.session_state.selected_image = selected_image

    uploaded_file = st.file_uploader("Or Upload Your Own Ultrasound Image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width =True)
    elif "selected_image" in st.session_state:
        selected_image = st.session_state.selected_image
        image = Image.open(selected_image)
        st.image(image, caption=f"Selected Image ({selected_image})", use_container_width =True)

    if image is not None:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
            pred_idx = np.argmax(probs)
            pred_label = class_labels[pred_idx]

        st.markdown(f"### Predicted Class: **{pred_label}**")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability (%)": probs * 100
        })
        colors = ["#1f77b4"] * len(class_labels)
        colors[pred_idx] = "#ff7f0e"
        fig3, ax3 = plt.subplots()
        bars = ax3.bar(prob_df['Class'], prob_df['Probability (%)'], color=colors)
        ax3.set_ylabel("Probability (%)")
        ax3.set_ylim(0, 100)

        # Add probability values on top of each bar
        for bar, prob in zip(bars, prob_df['Probability (%)']):
            height = bar.get_height()
            ax3.annotate(f"{prob:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

        st.pyplot(fig3)


        # --- Optional: Grad-CAM visualization ---
        ENABLE_GRADCAM = True
        #if ENABLE_GRADCAM:
        #    st.markdown("#### Grad-CAM Visualization (optional)")
        #    st.info("Grad-CAM logic would go here.")

        # --- Optional: Inference Logging ---
        ENABLE_LOGGING = True
        if ENABLE_LOGGING:
            log_entry = {
                "image": uploaded_file.name if uploaded_file else selected_image,
                "prediction": pred_label,
                "probabilities": probs.tolist()
            }
            with open(os.path.join(script_dir, "inference_log.jsonl"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    else:
        st.info("Please select or upload an image to begin inference.")
