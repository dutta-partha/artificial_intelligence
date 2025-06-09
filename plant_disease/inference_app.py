import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from glob import glob



# --- Constants and Configuration ---
TEST_DATA_PATH = "./TEST" # Path to your test images (if used for listing classes, though report labels are better)
NUM_CLASSES = 8

# Page config and title
st.set_page_config(layout="wide")
st.title("🌿 Tomato Plant Disease Classifier Dashboard")

# Define the custom PlantDNet model architecture (as provided by the user)
class PlantDNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantDNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Data Loading Functions ---

@st.cache_data
def load_eval_report(model_name):
    """Loads the evaluation report JSON for the specified model."""
    file_path = f"plant_disease_eval_report_{model_name}.json"
    if os.path.exists(file_path):
        with open(file_path) as f:
            report = json.load(f)
        return report
    else:
        st.error(f"Evaluation report for {model_name} not found at {file_path}")
        return None

@st.cache_data
def load_training_stats_data(model_name):
    """Loads the training statistics JSON for the specified model."""
    file_path = f"plant_disease_training_stats_{model_name}.json"
    if os.path.exists(file_path):
        return pd.read_json(file_path)
    else:
        st.error(f"Training stats for {model_name} not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame on error

@st.cache_data
def load_disease_details():
    """Loads general disease details (assuming this is a single file)."""
    file_path = "disease_details.json" # Assuming this file exists and is universal
    if os.path.exists(file_path):
        return pd.read_json(file_path)
    else:
        st.warning(f"Disease details file not found at {file_path}. Disease descriptions will not be available.")
        return {} # Return empty dict on error


# --- Model Loading Functions ---

@st.cache_resource
def load_model_mobilenet():
    """Loads the MobileNetV2 model."""
    model = models.mobilenet_v2(weights=None) # Start with no pretrained weights from torchvision
    # Modify the classifier head for 8 classes
    # The original MobileNetV2 classifier is nn.Sequential(nn.Dropout(...), nn.Linear(...))
    # We replace the final Linear layer
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False), # This maps to 'classifier.0' - no weights/bias expected.
        
        nn.Sequential(                    # This maps to 'classifier.1' - it's a Sequential module.
            nn.Identity(),                # This maps to 'classifier.1.0' - no weights/bias expected.
            nn.Linear(in_features, NUM_CLASSES) # This maps to 'classifier.1.1' - it has weights/bias.
        )
    )
    model_path = "plantd_model_MobileNetV2.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model
    else:
        st.error(f"MobileNetV2 model not found at {model_path}. Inferencing will not work.")
        return None

@st.cache_resource
def load_model_plantdnet():
    """Loads the custom PlantDNet model."""
    model = PlantDNet(num_classes=NUM_CLASSES)
    model_path = "plantd_model_PlantDNet.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model
    else:
        st.error(f"PlantDNet model not found at {model_path}. Inferencing will not work.")
        return None

# Universal model loader based on selection
@st.cache_resource
def get_model(model_type):
    if model_type == "MobileNetV2":
        return load_model_mobilenet()
    elif model_type == "PlantDNet":
        return load_model_plantdnet()
    return None # Should not happen if selectbox values are handled

# --- Plotting Functions (from previous code, slightly adapted for clarity) ---

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig

def plot_classification_report_chart(class_report_df, title="Classification Report (Per Class)"):
    """
    Plots a grouped bar chart for precision, recall, and f1-score for each class.
    Expects a DataFrame with 'precision', 'recall', 'f1-score' columns and class names as index.
    """
    metrics = ['precision', 'recall', 'f1-score']
    # Filter out accuracy, macro avg, weighted avg if they are present in the dataframe
    class_report_filtered = class_report_df.drop(
        columns=[col for col in ['support', 'accuracy'] if col in class_report_df.columns],
        errors='ignore'
    ).loc[
        ~class_report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])
    ]

    classes = class_report_filtered.index.tolist()
    # Check if there are any classes to plot after filtering
    if not classes:
        st.warning("No class-wise metrics found to plot in the classification report.")
        return None

    # Reshape for seaborn's barplot
    plot_df = class_report_filtered.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    plot_df = plot_df[plot_df['Metric'].isin(metrics)] # Ensure only desired metrics are plotted

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.8), 6))
    sns.barplot(x='index', y='Score', hue='Metric', data=plot_df, palette='viridis', ax=ax)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 1)
    ax.legend(title='Metrics')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Menu",
    [
        "ℹ️ About",
        "📂 Data Summary",
        "📈 Training Stats",
        "🧪 Inferencing",
    ]
)

# Load general disease details (universal)
disease_details = load_disease_details()

# --- Main App Logic ---

if page == "ℹ️ About":
    st.header("🌿 About Plant Disease Classifier")
    st.markdown("This application classifies plant diseases from leaf images using deep learning. " \
    "It demonstrates and compares a custom Convolutional Neural Network (PlantDNet) and a Transfer Learning approach (MobileNetV2)." \
    " The models are trained to identify 8 distinct conditions, including various diseases and healthy leaves.")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Placeholder for an image, replace with your actual logo
        st.image("landing_page_logo.png", width=300, caption="Plant Disease Classifier Logo")


elif page == "📂 Data Summary":
    st.header("📂 Data Summary")
    st.info("This section summarizes the overall dataset used for training and evaluation. It represents the combined data distribution for all models.")

    # Load one of the reports to get the labels and support
    # Using MobileNetV2's report for data summary as classes should be consistent
    sample_report = load_eval_report("MobileNetV2")
    if sample_report:
        labels = sample_report["labels"]
        st.subheader(f"Total Classes: {len(labels)}")

        # Calculate approximate distribution (assuming support in report is test set count per class)
        # Multiply by a factor (e.g., 5) to represent total dataset if your test set is 1/5th of total, etc.
        # This assumes your original data distribution image was based on total samples, not just test
        class_supports = [sample_report["classification_report"][label]["support"] for label in labels]
        dist_df = pd.DataFrame({
            "Class": labels,
            "Count": [s * (520 / sum(class_supports)) if sum(class_supports) > 0 else 0 for s in class_supports] # Scale to 520 total if support is test set count
        })
        dist_df['Count'] = dist_df['Count'].round().astype(int)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(dist_df["Class"], dist_df["Count"], color="#1f77b4")
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Approximate Data Class Distribution", fontsize=16)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("Could not load sample report for data summary. Please ensure 'plant_disease_eval_report_MobileNetV2.json' is available.")


elif page == "📈 Training Stats":
    st.header("📈 Training and Evaluation Statistics")

    selected_model_for_stats = st.selectbox(
        "Select Model for Training Statistics:",
        ("PlantDNet", "MobileNetV2"),
        key="stats_model_selector"
    )

    stats_df = load_training_stats_data(selected_model_for_stats)
    eval_report = load_eval_report(selected_model_for_stats)

    if not stats_df.empty and eval_report:
        st.subheader(f"📊 Training Metrics for {selected_model_for_stats}")
        st.line_chart(stats_df.set_index("epoch")[['train_acc', 'val_acc']])
        st.line_chart(stats_df.set_index("epoch")[['train_loss', 'val_loss']])

        st.subheader(f"📊 Overall Accuracy for {selected_model_for_stats}")
        overall_accuracy = eval_report['classification_report']['accuracy']
        st.metric("", f"✅  {overall_accuracy*100:.2f}%")

        st.subheader(f"🔢 Classification Report for {selected_model_for_stats}")
        # Convert classification report part to DataFrame for easy plotting
        class_report_df = pd.DataFrame(eval_report['classification_report']).T
        # Drop summary rows if they exist in the DataFrame before displaying or plotting
        class_report_display_df = class_report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(class_report_display_df.style.format("{:.4f}")) # Format to 4 decimal places

        # Show classification report chart
        fig_report = plot_classification_report_chart(class_report_df, title=f"Classification Report (Per Class) for {selected_model_for_stats}")
        if fig_report:
            st.pyplot(fig_report)
        else:
            st.warning("Could not generate classification report chart.")


        st.subheader(f"🧮 Confusion Matrix for {selected_model_for_stats}")
        labels = eval_report['labels']
        cm = np.array(eval_report['confusion_matrix'])
        # If the confusion matrix is too large, consider displaying a smaller version or just the plot
        # st.dataframe(pd.DataFrame(cm, index=labels, columns=labels)) # Optional: raw CM table

        # Show graphical confusion matrix
        fig_cm = plot_confusion_matrix(cm, labels, title=f"Confusion Matrix for {selected_model_for_stats}")
        st.pyplot(fig_cm)
    else:
        st.info(f"Select a model to view its training and evaluation statistics.")


elif page == "🧪 Inferencing":
    st.header("🧪 Plant Disease Image Inferencing")

    selected_model_for_inference = st.selectbox(
        "Select Model for Inferencing:",
        ("MobileNetV2", "PlantDNet"), # MobileNetV2 first as it's higher performing
        key="inference_model_selector"
    )

    current_model = get_model(selected_model_for_inference)

    # Load labels from the selected model's evaluation report for inferencing consistency
    current_eval_report = load_eval_report(selected_model_for_inference)
    if current_eval_report:
        class_labels = current_eval_report['labels']
    else:
        st.error("Could not load class labels for inferencing. Please check evaluation report JSON files.")
        class_labels = [] # Fallback to empty list

    if not current_model or not class_labels:
        st.warning("Model or class labels not loaded. Please check file paths and selection.")
        st.stop() # Stop execution if model/labels are missing


    # Enhanced grid layout with image selection using button under each image
    st.subheader("🔍 Select from Preloaded Images by click the button below. The prediction will appear at the end of this page.")
    preloaded_images = []
    preloaded_captions = []
    grid_cols = len(class_labels) # Number of columns based on number of classes
    grid_rows = 5 # Number of rows for the grid layout

    img_dir = "TEST" # Directory where preloaded images are stored
    for cls in class_labels:
        files = (glob(f"{img_dir}/{cls}/*.jpg") + glob(f"{img_dir}/{cls}/*.png"))
        for idx, file in enumerate(files[:grid_rows]):
            preloaded_images.append(file)
            preloaded_captions.append(f"{cls} - {idx+1}")

    selected_image = None

    for row in range(grid_rows):
        cols = st.columns(grid_cols)
        for i in range(grid_cols):
            idx = row * grid_cols + i
            if idx < len(preloaded_images):
                with cols[i]:
                    #st.image(preloaded_images[idx], caption=preloaded_captions[idx], width=120)
                    st.image(preloaded_images[idx],  width=90)
                    if st.button(preloaded_captions[idx]):
                        selected_image = preloaded_images[idx]
                        st.session_state.selected_image = selected_image


    col1, col2 = st.columns([1, 1.2])
    with col1:
        uploaded_file = st.file_uploader("Or Upload a plant leaf image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=False)
        elif "selected_image" in st.session_state:
            selected_image = st.session_state.selected_image
            image = Image.open(selected_image)
            st.image(image, caption=f"Selected Image ({selected_image})", use_container_width =False)

    with col2:
        if "selected_image" in st.session_state or uploaded_file:
            # Define image transformations for inference (standard for many models)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], # ImageNet means
                                     [0.229, 0.224, 0.225]) # ImageNet stds
            ])
            input_tensor = transform(image.convert("RGB")).unsqueeze(0) # Add batch dimension

            with torch.no_grad(): # Disable gradient calculation for inference
                output = current_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy() # Get probabilities
                predicted_idx = np.argmax(probabilities)
                predicted_label = class_labels[predicted_idx]

                st.subheader("📌 Predicted Class")
                
                # Extract actual class from filename for comparison (optional but good for testing)
                actual_class_from_filename = ""
                # Extract image name from selected image or uploaded image
                if uploaded_file is not None and hasattr(uploaded_file, "name"):
                    file_name_lower = uploaded_file.name.lower()
                elif selected_image is not None:
                    file_name_lower = os.path.basename(selected_image).lower()


                file_name_lower = uploaded_file.name.lower()
                    # Attempt to extract class from filename (e.g., "some_bacterial_spot_image.jpg")
                for label_token in [l.lower() for l in class_labels]:
                    if label_token in file_name_lower:
                        actual_class_from_filename = label_token.replace("_", " ") # Convert back to spaced
                        break

                is_correct_prediction = False

                if actual_class_from_filename:
                    # Simple comparison: if predicted label matches (case-insensitive, space-agnostic)
                    if predicted_label.lower().replace(" ", "_") == actual_class_from_filename.lower().replace(" ", "_"):
                        is_correct_prediction = True
                    
                    color = "#30df56" if is_correct_prediction else "#ff4b4b"
                    icon = "✅" if is_correct_prediction else "❌"
                    st.markdown(
                        f"### <span style='color:{color}'><b>`{predicted_label}`</b>  {icon}</span>",
                        unsafe_allow_html=True
                    )
                    if actual_class_from_filename and not is_correct_prediction:
                        st.markdown(f"*(Actual: **`{actual_class_from_filename}`**)*")

                    # Show extra info from knowledgebase if available and prediction is correct
                    if is_correct_prediction and (predicted_label in disease_details):
                        details = disease_details[predicted_label]
                        st.info(f"**Description**: {details.get('description', 'N/A')}")
                        st.warning(f"**Cause**: {details.get('cause', 'N/A')}")
                        st.success(f"**Remedies**: {details.get('remedies', 'N/A')}")
                    elif predicted_label in disease_details:
                         # Still show details even if incorrect, might be useful
                         details = disease_details[predicted_label]
                         st.info(f"**Description**: {details.get('description', 'N/A')}")
                         st.warning(f"**Cause**: {details.get('cause', 'N/A')}")
                         st.success(f"**Remedies**: {details.get('remedies', 'N/A')}")
                else: # If actual class couldn't be extracted from filename
                    st.markdown(f"### **`{predicted_label}`**")
                    if predicted_label in disease_details:
                         details = disease_details[predicted_label]
                         st.info(f"**Description**: {details.get('description', 'N/A')}")
                         st.warning(f"**Cause**: {details.get('cause', 'N/A')}")
                         st.success(f"**Remedies**: {details.get('remedies', 'N/A')}")


            st.subheader("📊 Class Probabilities")
            df_probs = pd.DataFrame({
                'Class': class_labels,
                'Probability (%)': probabilities * 100
            }).sort_values(by='Probability (%)', ascending=False) # Sort for better readability

            bar_colors = ['#1f77b4'] * len(class_labels)
            # Find the index of the predicted label in the sorted dataframe
            predicted_idx_sorted = df_probs.index.get_loc(df_probs[df_probs['Class'] == predicted_label].index[0])
            bar_colors[predicted_idx_sorted] = "#30df56" # Highlight predicted class

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(df_probs['Class'], df_probs['Probability (%)'], color=bar_colors) # Use barh for horizontal bars
            ax.set_xlabel("Probability (%)", fontsize=12)
            ax.set_ylabel("Class", fontsize=12)
            ax.set_xlim(0, 100)
            ax.set_title("Class Probabilities", fontsize=16)
            plt.gca().invert_yaxis() # Invert y-axis to have highest probability at top

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                        ha='left', va='center', fontsize=9, color='brown')

            st.pyplot(fig)
        else:
            st.info("Upload an image to begin Inferencing.")