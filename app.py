import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="Steel Defect Classification", page_icon="🔍", layout="wide")

st.title("🔍 Steel Surface Defect Classification")
st.write("Upload an image of a steel surface to predict the type of defect.")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_SIZE = 224
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
        self.features = nn.Sequential(
            conv_block(3,   32),
            conv_block(32,  64),
            conv_block(64,  128),
            conv_block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def build_resnet18(num_classes=6):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model

@st.cache_resource
def load_models():
    resnet_path = os.path.join(RESULTS_DIR, "resnet18_best.pth")
    resnet_model = None
    if os.path.exists(resnet_path):
        resnet_model = build_resnet18(num_classes=len(CLASSES))
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        resnet_model.to(DEVICE)
        resnet_model.eval()

    cnn_path = os.path.join(RESULTS_DIR, "custom_cnn_best.pth")
    cnn_model = None
    if os.path.exists(cnn_path):
        cnn_model = CustomCNN(num_classes=len(CLASSES))
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn_model.to(DEVICE)
        cnn_model.eval()
        
    return resnet_model, cnn_model

resnet_model, cnn_model = load_models()

st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["ResNet18", "Custom CNN"])

selected_model = resnet_model if model_choice == "ResNet18" else cnn_model

if selected_model is None:
    st.sidebar.error(f"{model_choice} weights not found! Please train the model first.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Prediction")
        if selected_model is not None:
            input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = selected_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                predicted_class = CLASSES[predicted_idx]
                confidence = probabilities[predicted_idx].item()
            
            st.success(f"**Predicted Defect:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            st.write("Class Probabilities:")
            prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
            st.bar_chart(prob_dict)
        else:
            st.error("Model is not loaded. Cannot make a prediction.")
