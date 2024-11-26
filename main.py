import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained ResNet model without the fully connected layer
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features

# Replace the fully connected layer to match the number of classes in your dataset (3 classes)
model.fc = torch.nn.Linear(num_ftrs, 3)  # Update to 3 classes for plant disease recognition

# Load the trained model weights (skip the mismatch in the final layer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load('crop_disease_detection.pth', weights_only=True), strict=False)


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model prediction function
def model_prediction(image_path):
    image = Image.open(image_path)  # Open the image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # Get model output
        _, predicted = torch.max(output, 1)  # Get the predicted class index
    return predicted.item()

# Sidebar for navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)  # Use the new parameter
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset contains images of healthy and diseased crop leaves, categorized into 38 different classes.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    # Display the uploaded image
    if test_image is not None:
        st.image(test_image, use_container_width=True)  # Use the new parameter
    
    # Predict button
    if st.button("Predict"):
        st.snow()  # Show snow animation while processing
        result_index = model_prediction(test_image)

        # Class names (ensure this list matches the model's class labels)
        class_name = [
            'Tomato___Bacterial_spot',
            'Tomato___Target_Spot', 
            'Tomato___Tomato_mosaic_virus', 
            
            # Add all the class names corresponding to your dataset here
        ]
        
        # Check if the index is valid
        if 0 <= result_index < len(class_name):
            st.success(f"Model is predicting it's a {class_name[result_index]}")
        else:
            st.error("Prediction out of range, please check your model or dataset.")
