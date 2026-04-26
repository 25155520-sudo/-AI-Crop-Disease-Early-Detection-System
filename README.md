This project is a simple AI-based crop disease detection system built using pure NumPy (no TensorFlow/PyTorch) and deployed with Streamlit.

It allows users to:

Upload crop leaf images
Train a custom CNN model
Predict whether a leaf is Healthy 🌿 or Diseased 🍂

How It Works
1. Data Input
Upload leaf images via UI
Label as:
Healthy
Diseased
2. Model Architecture
Convolution Layer
ReLU Activation
Max Pooling Layer
Flatten
Dense Layer
Softmax Output
3. Training
Uses:
Cross Entropy Loss
Accuracy Tracking
User selects number of epochs
4. Prediction
Upload a test image
Model outputs:
Class (Healthy/Diseased)
Confidence scores

How to Run = 
Step-1= Install dependencies
Step-2 = Run the app
Step-3 = Open the browser

Dataset
Images are stored locally in:

dataset/healthy/
dataset/diseased/
Images are:
Converted to grayscale
Resized to 32x32
Normalized (0 to 1)

Limitations
No backpropagation (weights are not updated)
Works on very small datasets only
Accuracy may be low for real-world use
Basic CNN (for learning purposes) 
