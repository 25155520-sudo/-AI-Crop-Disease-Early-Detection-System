import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from preprocess import load_dataset
from model import CropDiseaseCNN
import uuid

os.makedirs("dataset/healthy", exist_ok=True)
os.makedirs("dataset/diseased", exist_ok=True)

def cross_entropy(pred, label):
    pred = np.clip(pred, 1e-9, 1.0)
    return -np.log(pred[label])

def save_training_image(uploaded_file, label):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((32, 32))

    filename = str(uuid.uuid4()) + ".jpg"

    if label == "Healthy":
        save_path = os.path.join("dataset/healthy", filename)
    else:
        save_path = os.path.join("dataset/diseased", filename)

    img.save(save_path)

def main():
    st.title("🌾 AI Crop Disease Early Detection System")
    st.write("Built using Pure NumPy CNN")

    st.subheader("➕ Add Training Image")

    train_upload = st.file_uploader("Upload Training Image", type=["jpg", "png", "jpeg"], key="train")

    if train_upload is not None:
        label = st.radio("Select Label", ["Healthy", "Diseased"])

        if st.button("Save to Dataset"):
            save_training_image(train_upload, label)
            st.success(f"Image saved to dataset/{label.lower()}")

    st.markdown("---")

    if st.button("Train Model"):

        healthy_count = len(os.listdir("dataset/healthy"))
        diseased_count = len(os.listdir("dataset/diseased"))

        if healthy_count == 0 or diseased_count == 0:
            st.error("Add at least 1 Healthy and 1 Diseased image before training.")
            return

        st.write(f"Healthy Images: {healthy_count}")
        st.write(f"Diseased Images: {diseased_count}")

        st.write("Loading dataset...")
        X, y = load_dataset("dataset")

        model = CropDiseaseCNN()

        epochs = st.slider("Select Epochs", 1, 10, 3)
        progress = st.progress(0)

        losses = []
        accuracies = []

        for epoch in range(epochs):
            loss = 0
            correct = 0

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for i in range(len(X)):
                output = model.forward(X[i])
                loss += cross_entropy(output, y[i])

                if np.argmax(output) == y[i]:
                    correct += 1

            epoch_loss = loss / len(X)
            epoch_acc = correct / len(X)

            losses.append(epoch_loss)
            accuracies.append(epoch_acc)

            progress.progress((epoch + 1) / epochs)

        st.success("Training Complete!")

        # Plot Loss
        st.subheader("📈 Training Loss")
        fig = plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot(fig)

        # Plot Accuracy
        st.subheader("📊 Training Accuracy")
        fig2 = plt.figure()
        plt.plot(accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        st.pyplot(fig2)

        st.session_state.model = model

    st.markdown("---")

    st.subheader("📷 Upload Leaf Image for Prediction")

    uploaded_file = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"], key="test")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0

        st.image(img, caption="Uploaded Leaf Image", width=200)

        if "model" in st.session_state:
            model = st.session_state.model
            prediction = model.forward(img_array)

            predicted_class = np.argmax(prediction)
            result = "🌿 Healthy" if predicted_class == 0 else "🍂 Diseased"

            st.subheader("🔍 Prediction Result")
            st.success(result)

            st.subheader("📊 Prediction Confidence")
            fig = plt.figure()
            classes = ["Healthy", "Diseased"]
            plt.bar(classes, prediction)
            plt.xlabel("Class")
            plt.ylabel("Probability")
            st.pyplot(fig)

        else:
            st.warning("Train the model first.")

if __name__ == "__main__":
    main()