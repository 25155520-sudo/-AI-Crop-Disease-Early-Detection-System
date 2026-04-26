import os
import numpy as np
from PIL import Image

def load_images(folder_path, label, img_size=(32, 32)):
    data = []
    labels = []

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        try:
            img = Image.open(img_path).convert("L")  
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  

            data.append(img_array)
            labels.append(label)
        except:
            continue

    return data, labels


def load_dataset(base_path):
    healthy_path = os.path.join(base_path, "healthy")
    diseased_path = os.path.join(base_path, "diseased")

    healthy_data, healthy_labels = load_images(healthy_path, 0)
    diseased_data, diseased_labels = load_images(diseased_path, 1)

    X = np.array(healthy_data + diseased_data)
    y = np.array(healthy_labels + diseased_labels)

    return X, y