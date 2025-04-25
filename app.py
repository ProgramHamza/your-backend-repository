from flask import Flask
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
import base64
import re
from io import BytesIO
from tqdm import tqdm

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def bytes_to_int(b):
    return int.from_bytes(b, 'big')

def read_images(filename, n_max_images):
    images = []
    with open(filename, "rb") as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_colums = bytes_to_int(f.read(4))
        for _ in range(n_images):
            image = [[f.read(1) for _ in range(n_colums)] for _ in range(n_rows)]
            images.append(image)
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for _ in range(n_labels):
            labels.append(bytes_to_int(f.read(1)))
    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(a, b):
    return sum([(bytes_to_int(a_i) - bytes_to_int(b_i))**2 for a_i, b_i in zip(a, b)])**0.5

def get_training_distance_for_sample(test_sample, X_train):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def max_frequency(l):
    return max(l, key=l.count)

def knn(X_train, Y_train, X_test, k=3):
    y_pred = []
    for sample in tqdm(X_test, desc="Running k-NN", total=len(X_test), unit="sample"):
        distances = get_training_distance_for_sample(sample, X_train)
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        k_nearest = [Y_train[i] for i in sorted_indices[:k]]
        y_pred.append(max_frequency(k_nearest))
    return y_pred

# Load training data once
DATA_DIR = "C:/Users/mhrom/Downloads/data"
X_train = extract_features(read_images(f"{DATA_DIR}/train-images.idx3-ubyte", 10000))
Y_train = read_labels(f"{DATA_DIR}/train-labels.idx1-ubyte")

@socketio.on("predict_digit")
def handle_predict(data):
    try:
        image_data = data["image"]
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        img_bytes = BytesIO(base64.b64decode(img_str))
        img = Image.open(img_bytes).convert("L").resize((28, 28))
        img_np = np.asarray(img)

        sample = [[bytes([p]) for p in row] for row in img_np.tolist()]
        sample = extract_features([sample])
        prediction = knn(X_train, Y_train, sample, k=3)[0]

        emit("prediction_result", {"prediction": prediction})
    except Exception as e:
        emit("prediction_result", {"error": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True)
