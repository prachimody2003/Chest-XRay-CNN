# Chest X-Ray Pneumonia Detection Using CNN

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D
)
from tensorflow.keras.models import Sequential


# Dataset paths
dataset_path = "chest_xray"

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")


# Store images and labels
X = []
Y = []


# Load NORMAL images
for image_name in os.listdir(train_path + "/NORMAL"):

    image_path = os.path.join(
        train_path,
        "NORMAL",
        image_name
    )

    image = Image.open(image_path).convert("RGB")
    image = image.resize((100, 100))

    img_array = np.array(image) / 255.0

    X.append(img_array)
    Y.append(0)


# Load PNEUMONIA images
for image_name in os.listdir(train_path + "/PNEUMONIA"):

    image_path = os.path.join(
        train_path,
        "PNEUMONIA",
        image_name
    )

    image = Image.open(image_path).convert("RGB")
    image = image.resize((100, 100))

    img_array = np.array(image) / 255.0

    X.append(img_array)
    Y.append(1)


# Convert lists to NumPy arrays
X = np.array(X)
Y = np.array(Y)


# Display dataset information
print("Total images:", len(Y))
print("Normal images:", np.sum(Y == 0))
print("Pneumonia images:", np.sum(Y == 1))

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print("Training labels:", Y_train.shape)
print("Testing labels:", Y_test.shape)


# Build the CNN model
model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(100, 100, 3)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(Flatten())

model.add(
    Dense(
        units=128,
        activation="relu"
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)

model.summary()


# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train the model
history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, Y_test)
)


# Save the trained model
model.save("chest_xray_cnn.keras")
print("Model saved")


# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
plt.close()


# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
plt.close()


# Evaluate the model
print("Starting evaluation...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# Generate predictions
Y_pred_prob = model.predict(
    X_test,
    verbose=0
)

Y_pred = (
    Y_pred_prob > 0.5
).astype(int).ravel()


# Create confusion matrix
cm = confusion_matrix(
    Y_test,
    Y_pred
)

print("Confusion matrix:")
print(cm)


# Create classification report
report = classification_report(
    Y_test,
    Y_pred,
    target_names=["NORMAL", "PNEUMONIA"]
)

print("Classification report:")
print(report)


# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(
    Y_test,
    Y_pred_prob.ravel()
)

roc_auc = auc(
    fpr,
    tpr
)

print("AUC score:", roc_auc)


# Plot ROC curve
plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.show()
plt.close()


# Select a random test class
actual_class = random.choice(
    ["NORMAL", "PNEUMONIA"]
)

selected_folder = os.path.join(
    test_path,
    actual_class
)


# Select valid image files
image_list = [
    file_name
    for file_name in os.listdir(selected_folder)
    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


# Select one random test image
new_image_name = random.choice(image_list)

new_image_path = os.path.join(
    selected_folder,
    new_image_name
)


# Prepare the selected image
new_image = Image.open(
    new_image_path
).convert("RGB")

new_image = new_image.resize(
    (100, 100)
)

new_image_array = (
    np.array(new_image) / 255.0
)

new_image_array = np.expand_dims(
    new_image_array,
    axis=0
)


# Predict pneumonia probability
prediction_probability = model.predict(
    new_image_array,
    verbose=0
)[0][0]


# Convert probability into a class
if prediction_probability > 0.5:
    prediction_class = "PNEUMONIA"
else:
    prediction_class = "NORMAL"


# Compare prediction with actual class
if prediction_class == actual_class:
    result = "CORRECT"
else:
    result = "INCORRECT"


# Display the selected image
plt.imshow(new_image)

plt.title(
    f"Actual: {actual_class}\n"
    f"Predicted: {prediction_class}\n"
    f"Result: {result}"
)

plt.axis("off")
plt.show()
plt.close()


# Print prediction details
print("Image name:", new_image_name)
print("Actual class:", actual_class)
print(
    "Pneumonia probability:",
    prediction_probability
)
print(
    "Final prediction:",
    prediction_class
)
print(
    "Prediction result:",
    result
)