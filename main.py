# ==========================
# Chest X-Ray CNN project
# ==========================
#Import required libraries
# os         -> Work with files and folders (dataset paths, file names)
# numpy      -> Numerical computing and array operations (images become NumPy arrays)
# pandas     -> Data manipulation and analysis (tables/DataFrames)
# matplotlib -> Display images, graphs, and training results
# tensorflow -> Build, train, and evaluate the CNN model
# PIL (Image)-> Open, resize, convert, and process image files
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# ==========================
#Environment Check
# ==========================
#print("TensorFlow:", tf.__version__)
#print("NumPy:", np.__version__)
#print("Pandas:", pd.__version__)

#print("\n✅ Environment Ready!")

# ==========================
# Dataset Path Setup
# ==========================
dataset_path = "chest_xray"

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

#print()
#print(train_path)
#print(test_path)
#print(val_path)

# ==========================
#check folder contents
# ==========================
#print("\n Train folders:" , os.listdir(train_path))

# ==========================
# load one image(normal)
# ==========================
normal_image_name = os.listdir(train_path + "/NORMAL")[0]
#print("Image name:", normal_image_name)

# ==========================
#build the full img path
# ==========================
normal_image_path = os.path.join(train_path, "NORMAL", normal_image_name)
#print("Full path:", normal_image_path)

# ==========================
#open the image
# ==========================
image = Image.open(normal_image_path)
#print(type(image))

# ==========================
#Display the img
# ==========================
#plt.imshow(image)
#plt.title("NORMAL CHEST X-RAY")
#plt.axis("off")
#plt.show()

# ==========================
#Convert img into numpy array
# ==========================
image = image.convert("RGB")
image = image.resize((224,224))
img_array = np.array(image)
#print(type(img_array))
#print(img_array.shape)

# ==========================
# Initialize the containers
# ==========================
X = []
Y = []

# ==========================
#load normal images (label = 0)
# ==========================
# Loop through every image in the NORMAL folder
for image_name in os.listdir(train_path + "/NORMAL"):

    # Create the complete path to the current image
    image_path = os.path.join(train_path, "NORMAL", image_name)

    # Open the image and convert it to RGB (3 color channels)
    image = Image.open(image_path).convert("RGB")

    # Resize all images to the same size (100x100 pixels)
    image = image.resize((100,100))

    # Convert image to a NumPy array and normalize pixel values (0–255 → 0–1)
    img_array = np.array(image) / 255.0

    # Store the processed image in X (features)
    X.append(img_array)

    # Assign label 0 for NORMAL and store it in Y (labels)
    Y.append(0)

# ==========================
# load pneumonia images (label = 1)
# ==========================
for image_name in os.listdir(train_path + "/PNEUMONIA"):
    image_path = os.path.join(train_path, "PNEUMONIA", image_name)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((100, 100))

    img_array = np.array(image) / 255.0

    X.append(img_array)
    # Assign label 0 for NORMAL and store it in Y (labels)
    Y.append(1)

# ==========================
#convert to NumPy Arrays
# Before np.array(X), Your X was a Python list,Each item inside was already a NumPy array,But the whole collection was still a normal Python list.
# After np.array(X),You combine all images into one large NumPy array
#CNN models cannot directly work efficiently with a Python list, They expect a mathematical tensor
#Python List-NumPy Array-TensorFlow CNN input
# ==========================
X = np.array(X)
Y = np.array(Y)

# ==========================
#Check Shapes
# ==========================
# X shows: Num of Images, Height of each image, Width of each image, RGB channels
#print(X.shape)

# Y shows: Number of labels.
#print(Y.shape)

#==========================
# Dataset Exploration
#==========================
# check the total no. of images
# count NORMAL and PNEUMONIA samples
# and verify the final dataset shape
# Count the number of NORMAL images (label = 0)
#Y == 0 → checks which labels are 0 (True/False)
#np.sum() → counts all True values
# Count the number of PNEUMONIA images (label = 1)
print("Total images:", len(Y))
print("Normal images:", np.sum(Y == 0))
print("Pneumonia images:", np.sum(Y == 1))

print("X shape:", X.shape)
print("Y shape:", Y.shape)

#dataset is not balanced
#Approximate percentages
#NORMAL: 1341 / 5216 ≈ 26%
#PNEUMONIA: 3875 / 5216 ≈ 74%
#So there are many more pneumonia images than normal images.
# bias- Imbalanced dataset → Accuracy can be misleading
# so use Precision, Recall, Confusion Matrix, and ROC-AUC for proper evaluation.

#==========================
# Split the dataset
#==========================
# Divide data into training and testing sets
# model learns from training and evaluates on the testing set.
# Use 20% of the dataset for testing and 80% for training
#test_size=0.2
#5216 images
# 80% → Training = 4172,20% → Testing = 1044
# Fix the random split so the results are the same every time the code runs
# random_state=42
# Keep the same NORMAL and PNEUMONIA class proportion in both training and testing sets
#stratify=Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print("Training labels:", Y_train.shape)
print("Testing labels:", Y_test.shape)

# ==========================
# Build the CNN model
# ==========================
# Import TensorFlow Keras modules for creating the CNN.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
# Keras: Framework for building deep learning models.
# Sequential: Stack layers in order.
# Conv2D: Extract image features.
# MaxPooling2D: Reduce image size.
# Flatten: Convert image features into a 1D vector.
# Dense: Make the final classification.

# ==========================
# Create the CNN model
# ==========================
# Create an empty Sequential CNN model to which layers will be added.
# ==========================
# Conv2D Layer
# ==========================

# Conv2D extracts important features from images using learnable filters.
# It detects patterns such as edges, textures, and abnormalities.

# filters=32:
# The CNN learns 32 different feature detectors.
# Each filter produces one feature map.

# kernel_size=(3,3):
# Each filter uses a 3x3 window to scan the image.
# The kernel slides across the image and detects local patterns.

# activation="relu":
# ReLU introduces non-linearity.
# Negative values become 0, and positive values remain unchanged.

# input_shape=(100,100,3):
# Height = 100 pixels
# Width = 100 pixels
# 3 = RGB color channels

# Output shape calculation:
# Input: 100 x 100 x 3
# Kernel: 3 x 3
# No padding:
# Output size = (100 - 3 + 1) x (100 - 3 + 1)
#             = 98 x 98

# Since we use 32 filters:
# Output = 98 x 98 x 32

# Parameter calculation:
# Parameters = (kernel height x kernel width x input channels x filters) + biases
#
# = (3 x 3 x 3 x 32) + 32
# = 864 + 32
# = 896 trainable parameters
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),activation="relu",input_shape=(100,100,3)))
#model.summary()

# ==========================
# MaxPooling Layer
# ==========================

# MaxPooling reduces the size of feature maps while keeping important information.

# pool_size=(2,2):
# A 2x2 window moves across the feature map.
# It keeps only the maximum value from each region.

# Before pooling:
# 98 x 98 x 32

# After pooling:
# 98/2 x 98/2 x 32
# = 49 x 49 x 32

# MaxPooling does not learn weights or biases.
# Therefore:
# Parameters = 0
model.add(MaxPooling2D(pool_size=(2,2)))
#model.summary()

# ==========================
# Flatten Layer
# ==========================

# Flatten converts 3D feature maps into a 1D vector.
# This prepares the extracted features for Dense layers.

# Before Flatten:
# 49 x 49 x 32

# Calculation:
# 49 x 49 x 32 = 76832

# After Flatten:
# 76832 values

# Flatten only changes the shape.
# It does not learn parameters.

# Parameters = 0
model.add(Flatten())
#model.summary()

# ==========================
# Dense Layer
# ==========================

# Dense is a fully connected layer.
# Every input feature is connected to every neuron.

# units=128:
# Creates 128 neurons that learn combinations of extracted features.

# activation="relu":
# Adds non-linearity and helps learn complex patterns.

# Input features:
# 76832

# Neurons:
# 128

# Weight calculation:
# 76832 x 128
# = 9,834,496 weights

# Bias calculation:
# Each neuron has one bias:
# 128 neurons = 128 biases

# Total parameters:
# Weights + Biases
# = 9,834,496 + 128
# = 9,834,624 trainable parameters
model.add(Dense(units = 128 , activation = "relu"))
model.summary()
# = 9,834,624 + 896
# = 9,835,520  total trainable parameters

# ==========================
# Output Layer
# ==========================
# Dense(1) creates one output neuron for binary classification.
# Sigmoid converts the output into a probability between 0 and 1.
# Output:
# 0 → NORMAL
# 1 → PNEUMONIA

# Parameters:
# (128 × 1) + 1 = 129
model.add(Dense(units = 1 , activation = "sigmoid"))
model.summary()
# total parameters = 129 + 9,835,520 = 9,835,649

# ==========================
# Compile the CNN Model
# ==========================
# optimizer="adam"
# Updates weights and biases to reduce prediction errors.

# loss="binary_crossentropy"
# Calculates the error for binary classification (NORMAL vs PNEUMONIA).

# metrics=["accuracy"]
# Displays the percentage of correct predictions during training.

model.compile( optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# ==========================
# Train the CNN Model
# ==========================
# history stores the training results after every epoch.
# It saves values such as:
# - Training Loss
# - Training Accuracy
# - Validation Loss
# - Validation Accuracy

    # X_train contains the training images.
    # These are the images the CNN learns from.
    # Shape: (4172, 100, 100, 3)

  # Y_train contains the correct labels for each image.
    # 0 = NORMAL
    # 1 = PNEUMONIA
    # These labels help the model know whether its prediction was correct.

  # epochs=10 means the model will study the ENTIRE training dataset
    # 10 separate times.
    #
    # Example:
    # Epoch 1 -> Model sees all 4172 training images once.
    # Epoch 2 -> Model sees all 4172 images again.
    # ...
    # Epoch 10 -> Model has seen every training image 10 times.
    #
    # Each new epoch allows the model to improve by learning
    # from mistakes made in previous epochs.

    # batch_size=32 means the model does NOT process all 4172 images
    # at once.
    #
    # Instead, it divides the dataset into small groups
    # of 32 images called batches.
    #
    # For each batch:
    # 1. Make predictions.
    # 2. Calculate loss.
    # 3. Update weights and biases.
    # 4. Move to the next batch.
    #
    # Smaller batches require less memory and usually train faster.

    # validation_data is used to evaluate the model after
    # every epoch using data it has NEVER learned from.
    #
    # X_test -> unseen test images
    # Y_test -> correct labels for those images
    #
    # Validation helps measure how well the model generalizes
    # to new chest X-rays instead of simply memorizing
    # the training dataset.

# Uses unseen test images to evaluate the model after each epoch.
# It calculates validation accuracy and validation loss.
# The model does NOT learn from this data—it is only used to monitor performance.

history = model.fit(
X_train,
Y_train,
epochs=10,
batch_size=32,
validation_data=(X_test, Y_test)
)
# save before plotting graphs
model.save("chest_xray_cnn.keras")
print("Model saved")

# ==========================
# Training Results
# ==========================

# Epoch:
# One complete pass through the entire training dataset.

# 131/131:
# Total number of batches processed in one epoch.
# 4172 training images / batch size 32 ≈ 131 batches.

# accuracy:
# Percentage of correctly classified training images.

# loss:
# Measures how wrong the model's predictions are.
# Lower loss indicates better learning.

# val_accuracy:
# Accuracy on unseen test images after each epoch.
# Measures how well the model generalizes.

# val_loss:
# Error on unseen test images.
# Lower validation loss indicates better performance on new data.

#Training results may vary slightly each time the model is trained because
# the CNN starts with randomly initialized weights.
# However, the overall performance and conclusions usually remain the same.
# Using random seeds improves reproducibility,
# while random_state=42 only ensures the train-test split remains the same.

# ==========================
# Plot Accuracy Graph
# ==========================
plt.figure(figsize=(8,5))

# history.history["accuracy"]
# Returns the training accuracy after each epoch.
plt.plot(history.history["accuracy"],label = "Training accuracy")
# history.history["val_accuracy"]
# Returns the validation accuracy after each epoch.
plt.plot(history.history["val_accuracy"],label = "Validation accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
plt.close() # Closes the graph window and frees memory

# plt.plot()
# Draws the graph.

# plt.xlabel()
# Labels the x-axis (Epochs).

# plt.ylabel()
# Labels the y-axis (Accuracy).

# plt.legend()
# Displays labels for each plotted line.

# plt.show()
# Displays the completed graph.

# ==========================
# Accuracy Graph Interpretation
# ==========================

# Blue Line:
# Training accuracy increases with each epoch,
# showing that the CNN is learning from the training data.

# Orange Line:
# Validation accuracy measures performance on unseen images.
# High validation accuracy indicates good generalization.

# Observation:
# Both curves remain close to each other,
# suggesting good model performance.
# A slight drop in validation accuracy near the end
# indicates minor overfitting.

# ==========================
# Plot Loss Graph
# ==========================

plt.figure(figsize=(8,5))

# history.history["loss"]
# Returns the training loss after each epoch.
plt.plot(history.history["loss"], label="Training Loss")
# history.history["val_loss"]
# Returns the validation loss after each epoch
plt.plot(history.history["val_loss"], label="Validation Loss")
# Training loss:
# Shows how much error the model makes on training data.

# Validation loss:
# Shows how much error the model makes on unseen data.

# Lower loss indicates better model performance.
# Comparing training and validation loss helps identify overfitting

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
plt.close() # Closes the graph window and frees memory
# Training Loss:
# Measures prediction error on training data.
# Lower loss indicates better learning.

# Validation Loss:
# Measures prediction error on unseen data.
# If validation loss increases while training loss decreases,
# it indicates the model is beginning to overfit.

# 131/131
# Total batches processed in one epoch.
# 4172 training images ÷ batch size (32) ≈ 131 batches.
# When all 131 batches are processed, one epoch is complete.

# ==========================
# Evaluate the trained model
# ==========================
# model.evaluate() tests the trained CNN
# using unseen test images.

# X_test -> Unseen test images

# Y_test -> Correct labels for the test images

# Returns:
# test_loss -> Prediction error on the test set.
# test_accuracy -> Percentage of correctly classified test images.

# A high test accuracy indicates that the model
# # generalizes well to new data.
print("Starting evaluation..")
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose = 0)

# verbose
#  Controls how much information TensorFlow displays.

#  verbose=0
#  No progress bar (silent mode)

#verbose=1
# Shows progress bar (default)

#  verbose=2
#  Shows one summary line per operation

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
# During evaluation:
# 33/33 means TensorFlow processed all 33 batches of the test dataset.
# 1044 test images ÷ batch size (32) = 33 batches
# (32 full batches + 1 smaller batch of 20 images).

# ==========================
# Generate Predictions
# ==========================
# Predict the probability of pneumonia for every test image
Y_pred_prob = model.predict(X_test, verbose = 0)
# convert probabilities into class labels
# prob > 0.5 = pneumonia (1)
# prob <= 0.5 = normal (0)
Y_pred = ( Y_pred_prob > 0.5 ).astype(int).ravel()
# ravel()- coverts 2D array into 1D array, used so that predicted labels
# have same shape as Y_test

print("prediction probability shape:", Y_pred_prob.shape)
print("Final prediction shape:",Y_pred.shape)
# debugging step optional

# ==========================
# Create Confusion Matrix
# ==========================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# ==========================
# Classification Report
# ==========================
from sklearn.metrics import classification_report
# generate classification report
report = classification_report(Y_test, Y_pred, target_names = ["NORMAL", "PNEUMONIA"])
print(report)
# ==========================
# Classification Report
# ==========================

# Precision
# Out of all images predicted as Pneumonia,
# how many were actually Pneumonia.
# Higher Precision = Fewer False Positives.

# Recall
# Out of all actual Pneumonia images,
# how many were correctly detected.
# Higher Recall = Fewer False Negatives.
# Most important metric in medical diagnosis.

# F1-score
# Combines Precision and Recall into a single score.
# Higher F1-score indicates a better balance between both.

# Support
# Number of actual test images in each class.
# Does not measure model performance.

# Macro Average
# Average performance of all classes.
# Treats every class equally, regardless of dataset size.

# Weighted Average
# Average performance based on the number of samples.
# Classes with more images have greater influence on the final score.

# ==========================
# ROC Curve and AUC
# ==========================
from sklearn.metrics import roc_curve, auc
# roc_curve()- calculates the false positive rate (FPR),
# true positive rate (TPR), threshold
# auc()- calculates the area under the ROC curve
# Y_test
# Actual labels of the test images.
# Y_pred_prob
# Predicted probability of Pneumonia for each test image.
# Threshold
# Probability value used to classify images as Normal or Pneumonia.
# use predicted probabilities, not final 0/1 labels
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob.ravel())

# calculate area under ROC curve
roc_auc = auc(fpr, tpr)
print("AUC score:", roc_auc)

#plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
# f""
# Formatted string used to insert variable values into text.
# {roc_auc:.3f}
# Displays the AUC value with 3 decimal places.
plt.plot([0, 1], [0, 1], linestyle="--")
#Plot the diagonal reference line.
# Represents a random classifier (AUC = 0.5).
# A good ROC curve should lie above this line.
# Plot the diagonal reference line.
# Represents a random classifier (AUC = 0.5).
# A good ROC curve should lie above this line.

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
plt.close()

# ==========================
# Predict One Chest X-ray
# ==========================

# This section selects one image from either the NORMAL
# or PNEUMONIA test folder, preprocesses it, sends it to the
# trained CNN model, and checks whether the prediction is correct

# Randomly choose the actual class
actual_class = random.choice(["NORMAL", "PNEUMONIA"])

# Create the path to the selected class folder
selected_folder = os.path.join(test_path, actual_class)

#get only valid images from the selected folder
image_list = [ file_name
               for file_name in os.listdir(selected_folder)
               if file_name.lower().endswith((".jpg", ".jpeg", ".png"))]

# randomly select one image from the chosen folder
new_image_name = random.choice(image_list)

# create complete file path of that selected image
new_image_path = os.path.join(selected_folder, new_image_name)

# open the image and convert to rbg format
new_image = Image.open(new_image_path).convert("RGB")

#resize to match the size used during training
new_image = new_image.resize((100, 100))

# convert img to numpy array and normalize pixel values
new_image_array = np.array(new_image) / 255.0

# add batch dimensions, shape change from (100,100,3) to (1,100,100,3)
new_image_array = np.expand_dims(new_image_array, axis = 0)

#predict probability
prediction_probability = model.predict(new_image_array , verbose = 0)[0][0]
# model.predict() returns a 2D NumPy array, even when predicting a single image.
# The first [0] selects the first prediction in the batch,
# and the second [0] extracts the probability value from that prediction.

# convert probability into class label
if prediction_probability > 0.5:
    prediction_class = "PNEUMONIA"
else:
    prediction_class = "CLASS"

# compare predicted with actual class
if prediction_class == actual_class:
    result = "CORRECT"
else:
    result = "INCORRECT"

# display the selected chest X-ray image
plt.imshow(new_image)

# display actual class, predicted class and result
plt.title(
    f"Actual: {actual_class}\n" 
f"Predicted: {prediction_class} \n"
f"Result: {result}\n"
)

# hide axes
plt.axis('off')

# display the img
plt.show()

#close the figure and release memory
plt.close()

# print prediction details
print("Image name", new_image_name)
print("Actual class:", actual_class)
print("Pneumonia probability:", prediction_probability)
print("Final prediction" , prediction_class)
print("Prediction result:", result)
