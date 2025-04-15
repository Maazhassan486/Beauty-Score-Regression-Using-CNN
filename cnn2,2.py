from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images

import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths for the dataset and ratings file
data_dir = r"C:\Users\maazg\Downloads\ugly images\SCUT-FBP5500_v2.1\SCUT-FBP5500_v2\Images"
ratings_file = r"C:\Users\maazg\Downloads\ugly images\SCUT-FBP5500_v2.1\SCUT-FBP5500_v2\All_Ratings.xlsx"

# Define the path to save or load the model
save_path = r"C:\Users\maazg\PycharmProjects\PythonProject\.venv\ugly_model.keras"

# Load ratings from the Excel file.
# We assume the Excel file has a header row and that the image name is in the 2nd column and rating in the 3rd column.
df = pd.read_excel(ratings_file)
# Select the 2nd and 3rd columns (index 1 and 2) and rename them accordingly.
df = df.iloc[:, [1, 2]]
df.columns = ["filename", "beauty_score"]

print(df.head())

# Split the dataset into training and testing sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create ImageDataGenerators for training and testing.
# Training generator applies augmentation.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Testing generator only rescales images.
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Use flow_from_dataframe to load images and associate them with beauty scores.
# We use class_mode="raw" since our target (beauty_score) is a continuous value.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="filename",
    y_col="beauty_score",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=data_dir,
    x_col="filename",
    y_col="beauty_score",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw"
)

# Build the convolutional neural network model.
# The final Dense layer has one neuron (with linear activation by default) to output any score.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # Linear activation for regression
])

# Compile the model using mean squared error (MSE) for regression.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])  # Mean Absolute Error as a metric

# Check if the model exists; if yes, load it. Otherwise, train the model.
if os.path.exists(save_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(save_path)
else:
    print("No existing model found. Proceeding with training...")
    epochs = 1  # For testing purposes; adjust as needed.
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    # Evaluate the model on the test dataset.
    test_loss, test_mae = model.evaluate(test_generator)
    print("Test MAE:", test_mae)

    # Save the model.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Visualize predictions for a batch of images from the test set.
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)

plt.figure(figsize=(12, 12))
for i in range(min(9, len(test_images))):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])

    # Get the predicted beauty score for the i-th image.
    pred_score = predictions[i][0]
    # Define outcome based on a threshold (e.g., 2.5); adjust the threshold as needed.
    predicted_outcome = "Beauty" if pred_score > 2.5 else "Not Beauty"

    # Get the true beauty score for the i-th image.
    true_score = test_labels[i]

    # Print the predicted score and outcome to the console.
    print(f"Predicted Beauty Score: {pred_score:.2f}")
    print(f"Predicted Outcome: {predicted_outcome}")

    # Set the subplot title with predicted score, outcome, and true score.
    plt.title(f"Score: {pred_score:.2f}\nOutcome: {predicted_outcome}\nTrue: {true_score:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# Define a function to predict the beauty score and outcome for a single image.
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]
    # Define a threshold to classify the outcome (this can be adjusted)
    predicted_outcome = "Beauty" if prediction > 4 else "Not Beauty"

    print(f"Predicted Beauty Score: {prediction:.2f}")
    print(f"Predicted Outcome: {predicted_outcome}")

    plt.imshow(img)
    plt.title(f"Score: {prediction:.2f}\nOutcome: {predicted_outcome}")
    plt.axis('off')
    plt.show()

# Example usage:
# Update this path to point to an image you want to classify.
img_path = r"C:\Users\maazg\Documents\university assignments\AI\assignment1\21L-5793\Screenshots_Assignment1\irfan3.png"
predict_image(img_path, model)
