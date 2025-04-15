# Beauty Score Regression Using CNN

This project implements a Convolutional Neural Network (CNN) to predict a continuous beauty score from images. The model is trained on a dataset where each image is associated with a beauty rating from an Excel file. In addition to predicting a continuous score, the model classifies images as "Beauty" or "Not Beauty" based on a defined threshold.

## Dataset

- **Images:**  
  The images are downloaded from a dataset named SCUT-FBP5500_v2.

- **Ratings File:**  
  The beauty ratings are provided in an Excel file All_Ratings.xlsx.
  
  **Note:**  
  The Excel file should have a header row. This project expects the image filename in the 2nd column and the beauty score in the 3rd column.

## Requirements

- Python 3.6 or higher
- TensorFlow (with Keras integrated)
- Pandas
- scikit-learn
- matplotlib
- Pillow

### Installation

You can install the required packages using `pip`:

```bash
pip install tensorflow pandas scikit-learn matplotlib pillow
