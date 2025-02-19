# Traffic Sign Detection and Classification using AI and Computer Vision

## Overview
This project aims to detect and classify traffic signs using AI and computer vision techniques. The system leverages deep learning and image processing methods to accurately identify various traffic signs from images. The solution is implemented in Python using TensorFlow, Keras, OpenCV, and other essential libraries.

## Features
- **Traffic Sign Detection:** Locate and identify traffic signs in images.
- **Traffic Sign Classification:** Classify detected traffic signs into predefined categories.
- **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) for robust feature extraction and classification.
- **Image Preprocessing:** Includes steps for image resizing, normalization, and color space conversion.
- **Model Evaluation:** Provides performance evaluation on test data and visualizes prediction results.
- **Model Saving & Loading:** Save the trained model for later inference tasks.

## Dataset
The project uses CSV files to manage the dataset:
- **Train.csv:** Contains paths and labels for training images.
- **Test.csv:** Contains paths and labels for testing images.
- **Meta.csv:** Contains metadata and class information for traffic signs.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/alqabali2/TrafficSignDetection.git
   cd TrafficSignDetection
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8 or higher installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   The required libraries include TensorFlow, Keras, OpenCV, Pandas, NumPy, and Matplotlib.

## Usage
### Training the Model
Run the training script to preprocess data, train the CNN model, evaluate its performance, and save the trained model:
```bash
python train_model.py
```
The script will:
- Load and preprocess the training images.
- Train the CNN model.
- Evaluate the model on test data.
- Save the trained model as `traffic_signs_model.h5`.

### Testing the Model
After training, run the testing script to load the saved model and perform predictions on test images:
```bash
python test_model.py
```
The script will:
- Load test images.
- Perform predictions using the trained model.
- Display random test images along with their actual and predicted labels.

## Project Structure
```
TrafficSignDetection/
├── Train.csv
├── Test.csv
├── Meta.csv
├── train_model.py
├── test_model.py
├── requirements.txt
└── README.md
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.
4. Follow the project's coding guidelines and include tests for new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or further information, please contact:
- **Email:** [adel.aqlabali@gmail.com](mailto:adel.aqlabali@gmail.com)
- **GitHub:** [alqabali2](https://github.com/alqabali2)
