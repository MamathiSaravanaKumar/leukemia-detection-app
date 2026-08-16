# LeukemiaVision

### Blood Cell Classification Using CNN, CBAM Attention and Grad-CAM
LeukemiaVision is a deep learning-based computer vision application designed to classify blood-cell images into four categories: **Benign, Early, Pre, and Pro**. The project combines a **Convolutional Neural Network (CNN)** with the **Convolutional Block Attention Module (CBAM)** to improve feature representation. **Grad-CAM** is incorporated to provide visual explanations of the model's predictions. The trained model is integrated into an interactive **Streamlit** application where users can upload a blood-cell image and obtain a predicted class, confidence score, Grad-CAM visualization, and experimental nucleus-area analysis.
**Disclaimer:** LeukemiaVision is an educational/research prototype and is not intended to replace professional medical diagnosis or clinical decision-making.

## KEY FEATURES

-  Four-class blood-cell image classification
-  CNN-based deep learning model
-  CBAM attention mechanism
-  Grad-CAM visual explainability
-  Prediction confidence score
-  Experimental nucleus-area analysis
-  Interactive Streamlit web application
-  Blood-cell image preprocessing and visualization

## MODEL ARCHITECTURE

The LeukemiaVision model uses a CNN backbone followed by CBAM attention and a classification head.

Input Image (224 × 224 × 3) --> Conv2D (32 filters) --> Batch Normalization(ReLu) --> Max Pooling --> Conv2D (64 filters) --> Batch Normalization(ReLu) --> Max Pooling --> Conv2D (128 filters) --> Batch Normalization(ReLu) --> Max Pooling --> CBAM Attention --> Channel or  Spatial Attention --> Global Average Pooling --> Dense (128) --> Dropout (0.5) --> Dense (4) -->  Softmax --> Benign / Early / Pre / Pro

**CBAM ATTENTION**

The Convolutional Block Attention Module (CBAM) helps the network focus on more informative features within blood-cell images. CBAM consists of two sequential attention mechanisms:
  Channel Attention
    Channel attention identifies the feature channels that contain more useful information.
  Spatial Attention
    Spatial attention identifies important spatial regions within the feature maps.
By combining channel and spatial attention, the model can refine the features used for classification.

## GRAD CAM EXPLAINABILITY
LeukemiaVision uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize regions of the input image that contribute to the model's prediction. The generated heatmap is overlaid on the original blood-cell image to provide an interpretable visual representation of the model's decision.

Blood Cell Image --> CNN + CBAM Model --> Model Prediction --> Grad-CAM --> Activation Heatmap -->Visualization Overlay

## DATASET
  The model was trained using a blood-cell leukemia image dataset obtained from Kaggle. The dataset contains four classification categories:
     Benign
     Early
     Pre
     Pro
The training pipeline uses an 80/20 training-validation split.

## IMAGE PREPROCESSING
 Before being passed to the model, input images undergo preprocessing.
  Preprocessing Steps
    Resize images to 224 × 224 pixels
    Normalize pixel values to the range 0–1
    Apply data augmentation during training
  Data Augmentation
   The training pipeline uses:
     Random horizontal flipping
     Random rotation
     Random zoom
Validation images are resized and normalized without the training augmentation operations.

## MODEL TRAINING
  The model was trained using the following configuration:
| Parameter               | Value                    |
| ----------------------- | ------------------------ |
| Input Image Size        | 224 × 224 × 3            |
| Batch Size              | 32                       |
| Optimizer               | Adam                     |
| Learning Rate           | 0.0003                   |
| Loss Function           | Categorical Crossentropy |
| Label Smoothing         | 0.1                      |
| Maximum Epochs          | 25                       |
| Dropout                 | 0.5                      |
| Early Stopping          | Enabled                  |
| Learning Rate Reduction | Enabled                  |

## MODEL PERFORMANCE
 The model achieved an overall 96% accuracy on 651 evaluated samples.
Classification Report
| Class                | Precision |   Recall | F1-Score | Support |
| -------------------- | --------: | -------: | -------: | ------: |
| Benign               |      0.93 |     0.91 |     0.92 |     104 |
| Early                |      0.95 |     0.98 |     0.96 |     192 |
| Pre                  |      0.99 |     0.95 |     0.97 |     187 |
| Pro                  |      0.98 |     1.00 |     0.99 |     168 |
| **Macro Average**    |  **0.96** | **0.96** | **0.96** | **651** |
| **Weighted Average** |  **0.96** | **0.96** | **0.96** | **651** |

## CONFUSION MATRIX

| Actual \ Predicted | Benign | Early | Pre | Pro |
| ------------------ | -----: | ----: | --: | --: |
| **Benign**         |     95 |     6 |   1 |   2 |
| **Early**          |      2 |   188 |   1 |   1 |
| **Pre**            |      5 |     4 | 177 |   1 |
| **Pro**            |      0 |     0 |   0 | 168 |

The model achieved the highest F1-score for the Pro class at 0.99, followed by Pre (0.97), Early (0.96), and Benign (0.92).

## EXPERIMENTAL NUCLEUS ANALYSIS
The Streamlit application includes an additional experimental image-processing component for nucleus-area analysis.
The process involves:
  Converting the image to grayscale
  Applying adaptive thresholding
  Performing morphological processing
  Estimating the segmented pixel area
  Calculating an estimated area ratio
The processed nucleus image is displayed alongside the model prediction.
This is an experimental image-processing feature and should not be interpreted as clinical leukemia staging or diagnosis.

## APPLICATION
The application is built using Streamlit and allows users to:
 Upload a blood-cell image
 View the processed image
 Obtain the predicted class
 View the prediction confidence
 Generate a Grad-CAM visualization
 View the experimental nucleus analysis

## TECHNOLOGIES USED
Python
TensorFlow
Keras
OpenCV
NumPy
Pandas
Pillow
Streamlit

## INSTALLATIONS
  Clone the Repository
      git clone https://github.com/MamathiSaravanaKumar/leukemia-detection-app.git
      cd leukemia-detection-app
 Create a Virtual Environment
      python -m venv venv
## ACTIVATE THE ENVIRONMENT
Windows:
  venv\Scripts\activate
Linux/macOS:
  source venv/bin/activate
## INSTALL DEPENDENCIES
pip install -r requirements.txt
## RUN THE APPLICATIONS
 Run the Streamlit application using:
  streamlit run app.py
Upload a .jpg, .jpeg, or .png blood-cell image to obtain the classification result and visual analysis.

## PROJECT STRUCTURE
leukemia-detection-app/
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── leukemia_model_fixed.h5
│
├── results/
│
└── architecture/

## FUTURE ENHANCEMENTS
Evaluation using larger and more diverse datasets
Independent external dataset validation
Comparison with established CNN architectures
Improved nucleus segmentation and analysis
Additional model explainability techniques
Improved visualization and user interface
Further research toward clinically validated systems

## LIMITATIONS
Model performance depends on the characteristics and distribution of the training dataset.
Results may vary for images that differ significantly from the training data.
The model has not been presented as a clinically validated diagnostic system.
The nucleus-area analysis is an experimental image-processing feature.
Independent clinical validation would be required before any clinical application.


