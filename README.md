WBC Classification using Vision Transformer (ViT)

A Deep Learning Project for Automated White Blood Cell Recognition
This project implements a Vision Transformer (ViT)-based deep learning model to classify four types of white blood cells (WBCs) from microscopic images. It helps automate blood smear analysis in medical workflows by providing consistent, high-accuracy predictions. A full Flask-based web interface is included so users can upload an image and get real-time predictions.\

Table of Contents
1) Project Overview
2) Problem Statement
3) Dataset Used
4) Model Architecture
5) Features
6) Tech Stack
7) Project Structure
8) How to Run the Project
9) How to Reproduce the Model Training
10) Results
11) Future Improvements
12) License
    
1. Project Overview
This project uses a fine-tuned Vision Transformer (ViT Base 16) to classify WBC images into four categories:
Eosinophil
Lymphocyte
Monocyte
Neutrophil
It includes:
A Flask web application for inference
Preprocessing pipeline for medical images
Attention-based feature extraction
A clean UI for uploading images and receiving predictions

2. Problem Statement
Manual blood smear analysis is time-consuming, requires expert supervision, and is subject to human inconsistency. Automated WBC classification can improve reliability and reduce diagnostic workload in hematology labs.
This project aims to develop a production-ready model capable of accurately classifying WBC types using transformer-based computer vision.

3. Dataset Used
The model is trained on the Raabin-WBC dataset, a widely used open-source hematology dataset.
Dataset Details
Contains labeled microscopic cell images
High variability in illumination, staining, orientation, and cell shape
Four relevant classes used in this project:
Eosinophil
Lymphocyte
Monocyte
Neutrophil

Preprocessing Applied:
-Resizing to 224×224
-Normalization (ImageNet mean/std)
-Data augmentation:
-Horizontal flip
-Random rotation
-Color jitter


4. Model Architecture
-Vision Transformer (ViT Base - Patch Size 16)
-Key components:
-Patch embedding
-Multi-head self-attention
-Transformer encoder blocks
-Classification head replaced with a custom linear layer
-Softmax output for 4 classes
-The pretrained ViT model is fine-tuned on the Raabin dataset.

5. Features
-Transformer-based classifier for medical imaging
-High accuracy with strong generalization across cell variations
-Attention-based learning instead of convolutions
-Simple, clean Flask UI for uploading images
-Handles various file formats (.jpg, .png, .jpeg)
-Deployed-ready structure

6. Tech Stack
-Machine Learning
-PyTorch
-Torchvision
-Vision Transformer (ViT Base 16)
-Scikit-Learn
-Web Framework
-Flask
-Jinja2
-Others
-PIL
-NumPy
-HTML/CSS


7. Project Structure
├── vit_attention_app.py        # Flask app for inference
├── model/                      # Saved model checkpoint
│   └── vit_best_model.pth
├── static/                     # CSS, JS, images
├── templates/                  # HTML templates
│   └── index.html
├── utils/
│   └── transforms.py           # Preprocessing functions
├── README.md
└── requirements.txt


8. How to Run the Project (Locally)
   
Step 1: Clone the Repository
git clone https://github.com/yourusername/wbc-vit-classification.git
cd wbc-vit-classification

Step 2: Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Flask App
python vit_attention_app.py

The server will start at:
http://127.0.0.1:5000
Upload an image and get predictions instantly.


9. How to Reproduce Model Training
If you want to retrain the model:
-Download the Raabin-WBC dataset
-Organize it by class folders
-Run the training script (if included) or use your own training pipeline
-Save the model weights as vit_best_model.pth in the project directory

10. Results
Performance Summary
Achieved strong accuracy using ViT fine-tuning
Improved feature extraction using transformer attention
Robust classification across varied staining and lighting
(Replace with your exact results if available)

11. Future Improvements
Potential enhancements:
-Add Grad-CAM or attention map visualization
-Deployment using Docker + AWS
-Add support for more WBC types
-Build an end-to-end lab assistant dashboard

12. License
This project is open-source under the MIT License.

