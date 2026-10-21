🎨 AI Art Authentication – Multi-Signal Detection and Forensic Reporting System

An AI-powered artwork classification system designed to distinguish **AI-generated images from real, human-created artwork** using deep learning. The project uses a ResNet-50 convolutional neural network to analyze uploaded images and return a predicted class with a confidence score through a web-based application.

📌 Project Overview

With the rapid growth of generative AI tools, distinguishing AI-generated artwork from human-created artwork has become increasingly challenging. This project aims to support artwork analysis by developing a deep-learning-based classification system that examines an input image and predicts whether it belongs to the AI-generated or real-art category.

The system integrates a trained ResNet-50 model with a FastAPI backend and a web frontend to provide an accessible image-analysis workflow.

 Objectives

- Classify images into **AI-GENERATED** and **REAL** categories.
- Apply deep learning using the ResNet-50 architecture.
- Provide image-upload and prediction functionality through a web interface.
- Display the predicted category and model confidence score.
- Evaluate model performance using standard classification metrics.
- Establish a foundation for additional forensic analysis and explainability features.

✨ Key Features

- **AI vs. Real Classification:** Predicts whether an uploaded image belongs to the AI-generated or real-image category.
- **ResNet-50 Deep Learning Model:** Uses a trained convolutional neural network for image classification.
- **Web-Based Interface:** Integrates a frontend with the backend for image submission and result display.
- **FastAPI Backend:** Provides an API endpoint for processing uploaded images and returning predictions.
- **Confidence Score:** Displays the model's confidence for its predicted category.
- **GPU Acceleration:** Supports CUDA-enabled inference when a compatible GPU and PyTorch installation are available.
- **Model Evaluation:** Supports evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.

## 🧠 Model and Dataset

The current model is based on **ResNet-50** and was trained using a binary classification setup with the AI-ArtBench-style dataset downloaded for this project.

The training dataset contains images from AI-generated categories and human-created artwork styles. The model was trained using a balanced subset of 10,000 images, comprising 5,000 images per class, for one training epoch in the current pilot configuration.

### Dataset Summary

| Split | AI-generated | Real artwork | Total |
|---|---:|---:|---:|
| Training dataset available | 105,015 | 50,000 | 155,015 |
| Test dataset | 20,000 | 10,000 | 30,000 |
| Balanced training subset used | 5,000 | 5,000 | 10,000 |

### Initial Evaluation Results

The current model achieved **99.86% accuracy** on the dataset's 30,000-image held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 99.86% |
| Precision (REAL) | 99.76% |
| Recall (REAL) | 99.82% |
| F1-score (REAL) | 99.79% |

These results apply to the specified dataset and evaluation setup. They do not guarantee equivalent performance on unseen artists, new image generators, social-media images, or real-world artwork. Further independent validation is necessary.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Model Architecture:** ResNet-50
- **Image Processing:** Pillow, Torchvision
- **Backend API:** FastAPI
- **API Server:** Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Hardware Acceleration:** NVIDIA CUDA, when available

## 🏗️ System Architecture

1. **Image Input:** The user uploads an image through the frontend.
2. **API Request:** The frontend sends the image to the FastAPI backend.
3. **Preprocessing:** The backend resizes and normalizes the image for the trained model.
4. **Classification:** ResNet-50 generates scores for the two classes.
5. **Result Generation:** The backend returns the predicted class and confidence score.
6. **Result Display:** The frontend presents the response to the user.

## 📂 Project Structure

```text
AI-Art-Authentication/
├── backend.py
├── resnet50_artbench.pth
├── train_resnet_artbench.py
├── evaluate_resnet_artbench.py
├── predict_artbench.py
├── art_dataset/
│   └── Real_AI_SD_LD_Dataset/
│       ├── train/
│       └── test/
└── README.md

AI-Art-Authentication-system/
├── index.html
├── login.html
├── home.html
├── scan.html
├── result.html
├── analysis.html
├── evidence.html
├── history.html
├── report.html
├── settings.html
├── script.js
└── style.css
```

*The structure above summarizes the main project files. Keep large datasets and model checkpoints out of the repository unless you intentionally distribute them.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_ART_AUTHENTICATION
```

Replace the placeholder with your repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch torchvision fastapi uvicorn pillow python-multipart
```

Install any additional dependencies required by your training and evaluation scripts.

### 4. Prepare the model checkpoint

Place the trained `resnet50_artbench.pth` checkpoint in the project directory expected by `backend.py`.

The checkpoint is a trained model artifact; it is not automatically downloaded by the installation commands above.

### 5. Start the backend

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

### 6. Access the API documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the interactive API documentation to test the image-prediction endpoint.

### 7. Run the frontend

Open the frontend using your chosen local development server or the project's configured hosting method. Ensure its API URL points to the running backend.

## 🔌 API Endpoint

### `POST /predict`

Accepts an uploaded image and returns its predicted category and confidence score.

**Example response:**

```json
{
  "prediction": "AI-GENERATED",
  "confidence": 98.45,
  "model": "ResNet-50"
}
```

The values shown are illustrative. Actual results depend on the uploaded image and model output.

## 📊 Evaluation Metrics

The model can be evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct for the selected class.
- **Recall:** Proportion of actual samples of the selected class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Breakdown of correct and incorrect predictions for both classes.

## 🚀 Future Enhancements

The project is designed to support further development, including:

- **FFT-based Frequency Analysis:** Explore frequency-domain patterns in images.
- **Grad-CAM Visualization:** Highlight image regions that influence the model's prediction.
- **Metadata Analysis:** Examine available image metadata as supporting forensic evidence.
- **Forensic Report Generation:** Combine prediction results, visual explanations, and metadata into a structured report.
- **Analysis History:** Allow users to review previous image analyses.
- **Mobile Integration:** Extend the system to a mobile interface.
- **Cross-Generator Evaluation:** Test generalization on AI generators and artwork sources not represented in training.

These enhancements should be described as planned features until they are implemented and tested.

## ⚠️ Limitations

- The model predicts image categories; it does not prove authorship or establish the authenticity of an artwork.
- High performance on a benchmark dataset does not necessarily imply high real-world accuracy.
- Performance may vary across artists, art styles, image formats, compression levels, and AI generators.
- Confidence scores are model outputs and should not be interpreted as calibrated probabilities without additional validation.
- Metadata may be absent, modified, or unreliable and should not be used alone to determine authenticity.

## 🎓 Project Information

**Project Title:** Multi-Signal Detection and Forensic Reporting System for AI-Generated Art Authentication

**Domain:** Artificial Intelligence, Deep Learning, Digital Image Forensics

**Model:** ResNet-50

**Application Type:** Web-Based Image Classification System

## 👩‍💻 Contributors

Aiswarya Sumesh - Backend development,
Alina Francis - Frontend development,
Amal N.U - App development,
Anandhu V.S - Integration


*This project is developed for academic and research purposes to explore deep-learning-based AI-generated image detection and digital artwork analysis.*
