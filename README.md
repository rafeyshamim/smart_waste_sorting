# Smart Waste Sorting App

An intelligent waste classification application that uses deep learning to classify waste images into six categories: glass, paper, cardboard, plastic, metal, and trash.

## Features

- 🖼️ Image upload and classification
- 🤖 Deep learning-based waste classification using ResNet18
- 📊 Detailed probability scores for all waste categories
- 🎨 Modern and intuitive user interface
- ⚡ Real-time predictions

## Dataset

This app uses the [TrashNet dataset](https://github.com/garythung/trashnet) which contains 2,527 images across 6 categories:
- Glass: 501 images
- Paper: 594 images
- Cardboard: 403 images
- Plastic: 482 images
- Metal: 410 images
- Trash: 137 images

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd smart_waste_sorting
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Extract the dataset (if not already extracted):**
   The dataset should be extracted automatically, but if needed:
   ```bash
   python -c "import zipfile; z = zipfile.ZipFile('data/trashnet/data/dataset-resized.zip'); z.extractall('data/trashnet/data/')"
   ```

## Training the Model

Before using the app, you need to train the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the TrashNet dataset
- Train a ResNet18 model with transfer learning
- Save the trained model to `models/waste_classifier.pth`
- Display training progress and test accuracy

**Note:** Training may take 30-60 minutes depending on your hardware. The model will be automatically saved when validation accuracy improves.

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Upload an image** and click "Detect Waste" to get predictions!

## Usage

1. Click "Choose File" to select an image
2. Preview the selected image
3. Click "Detect Waste" to analyze the image
4. View the predicted waste category and confidence scores
5. See probability scores for all waste categories

## Project Structure

```
smart_waste_sorting/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── predict.py             # Prediction module
├── requirements.txt       # Python dependencies
├── models/                # Trained model directory (created after training)
│   ├── waste_classifier.pth
│   └── class_mapping.json
├── uploads/               # Upload directory (created automatically)
├── templates/
│   └── home.html         # Web interface
├── static/
│   └── style.css         # Styling
└── data/
    └── trashnet/
        └── data/
            └── dataset-resized/  # Dataset images
```

## Model Architecture

The app uses a ResNet18 convolutional neural network pre-trained on ImageNet, with transfer learning applied to the waste classification task:

- **Base Model:** ResNet18 (ImageNet weights)
- **Input Size:** 224x224 RGB images
- **Output:** 6 classes (glass, paper, cardboard, plastic, metal, trash)
- **Training:** Adam optimizer with learning rate scheduling

## API Endpoints

### POST /predict
Upload an image and get waste classification predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `photo` (image file)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "plastic",
    "confidence": 85.23,
    "all_probabilities": {
      "glass": 0.05,
      "paper": 0.02,
      "cardboard": 0.01,
      "plastic": 0.85,
      "metal": 0.04,
      "trash": 0.03
    }
  }
}
```

## Technologies Used

- **Backend:** Flask (Python)
- **Deep Learning:** PyTorch, Torchvision
- **Image Processing:** Pillow
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** TrashNet

## Requirements

- Python 3.8+
- PyTorch 2.7.0+
- Flask 3.1.0+
- Pillow 11.0.0+
- NumPy 2.2.5+
- scikit-learn 1.5.0+

## Notes

- The model needs to be trained before use (run `train_model.py`)
- First prediction may take longer as the model loads into memory
- For best results, use clear images of waste items on a white or neutral background
- The app supports common image formats: PNG, JPG, JPEG, GIF, WEBP

## License

This project uses the TrashNet dataset. Please refer to the [TrashNet repository](https://github.com/garythung/trashnet) for dataset licensing information.

## Acknowledgments

- TrashNet dataset by Gary Thung and Mindy Yang
- PyTorch and Torchvision teams
- Flask framework

