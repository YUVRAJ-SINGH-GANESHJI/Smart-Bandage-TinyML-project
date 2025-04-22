# ML Models Comparison for Smart Bandage

This folder contains a comparison of different machine learning models for wound classification using the smart bandage sensor data.

## Models Compared

The script compares the following machine learning models:

1. **Random Forest** - The current model used in the project
2. **Logistic Regression** - A simple linear model for classification
3. **Support Vector Machine (SVM)** - A powerful model for non-linear classification
4. **Convolutional Neural Network (CNN)** - A simple neural network model

## Metrics Evaluated

For each model, the following metrics are evaluated:

- **Accuracy** - The proportion of correct predictions
- **Precision** - The proportion of positive identifications that were actually correct
- **Recall** - The proportion of actual positives that were identified correctly
- **F1 Score** - The harmonic mean of precision and recall
- **Inference Time** - Average time required for a single prediction
- **Training Time** - Time required to train the model

## Visualization Outputs

The script generates several visualizations in the `results` directory:

- **model_comparison.png** - Bar charts comparing all models across different metrics
- **confusion_matrix_{model_name}.png** - Individual confusion matrices for each model
- **cnn_training_history.png** - Training and validation accuracy over epochs for the CNN model

## Results

The results are saved in two formats:

1. **model_comparison_results.json** - JSON file containing all detailed metrics
2. **model_comparison_summary.txt** - Human-readable text file with performance summary

## Usage

To run the comparison:

```bash
cd ml_models_comparison
python compare_ml_models.py
```

Note: The script will automatically load existing data from the `data` directory or generate new data if needed.

## Requirements

In addition to the project's base requirements, this script requires:
- tensorflow
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy 