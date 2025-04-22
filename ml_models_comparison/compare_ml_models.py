import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import time
import json

# Add parent directory to path to import data generation
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Create a directory for results if it doesn't exist
results_dir = os.path.join("ml_models_comparison", "results")
os.makedirs(results_dir, exist_ok=True)

def load_data():
    """Load training and testing data."""
    try:
        # Try to load existing NPY data
        X_train = np.load(os.path.join(parent_dir, "data", "X_train.npy"))
        y_train = np.load(os.path.join(parent_dir, "data", "y_train.npy"))
        X_test = np.load(os.path.join(parent_dir, "data", "X_test.npy"))
        y_test = np.load(os.path.join(parent_dir, "data", "y_test.npy"))
        
        print(f"Loaded existing NPY data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError:
        print("NPY files not found. Attempting to load from CSV files...")
        
        # Try to load from CSV files
        try:
            # Look for CSV files in the data directory
            train_csv_files = [f for f in os.listdir(os.path.join(parent_dir, "data")) if f.startswith("wound_data_train") and f.endswith(".csv")]
            test_csv_files = [f for f in os.listdir(os.path.join(parent_dir, "data")) if f.startswith("wound_data_test") and f.endswith(".csv")]
            
            if not train_csv_files or not test_csv_files:
                print("No CSV files found in the data directory.")
                sys.exit(1)
                
            # Use the most recent files (assuming filenames contain dates)
            train_file = sorted(train_csv_files)[-1]
            test_file = sorted(test_csv_files)[-1]
            
            print(f"Loading from CSV files: {train_file} and {test_file}")
            
            # Load the CSV files
            train_data = pd.read_csv(os.path.join(parent_dir, "data", train_file))
            test_data = pd.read_csv(os.path.join(parent_dir, "data", test_file))
            
            # Extract features and labels
            feature_cols = ['pH', 'temperature', 'humidity', 'exudate_level', 'oxygen_saturation']
            
            X_train = train_data[feature_cols].values
            y_train = train_data['condition'].map({'healthy': 0, 'infected': 1, 'wound_healing': 2}).values
            
            X_test = test_data[feature_cols].values
            y_test = test_data['condition'].map({'healthy': 0, 'infected': 1, 'wound_healing': 2}).values
            
            print(f"Loaded data from CSV: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
            # Save as NPY for future use
            np.save(os.path.join(parent_dir, "data", "X_train.npy"), X_train)
            np.save(os.path.join(parent_dir, "data", "y_train.npy"), y_train)
            np.save(os.path.join(parent_dir, "data", "X_test.npy"), X_test)
            np.save(os.path.join(parent_dir, "data", "y_test.npy"), y_test)
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"Error loading CSV data: {str(e)}")
            sys.exit(1)

def preprocess_data(X_train, X_test):
    """Preprocess the data by scaling features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return performance metrics."""
    start_time = time.time()
    
    if model_name == "CNN":
        # For CNN, we need one-hot encoded targets for prediction
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
        
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_time": inference_time,
        "confusion_matrix": cm
    }

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier with hyperparameter tuning."""
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],  # Reduced options for faster execution
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Use GridSearchCV for hyperparameter tuning
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    training_time = time.time() - start_time
    
    print(f"Random Forest training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_rf, training_time

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier with hyperparameter tuning."""
    print("Training Logistic Regression model...")
    start_time = time.time()
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    
    # Use GridSearchCV for hyperparameter tuning
    lr = LogisticRegression(random_state=42, multi_class='multinomial')
    grid_search = GridSearchCV(lr, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_lr = grid_search.best_estimator_
    training_time = time.time() - start_time
    
    print(f"Logistic Regression training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_lr, training_time

def train_svm(X_train, y_train):
    """Train an SVM classifier with hyperparameter tuning."""
    print("Training SVM model...")
    start_time = time.time()
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    # Use GridSearchCV for hyperparameter tuning
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm = grid_search.best_estimator_
    training_time = time.time() - start_time
    
    print(f"SVM training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_svm, training_time

def train_cnn(X_train, y_train):
    """Train a simple CNN (actually a simple neural network) classifier."""
    print("Training CNN model...")
    start_time = time.time()
    
    # One-hot encode the target variable
    y_train_categorical = to_categorical(y_train)
    
    # Get the number of features and classes
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Create a simple neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train, 
        y_train_categorical,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"CNN training completed in {training_time:.2f} seconds")
    
    return model, training_time, history

def plot_comparison(results):
    """Plot comparison of model performance metrics."""
    # Extract metrics for plotting
    model_names = [result["model_name"] for result in results]
    accuracies = [result["accuracy"] for result in results]
    precisions = [result["precision"] for result in results]
    recalls = [result["recall"] for result in results]
    f1_scores = [result["f1_score"] for result in results]
    inference_times = [result["inference_time"] * 1000 for result in results]  # Convert to ms
    
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot accuracy
    axs[0, 0].bar(model_names, accuracies, color='skyblue')
    axs[0, 0].set_ylim(0.8, 1.0)  # Assuming accuracy is high
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Model Accuracy Comparison')
    
    # Plot precision, recall, and F1 score
    bar_width = 0.25
    r1 = np.arange(len(model_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    axs[0, 1].bar(r1, precisions, color='lightgreen', width=bar_width, label='Precision')
    axs[0, 1].bar(r2, recalls, color='salmon', width=bar_width, label='Recall')
    axs[0, 1].bar(r3, f1_scores, color='gold', width=bar_width, label='F1 Score')
    axs[0, 1].set_xticks([r + bar_width for r in range(len(model_names))])
    axs[0, 1].set_xticklabels(model_names)
    axs[0, 1].set_ylim(0.8, 1.0)  # Assuming metrics are high
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].set_title('Precision, Recall, and F1 Score Comparison')
    axs[0, 1].legend()
    
    # Plot inference time (logarithmic scale for better visualization)
    axs[1, 0].bar(model_names, inference_times, color='purple')
    axs[1, 0].set_ylabel('Inference Time (ms per sample)')
    axs[1, 0].set_title('Inference Time Comparison')
    
    # Plot confusion matrices for all models
    axs[1, 1].axis('off')
    axs[1, 1].text(0.5, 0.5, 'Confusion matrices saved separately', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison.png"))
    
    # Plot confusion matrices separately
    class_names = ['Healthy', 'Infected', 'Healing']
    for result in results:
        plt.figure(figsize=(8, 6))
        sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                  xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {result["model_name"]}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrix_{result['model_name']}.png"))
        plt.close()

def save_results(results, training_times):
    """Save all results to a JSON file."""
    # Convert NumPy arrays to lists for JSON serialization
    for result in results:
        result["confusion_matrix"] = result["confusion_matrix"].tolist()
    
    # Combine results and training times
    full_results = {
        "model_metrics": results,
        "training_times": training_times
    }
    
    # Save to file
    with open(os.path.join(results_dir, "model_comparison_results.json"), 'w') as f:
        json.dump(full_results, f, indent=4)
    
    # Also save as readable text file
    with open(os.path.join(results_dir, "model_comparison_summary.txt"), 'w') as f:
        f.write("ML Models Comparison for Smart Bandage Wound Classification\n")
        f.write("="*70 + "\n\n")
        
        # Write overall comparison
        f.write("Overall Performance Comparison:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Inference Time (ms)':<20} {'Training Time (s)':<20}\n")
        for i, result in enumerate(results):
            model_name = result["model_name"]
            f.write(f"{model_name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1_score']:<10.4f} {result['inference_time']*1000:<20.4f} {training_times[model_name]:<20.4f}\n")
        
        f.write("\n\nDetailed Results by Model:\n")
        f.write("="*70 + "\n\n")
        
        # Write detailed results for each model
        for result in results:
            model_name = result["model_name"]
            f.write(f"Model: {model_name}\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"Inference Time: {result['inference_time']*1000:.4f} ms per sample\n")
            f.write(f"Training Time: {training_times[model_name]:.4f} seconds\n\n")
            
            f.write("Confusion Matrix:\n")
            cm = np.array(result["confusion_matrix"])
            class_names = ['Healthy', 'Infected', 'Healing']
            f.write(f"{'':<10} {class_names[0]:<10} {class_names[1]:<10} {class_names[2]:<10}\n")
            for i, row in enumerate(cm):
                f.write(f"{class_names[i]:<10} {row[0]:<10} {row[1]:<10} {row[2]:<10}\n")
            f.write("\n\n")

def main():
    print("="*50)
    print("ML Models Comparison for Smart Bandage")
    print("="*50)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Preprocess data
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Train models
    print("\nTraining models...\n")
    training_times = {}
    
    # Random Forest
    rf_model, rf_time = train_random_forest(X_train_scaled, y_train)
    training_times["Random Forest"] = rf_time
    
    # Logistic Regression
    lr_model, lr_time = train_logistic_regression(X_train_scaled, y_train)
    training_times["Logistic Regression"] = lr_time
    
    # SVM
    svm_model, svm_time = train_svm(X_train_scaled, y_train)
    training_times["SVM"] = svm_time
    
    # CNN
    # Convert targets to one-hot encoding for CNN
    cnn_model, cnn_time, history = train_cnn(X_train_scaled, y_train)
    training_times["CNN"] = cnn_time
    
    # Evaluate models
    print("\nEvaluating models...\n")
    results = []
    
    # Evaluate Random Forest
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    results.append(rf_results)
    
    # Evaluate Logistic Regression
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    results.append(lr_results)
    
    # Evaluate SVM
    svm_results = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
    results.append(svm_results)
    
    # Evaluate CNN
    # For CNN evaluation, we need to convert targets to categorical
    y_test_categorical = to_categorical(y_test)
    cnn_results = evaluate_model(cnn_model, X_test_scaled, y_test, "CNN")
    results.append(cnn_results)
    
    # Save results
    print("\nSaving results...\n")
    save_results(results, training_times)
    
    # Plot comparison
    print("Generating visualizations...\n")
    plot_comparison(results)
    
    # Save CNN model history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('CNN Training History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "cnn_training_history.png"))
    
    print(f"\nResults saved to {results_dir}")
    print("="*50)

if __name__ == "__main__":
    main() 