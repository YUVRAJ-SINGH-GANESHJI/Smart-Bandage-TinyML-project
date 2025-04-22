import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(data_path='data'):
    """
    Load the most recent training and testing datasets
    """
    # Get the most recent training and test files
    train_files = [f for f in os.listdir(data_path) if f.startswith('wound_data_train')]
    test_files = [f for f in os.listdir(data_path) if f.startswith('wound_data_test')]
    
    if not train_files or not test_files:
        raise FileNotFoundError("Dataset files not found. Run data_generation.py first.")
    
    # Sort by timestamp (most recent first)
    train_files.sort(reverse=True)
    test_files.sort(reverse=True)
    
    train_data = pd.read_csv(os.path.join(data_path, train_files[0]))
    test_data = pd.read_csv(os.path.join(data_path, test_files[0]))
    
    print(f"Loaded training data: {train_files[0]}")
    print(f"Loaded test data: {test_files[0]}")
    
    return train_data, test_data

def preprocess_data(train_df, test_df):
    """
    Preprocess the data for training
    """
    # Separate features and target
    X_train = train_df.drop('condition', axis=1)
    y_train = train_df['condition']
    X_test = test_df.drop('condition', axis=1)
    y_test = test_df['condition']
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier with hyperparameter tuning
    """
    print("Training Random Forest classifier...")
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save the confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/confusion_matrix_{timestamp}.png')
    print(f"Confusion matrix saved to results/confusion_matrix_{timestamp}.png")
    
    return accuracy, y_pred

def feature_importance(model, feature_names):
    """
    Plot and save feature importance
    """
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/feature_importance_{timestamp}.png')
    print(f"Feature importance plot saved to results/feature_importance_{timestamp}.png")

def save_model(model, scaler):
    """
    Save the trained model and scaler
    """
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'models/random_forest_model_{timestamp}.joblib'
    scaler_filename = f'models/scaler_{timestamp}.joblib'
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")
    
    return model_filename, scaler_filename

def main():
    """
    Main function to train and evaluate the model
    """
    # Load data
    train_data, test_data = load_data()
    
    # Preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(train_data, test_data)
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    feature_importance(model, X_train.columns)
    
    # Save model
    save_model(model, scaler)

if __name__ == "__main__":
    main() 