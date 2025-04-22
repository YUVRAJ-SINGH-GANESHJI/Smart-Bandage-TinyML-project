import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from datetime import datetime
from matplotlib.colors import ListedColormap
import itertools

def load_latest_data(data_dir='data'):
    """Load the most recent train and test datasets"""
    # Find the most recent files
    train_files = [f for f in os.listdir(data_dir) if f.startswith('wound_data_train_')]
    test_files = [f for f in os.listdir(data_dir) if f.startswith('wound_data_test_')]
    
    if not train_files or not test_files:
        raise FileNotFoundError("No dataset files found. Run data_generation.py first.")
    
    # Sort by timestamp (most recent first)
    train_files.sort(reverse=True)
    test_files.sort(reverse=True)
    
    # Load the data
    train_path = os.path.join(data_dir, train_files[0])
    test_path = os.path.join(data_dir, test_files[0])
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Loaded training data: {train_files[0]} ({len(train_data)} samples)")
    print(f"Loaded test data: {test_files[0]} ({len(test_data)} samples)")
    
    return train_data, test_data

def load_latest_model(model_dir='models'):
    """Load the most recent model file"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    if not model_files:
        raise FileNotFoundError("No model files found. Train the model first.")
    
    # Sort by timestamp (most recent first)
    model_files.sort(reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    # Load model
    model = joblib.load(model_path)
    print(f"Loaded model: {model_files[0]}")
    
    return model, model_path

def create_output_dir():
    """Create output directory for visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/model_visualizations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}/")
    return output_dir

def plot_confusion_matrix(model, X_test, y_test, output_dir):
    """Plot confusion matrix for model predictions"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    classes = np.unique(y_test)
    
    # Plot with numbers and percentages
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    
    # Add normalized confusion matrix (percent)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Add percentages as annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j + 0.5, i + 0.85, f"({cm_norm[i, j]:.1%})", 
                 horizontalalignment="center", fontsize=9, 
                 color="black" if cm_norm[i, j] < 0.7 else "white")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print("Saved confusion_matrix.png")

def plot_feature_importance(model, feature_names, output_dir):
    """Plot feature importance for the model"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.title('Feature Importances (MDI)', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        
        plt.savefig(os.path.join(output_dir, "feature_importance_mdi.png"), dpi=300)
        plt.close()
        print("Saved feature_importance_mdi.png")
    else:
        print("Model does not have feature_importances_ attribute. Skipping MDI plot.")

def plot_permutation_importance(model, X_test, y_test, feature_names, output_dir):
    """Plot permutation importance for the model"""
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(perm_importance.importances[sorted_idx].T,
               vert=False, labels=[feature_names[i] for i in sorted_idx])
    plt.title("Permutation Importance (Test Set)", fontsize=14)
    plt.xlabel('Decrease in Model Accuracy', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "permutation_importance.png"), dpi=300)
    plt.close()
    print("Saved permutation_importance.png")

def plot_roc_curves(model, X_test, y_test, output_dir):
    """Plot ROC curves for each class (one-vs-rest)"""
    # Get class probabilities
    y_probs = model.predict_proba(X_test)
    
    # Get unique classes
    classes = model.classes_
    n_classes = len(classes)
    
    # Convert y_test to one-hot encoding for ROC calculation
    y_test_encoded = pd.get_dummies(y_test).values
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot for each class
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_encoded[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line (random guess)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()
    print("Saved roc_curves.png")

def plot_precision_recall_curves(model, X_test, y_test, output_dir):
    """Plot precision-recall curves for each class"""
    # Get class probabilities
    y_probs = model.predict_proba(X_test)
    
    # Get unique classes
    classes = model.classes_
    n_classes = len(classes)
    
    # Convert y_test to one-hot encoding for PR calculation
    y_test_encoded = pd.get_dummies(y_test).values
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot for each class
    for i, label in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_encoded[:, i], y_probs[:, i])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, lw=2, label=f'{label} (Avg Precision = {avg_precision:.3f})')
    
    # Customize plot
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()
    print("Saved precision_recall_curves.png")

def plot_decision_boundaries(model, X_test, y_test, output_dir, features_to_plot=None):
    """Plot decision boundaries for pairs of features"""
    # If specific features not provided, use the most important ones
    if features_to_plot is None and hasattr(model, 'feature_importances_'):
        # Get the two most important features
        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[::-1][:2]
    else:
        # Use first two features if no importances available
        feature_indices = [0, 1] if features_to_plot is None else features_to_plot
    
    # Get feature names
    feature_names = X_test.columns
    
    # Get the selected features
    X_subset = X_test.iloc[:, feature_indices].values
    feature1_name = feature_names[feature_indices[0]]
    feature2_name = feature_names[feature_indices[1]]
    
    # Create mesh grid for decision boundary plotting
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict on the mesh grid
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    # Create a copy of a sample from X_test to use as a template
    X_pred_template = X_test.iloc[0].copy()
    
    # For each point in the mesh, create a full feature vector
    Z = []
    for point in X_mesh:
        X_full = X_pred_template.copy()
        X_full[feature1_name] = point[0]
        X_full[feature2_name] = point[1]
        # Reshape to match model expectation (1, n_features)
        X_full = X_full.values.reshape(1, -1)
        Z.append(model.predict(X_full)[0])
    
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for decision regions
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot training points
    scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_test, cmap=cmap, 
                edgecolor='k', s=100, alpha=0.7)
    
    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=model.classes_)
    
    # Customize plot
    plt.xlabel(feature1_name, fontsize=12)
    plt.ylabel(feature2_name, fontsize=12)
    plt.title(f'Decision Boundary using {feature1_name} and {feature2_name}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "decision_boundary.png"), dpi=300)
    plt.close()
    print("Saved decision_boundary.png")
    
    # Plot additional feature pairs if model has feature importances
    if hasattr(model, 'feature_importances_') and len(feature_names) >= 4:
        # Get top 4 features
        top_features = np.argsort(model.feature_importances_)[::-1][:4]
        
        # Plot pairwise decision boundaries for top features
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                # Skip if this is the same as the first pair we already plotted
                if set([feat1, feat2]) == set(feature_indices):
                    continue
                    
                # Plot decision boundary for this feature pair
                plot_feature_pair_decision_boundary(
                    model, X_test, y_test, feat1, feat2, 
                    feature_names, output_dir
                )

def plot_feature_pair_decision_boundary(model, X_test, y_test, feat1_idx, feat2_idx, feature_names, output_dir):
    """Helper function to plot decision boundary for a specific feature pair"""
    # Get feature names
    feature1_name = feature_names[feat1_idx]
    feature2_name = feature_names[feat2_idx]
    
    # Get the selected features
    X_subset = X_test.iloc[:, [feat1_idx, feat2_idx]].values
    
    # Create mesh grid for decision boundary plotting
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
    
    # Predict on the mesh grid
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    # Create a copy of a sample from X_test to use as a template
    X_pred_template = X_test.iloc[0].copy()
    
    # For each point in the mesh, create a full feature vector
    Z = []
    for point in X_mesh:
        X_full = X_pred_template.copy()
        X_full[feature1_name] = point[0]
        X_full[feature2_name] = point[1]
        # Reshape to match model expectation (1, n_features)
        X_full = X_full.values.reshape(1, -1)
        Z.append(model.predict(X_full)[0])
    
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for decision regions
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot training points
    scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_test, cmap=cmap, 
                edgecolor='k', s=100, alpha=0.7)
    
    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=model.classes_)
    
    # Customize plot
    plt.xlabel(feature1_name, fontsize=12)
    plt.ylabel(feature2_name, fontsize=12)
    plt.title(f'Decision Boundary using {feature1_name} and {feature2_name}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f"decision_boundary_{feature1_name}_{feature2_name}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

def plot_probability_distributions(model, X_test, y_test, output_dir):
    """Plot probability distributions of model predictions"""
    # Get class probabilities
    y_probs = model.predict_proba(X_test)
    classes = model.classes_
    
    # Create DataFrame for easier plotting
    probs_df = pd.DataFrame(y_probs, columns=classes)
    probs_df['true_label'] = y_test.values
    
    # Melt the DataFrame for seaborn
    melted_df = pd.melt(probs_df, id_vars=['true_label'], value_vars=classes,
                       var_name='predicted_class', value_name='probability')
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Create violin plots of probability distributions
    ax = sns.violinplot(x='true_label', y='probability', hue='predicted_class', 
                     data=melted_df, split=True, inner="quart", palette="Set2")
    
    # Customize plot
    plt.title('Prediction Probability Distributions by True Label', fontsize=14)
    plt.xlabel('True Label', fontsize=12)
    plt.ylabel('Prediction Probability', fontsize=12)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Predicted Class')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "probability_distributions.png"), dpi=300)
    plt.close()
    print("Saved probability_distributions.png")
    
    # Plot probability calibration histograms
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4 * len(classes)))
    
    # For each class, plot histogram of prediction probabilities
    for i, cls in enumerate(classes):
        # Get samples where this is the true class
        mask = y_test == cls
        if mask.sum() > 0:  # Only plot if we have samples
            # Get probabilities for this class
            probs = y_probs[mask, i]
            
            # Plot histogram
            if len(classes) > 1:
                ax = axes[i]
            else:
                ax = axes
                
            ax.hist(probs, bins=20, alpha=0.8, color=f'C{i}')
            ax.set_title(f'Prediction Probability Distribution for True {cls} Class', fontsize=12)
            ax.set_xlabel('Predicted Probability', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_xlim([0, 1])
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_calibration.png"), dpi=300)
    plt.close()
    print("Saved probability_calibration.png")

def main():
    """Main function to generate all model visualizations"""
    # Create output directory
    output_dir = create_output_dir()
    
    # Load data
    train_data, test_data = load_latest_data()
    
    # Load model
    try:
        model, model_path = load_latest_model()
        
        # Prepare data
        X_train = train_data.drop('condition', axis=1)
        y_train = train_data['condition']
        X_test = test_data.drop('condition', axis=1)
        y_test = test_data['condition']
        
        # Generate visualizations
        plot_confusion_matrix(model, X_test, y_test, output_dir)
        plot_feature_importance(model, X_train.columns, output_dir)
        plot_permutation_importance(model, X_test, y_test, X_test.columns, output_dir)
        plot_roc_curves(model, X_test, y_test, output_dir)
        plot_precision_recall_curves(model, X_test, y_test, output_dir)
        plot_decision_boundaries(model, X_test, y_test, output_dir)
        plot_probability_distributions(model, X_test, y_test, output_dir)
        
        print("\nAll model visualizations generated successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please train the model first.")

if __name__ == "__main__":
    main() 