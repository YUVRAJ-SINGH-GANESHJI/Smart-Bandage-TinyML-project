import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from datetime import datetime

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

def create_output_dir():
    """Create output directory for visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/visualizations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}/")
    return output_dir

def plot_parameter_distributions(data, output_dir):
    """Plot histograms of parameter distributions by condition"""
    features = [col for col in data.columns if col != 'condition']
    conditions = data['condition'].unique()
    colors = {'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'}
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        for condition in conditions:
            subset = data[data['condition'] == condition]
            plt.hist(subset[feature], alpha=0.7, label=condition, color=colors[condition], bins=15)
        
        plt.title(f'Distribution of {feature} by Wound Condition', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"{feature}_distribution.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}")

def plot_correlation_matrix(data, output_dir):
    """Plot correlation matrix of parameters"""
    features = [col for col in data.columns if col != 'condition']
    
    # Overall correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = data[features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, mask=mask,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title("Parameter Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    print("Saved correlation_matrix.png")
    
    # Correlation matrices by condition
    for condition in data['condition'].unique():
        subset = data[data['condition'] == condition]
        
        plt.figure(figsize=(12, 10))
        corr_matrix = subset[features].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(f"Parameter Correlation Matrix - {condition.capitalize()} Wounds", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"correlation_matrix_{condition}.png"), dpi=300)
        plt.close()
        print(f"Saved correlation_matrix_{condition}.png")

def plot_pca_visualization(data, output_dir):
    """Create PCA visualization of the dataset"""
    # Prepare data
    X = data.drop('condition', axis=1)
    y = data['condition']
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'condition': y
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    colors = {'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'}
    
    for condition, color in colors.items():
        subset = pca_df[pca_df['condition'] == condition]
        plt.scatter(subset['PCA1'], subset['PCA2'], 
                   c=color, label=condition, alpha=0.7, edgecolors='w', s=70)
    
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1f}%)', fontsize=12)
    plt.title('PCA Visualization of Wound Parameters', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "pca_visualization.png"), dpi=300)
    plt.close()
    print("Saved pca_visualization.png")
    
    # Create feature contribution plot
    plt.figure(figsize=(10, 8))
    features = X.columns
    
    # Plot feature loadings
    for i, feature in enumerate(features):
        plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        plt.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, 
                feature, fontsize=12)
    
    plt.grid(alpha=0.3)
    circle = plt.Circle((0, 0), 1, fc='white', ec='black', alpha=0.2)
    plt.gca().add_patch(circle)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title('PCA Feature Contributions', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "pca_feature_contributions.png"), dpi=300)
    plt.close()
    print("Saved pca_feature_contributions.png")

def plot_parameter_boxplots(data, output_dir):
    """Create boxplots of parameters by condition"""
    features = [col for col in data.columns if col != 'condition']
    
    # Set colors for boxplots
    colors = {'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'}
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # Create boxplot
        boxplot = sns.boxplot(x='condition', y=feature, data=data, 
                            palette=colors, width=0.5)
        
        # Add swarmplot for individual points
        sns.swarmplot(x='condition', y=feature, data=data, 
                     color='black', alpha=0.5, size=4)
        
        # Customize plot
        plt.title(f'{feature} by Wound Condition', fontsize=14)
        plt.xlabel('Wound Condition', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"{feature}_boxplot.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}")

def plot_parameter_violin(data, output_dir):
    """Create violin plots of parameters by condition"""
    features = [col for col in data.columns if col != 'condition']
    
    # Set colors for violinplots
    colors = {'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'}
    
    # Create one combined figure with subplots
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4 * len(features)))
    
    for i, feature in enumerate(features):
        # Create violinplot on the corresponding subplot
        sns.violinplot(x='condition', y=feature, data=data, 
                      palette=colors, ax=axes[i], inner="quartile")
        
        # Customize subplot
        axes[i].set_title(f'{feature} Distribution by Wound Condition', fontsize=14)
        axes[i].set_xlabel('')  # Remove x label for all but the last subplot
        axes[i].set_ylabel(feature, fontsize=12)
        axes[i].grid(axis='y', alpha=0.3)
    
    # Add general x-label for the bottom subplot
    axes[-1].set_xlabel('Wound Condition', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_violins.png"), dpi=300)
    plt.close()
    print("Saved parameter_violins.png")

def plot_pairwise_relationships(data, output_dir):
    """Create pairplot to show pairwise relationships between parameters"""
    # Subsample data for faster plotting if dataset is large
    if len(data) > 500:
        data_sample = data.sample(n=500, random_state=42)
    else:
        data_sample = data
    
    # Create pairplot
    plt.figure(figsize=(15, 15))
    g = sns.pairplot(data_sample, hue='condition', 
                    palette={'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'},
                    diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'},
                    height=2.5)
    
    # Customize plot
    g.fig.suptitle('Pairwise Relationships Between Parameters', fontsize=16, y=1.02)
    
    plt.savefig(os.path.join(output_dir, "pairwise_relationships.png"), dpi=300)
    plt.close()
    print("Saved pairwise_relationships.png")

def main():
    """Main function to generate all visualizations"""
    # Load data
    train_data, test_data = load_latest_data()
    
    # Combine datasets for more comprehensive visualizations
    all_data = pd.concat([train_data, test_data])
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Generate visualizations
    plot_parameter_distributions(all_data, output_dir)
    plot_correlation_matrix(all_data, output_dir)
    plot_pca_visualization(all_data, output_dir)
    plot_parameter_boxplots(all_data, output_dir)
    plot_parameter_violin(all_data, output_dir)
    plot_pairwise_relationships(all_data, output_dir)
    
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main() 