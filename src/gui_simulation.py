import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Try to import tkinter
try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("WARNING: tkinter not available. Running in headless mode.")

class WoundClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Bandage Wound Classifier")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Load the most recent model and scaler
        self.load_model()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        title_label = ttk.Label(self.main_frame, 
                               text="Smart Bandage Wound Classification Simulator", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create the parameter frame
        self.param_frame = ttk.LabelFrame(self.main_frame, text="Wound Parameters", padding="10")
        self.param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create parameter sliders
        self.create_sliders()
        
        # Create result frame
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Classification Results", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create visualization
        self.create_visualization()
        
        # Create classification result label
        self.result_label = ttk.Label(self.result_frame, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        # Create probability distribution label
        self.prob_label = ttk.Label(self.result_frame, text="", font=("Arial", 12))
        self.prob_label.pack(pady=5)
        
        # Update the classification
        self.update_classification()
        
    def load_model(self):
        """Load the most recent model and scaler"""
        model_dir = 'models'
        
        # Get the most recent model and scaler files
        model_files = [f for f in os.listdir(model_dir) if f.startswith('random_forest_model_')]
        scaler_files = [f for f in os.listdir(model_dir) if f.startswith('scaler_')]
        
        if not model_files or not scaler_files:
            raise FileNotFoundError("Model files not found. Run model_training.py first.")
        
        # Sort by timestamp (most recent first)
        model_files.sort(reverse=True)
        scaler_files.sort(reverse=True)
        
        # Load the model and scaler
        self.model = joblib.load(os.path.join(model_dir, model_files[0]))
        self.scaler = joblib.load(os.path.join(model_dir, scaler_files[0]))
        
        print(f"Loaded model: {model_files[0]}")
        print(f"Loaded scaler: {scaler_files[0]}")
    
    def create_sliders(self):
        """Create sliders for each parameter"""
        # Define the parameters and their ranges based on medical literature
        self.parameters = {
            'pH': (5.0, 9.5, 7.4),
            'temperature': (30.0, 38.0, 33.0),
            'humidity': (30.0, 100.0, 60.0),
            'exudate_level': (0.0, 10.0, 5.0),
            'oxygen_saturation': (50.0, 100.0, 80.0)
        }
        
        # Create sliders for each parameter
        self.sliders = {}
        self.slider_values = {}
        
        # Create a frame for each row
        for i, (param, (min_val, max_val, default)) in enumerate(self.parameters.items()):
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=5)
            
            # Create label
            label = ttk.Label(frame, text=f"{param}:", width=20)
            label.pack(side=tk.LEFT, padx=5)
            
            # Create slider
            slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                              length=400, value=default,
                              command=lambda val, param=param: self.update_slider_value(param, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Create value label
            value_var = tk.StringVar(value=f"{default:.2f}")
            value_label = ttk.Label(frame, textvariable=value_var, width=10)
            value_label.pack(side=tk.LEFT, padx=5)
            
            # Store references
            self.sliders[param] = slider
            self.slider_values[param] = value_var
            
            # Set initial value
            self.update_slider_value(param, default)
    
    def update_slider_value(self, param, value):
        """Update the slider value display and trigger classification update"""
        # Update the display value
        value = float(value)
        self.slider_values[param].set(f"{value:.2f}")
        
        # Schedule an update to the classification
        self.root.after(100, self.update_classification)
    
    def create_visualization(self):
        """Create visualization panels"""
        # Create frame for visualization
        viz_frame = ttk.Frame(self.result_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create figure for probability chart
        self.fig = Figure(figsize=(9, 4), dpi=100)
        
        # Create probability chart
        self.prob_ax = self.fig.add_subplot(121)
        self.prob_ax.set_title('Classification Probabilities')
        self.prob_bars = self.prob_ax.bar(['Healthy', 'Infected', 'Healing'], [0, 0, 0])
        self.prob_ax.set_ylim(0, 1)
        self.prob_ax.set_ylabel('Probability')
        
        # Create feature importance chart
        self.imp_ax = self.fig.add_subplot(122)
        self.imp_ax.set_title('Feature Contribution')
        self.feature_bars = self.imp_ax.barh(list(self.parameters.keys()), [0] * len(self.parameters))
        self.imp_ax.set_xlim(-2, 2)
        
        # Add figure to canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Adjust spacing
        self.fig.tight_layout()
    
    def update_classification(self):
        """Update the classification based on current slider values"""
        # Get the current parameter values
        params = {}
        for param in self.parameters:
            params[param] = float(self.slider_values[param].get())
        
        # Create a DataFrame with the current values
        df = pd.DataFrame([params])
        
        # Scale the data
        X_scaled = self.scaler.transform(df)
        
        # Get prediction and probabilities
        pred_class = self.model.predict(X_scaled)[0]
        pred_probs = self.model.predict_proba(X_scaled)[0]
        
        # Update classification result
        classes = {'healthy': 'Healthy', 'infected': 'Infected', 'wound_healing': 'Healing'}
        self.result_label.config(text=f"Predicted Wound Condition: {classes[pred_class]}", 
                               foreground=self.get_color_for_class(pred_class))
        
        # Update probability display
        probs_text = " | ".join([f"{classes[c]}: {p:.2f}" for c, p in 
                               zip(self.model.classes_, pred_probs)])
        self.prob_label.config(text=f"Probabilities: {probs_text}")
        
        # Update probability chart
        for i, (bar, prob) in enumerate(zip(self.prob_bars, pred_probs)):
            bar.set_height(prob)
            bar.set_color(self.get_color_for_index(i))
        
        # Compute feature contributions
        feature_contribs = self.compute_feature_contribution(X_scaled[0])
        
        # Update feature importance chart
        feature_names = list(params.keys())
        for i, (bar, contrib) in enumerate(zip(self.feature_bars, feature_contribs)):
            bar.set_width(contrib)
            bar.set_color(self.get_contrib_color(contrib))
        
        # Redraw the canvas
        self.canvas.draw()
    
    def get_color_for_class(self, class_name):
        """Get color for classification result"""
        colors = {
            'healthy': 'green',
            'infected': 'red',
            'wound_healing': 'orange'
        }
        return colors.get(class_name, 'black')
    
    def get_color_for_index(self, index):
        """Get color for classification result by index"""
        colors = ['green', 'red', 'orange']
        return colors[index]
    
    def get_contrib_color(self, value):
        """Get color based on feature contribution"""
        if value > 0:
            return 'green'
        else:
            return 'red'
    
    def compute_feature_contribution(self, X_scaled):
        """
        Compute simplified feature contributions for visualization
        """
        # This is a simplified approach - for a real system, SHAP values would be more accurate
        # Get feature importances from the model
        importances = self.model.feature_importances_
        
        # Scale the feature values to show their contribution direction
        # Positive values push toward the prediction, negative away from it
        contributions = []
        
        # Normalize scaled values
        X_norm = (X_scaled - np.mean(X_scaled)) / (np.std(X_scaled) or 1)
        
        # Multiply normalized values by importance
        for x, imp in zip(X_norm, importances):
            # Scale for visualization
            contrib = x * imp * 2
            contributions.append(contrib)
        
        return contributions

def run_headless_test():
    """Run a headless test of the classifier with sample values"""
    print("Running headless test of wound classifier...")
    
    # Find the latest model and scaler
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('random_forest_model_')]
    scaler_files = [f for f in os.listdir(model_dir) if f.startswith('scaler_')]
    
    if not model_files or not scaler_files:
        print("Model files not found. Run model_training.py first.")
        return
    
    # Sort by timestamp (most recent first)
    model_files.sort(reverse=True)
    scaler_files.sort(reverse=True)
    
    # Load the model and scaler
    model = joblib.load(os.path.join(model_dir, model_files[0]))
    scaler = joblib.load(os.path.join(model_dir, scaler_files[0]))
    
    print(f"Loaded model: {model_files[0]}")
    print(f"Loaded scaler: {scaler_files[0]}")
    
    # Test with sample values for each class
    test_samples = [
        # Healthy sample
        {'pH': 7.4, 'temperature': 32.0, 'humidity': 50.0, 'exudate_level': 2.0, 'oxygen_saturation': 90.0},
        # Infected sample
        {'pH': 8.5, 'temperature': 35.0, 'humidity': 80.0, 'exudate_level': 8.0, 'oxygen_saturation': 70.0},
        # Healing sample
        {'pH': 7.2, 'temperature': 33.0, 'humidity': 60.0, 'exudate_level': 4.0, 'oxygen_saturation': 85.0}
    ]
    
    for i, sample in enumerate(test_samples):
        print(f"\nTest Sample {i+1}:")
        for param, value in sample.items():
            print(f"  {param}: {value:.2f}")
        
        # Create DataFrame from sample
        df = pd.DataFrame([sample])
        
        # Scale the data
        X_scaled = scaler.transform(df)
        
        # Get prediction and probabilities
        pred_class = model.predict(X_scaled)[0]
        pred_probs = model.predict_proba(X_scaled)[0]
        
        # Print prediction
        print(f"  Predicted class: {pred_class}")
        print(f"  Probabilities: {dict(zip(model.classes_, pred_probs))}")
    
    # Create and save a visualization of the test samples
    create_sample_visualization(test_samples, model, scaler)
    
def create_sample_visualization(samples, model, scaler):
    """Create and save a visualization of test samples"""
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Set up colors and markers
    colors = {'healthy': 'green', 'infected': 'red', 'wound_healing': 'orange'}
    
    # Plot the samples
    for i, sample in enumerate(samples):
        df = pd.DataFrame([sample])
        X_scaled = scaler.transform(df)
        pred_class = model.predict(X_scaled)[0]
        
        # Plot in 2D space (pH vs temperature)
        plt.scatter(sample['pH'], sample['temperature'], 
                   color=colors[pred_class], 
                   s=100, 
                   label=f"Sample {i+1}: {pred_class}")
    
    # Add labels and title
    plt.xlabel('pH')
    plt.ylabel('Temperature (°C)')
    plt.title('Wound Classification Test Samples')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/test_samples.png')
    print("\nSample visualization saved to results/test_samples.png")

def main():
    """Main function to run the app"""
    if TKINTER_AVAILABLE:
        # Create the root window
        root = tk.Tk()
        
        # Create the app
        app = WoundClassifierApp(root)
        
        # Start the main loop
        root.mainloop()
    else:
        # Run headless test
        run_headless_test()

if __name__ == "__main__":
    main() 