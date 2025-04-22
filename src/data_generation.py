import pandas as pd
import numpy as np
import os
from datetime import datetime

# References:
# 1. Percival et al. (2014). Wound healing and biofilm - consensus document. Journal of Wound Care, 23(3).
# 2. Gethin, G. (2007). The significance of surface pH in chronic wounds. Wounds UK, 3(3), 52-56.
# 3. Power, G. et al. (2017). Evaluating moisture in wound healing. Journal of Wound Care, 26(11).
# 4. Nakagami, G. et al. (2010). Predicting delayed pressure ulcer healing using thermography. Journal of Wound Care, 19(11).
# 5. Schreml, S. et al. (2010). Oxygen in acute and chronic wound healing. British Journal of Dermatology, 163(2).

def generate_dataset(n_samples=500, random_state=42):
    """
    Generate synthetic dataset for wound classification based on medical literature.
    
    Parameters:
    - n_samples: Number of samples to generate
    - random_state: Random seed for reproducibility
    
    Returns:
    - DataFrame with wound parameters and classification
    """
    np.random.seed(random_state)
    
    # Class distribution (slightly imbalanced, as in real-world scenarios)
    classes = ['healthy', 'infected', 'wound_healing']
    class_probs = [0.3, 0.4, 0.3]  # 30% healthy, 40% infected, 30% healing
    
    # Generate classifications
    classifications = np.random.choice(classes, size=n_samples, p=class_probs)
    
    # Parameters based on medical literature
    # pH:
    # - Healthy wounds: 7.0-7.8 (neutral to slightly alkaline)
    # - Infected wounds: 7.8-9.0 (alkaline)
    # - Healing wounds: 6.5-8.0 (transitions from alkaline to neutral)
    
    # Temperature (°C):
    # - Healthy wounds: 31-33 (normal skin temperature)
    # - Infected wounds: 33-37 (elevated due to inflammation)
    # - Healing wounds: 32-34 (slightly elevated compared to healthy)
    
    # Humidity (%):
    # - Healthy wounds: 40-60 (balanced moisture)
    # - Infected wounds: 60-90 (excessive moisture promotes bacterial growth)
    # - Healing wounds: 50-70 (moist but not overly wet)
    
    # Exudate level (scale 0-10):
    # - Healthy wounds: 1-3 (minimal)
    # - Infected wounds: 6-10 (high)
    # - Healing wounds: 3-6 (moderate, decreasing over time)
    
    # Oxygen saturation (%):
    # - Healthy wounds: 85-95 (good oxygenation)
    # - Infected wounds: 60-80 (reduced due to increased metabolic demands)
    # - Healing wounds: 75-90 (improving)
    
    # Initialize the parameter arrays
    ph_values = []
    temperature_values = []
    humidity_values = []
    exudate_values = []
    oxygen_values = []
    
    # First, generate initial values for each condition (before correlation adjustments)
    for condition in classifications:
        if condition == 'healthy':
            ph_values.append(np.random.uniform(7.0, 7.8))
            temperature_values.append(np.random.uniform(31, 33))
            humidity_values.append(np.random.uniform(40, 60))
            exudate_values.append(np.random.uniform(1, 3))
            oxygen_values.append(np.random.uniform(85, 95))
            
        elif condition == 'infected':
            ph_values.append(np.random.uniform(7.8, 9.0))
            temperature_values.append(np.random.uniform(33, 37))
            humidity_values.append(np.random.uniform(60, 90))
            exudate_values.append(np.random.uniform(6, 10))
            oxygen_values.append(np.random.uniform(60, 80))
            
        else:  # wound_healing
            ph_values.append(np.random.uniform(6.5, 8.0))
            temperature_values.append(np.random.uniform(32, 34))
            humidity_values.append(np.random.uniform(50, 70))
            exudate_values.append(np.random.uniform(3, 6))
            oxygen_values.append(np.random.uniform(75, 90))
    
    # Convert to numpy arrays for easier manipulation
    ph_values = np.array(ph_values)
    temperature_values = np.array(temperature_values)
    humidity_values = np.array(humidity_values)
    exudate_values = np.array(exudate_values)
    oxygen_values = np.array(oxygen_values)
    
    # Add initial noise (5%)
    ph_values *= np.random.normal(1, 0.03, n_samples)
    temperature_values *= np.random.normal(1, 0.02, n_samples)
    humidity_values *= np.random.normal(1, 0.04, n_samples)
    exudate_values *= np.random.normal(1, 0.06, n_samples)
    oxygen_values *= np.random.normal(1, 0.03, n_samples)
    
    # Apply correlations based on research literature
    # 1. pH and Temperature Correlation (r=0.73)
    # For each 0.1 increase in pH above 7.6, temperature increases by ~0.2°C
    for i in range(n_samples):
        if ph_values[i] > 7.6:
            # Add temperature adjustment based on pH
            ph_diff = ph_values[i] - 7.6
            temp_adjustment = 0.2 * (ph_diff / 0.1)
            temperature_values[i] += temp_adjustment * np.random.uniform(0.8, 1.2)  # Add some variation
    
    # 2. Exudate and Humidity Correlation (r=0.81)
    # High exudate (7-10) correlates with humidity (78-92%)
    for i in range(n_samples):
        if exudate_values[i] > 5:
            # Scale humidity based on exudate level
            humidity_adjustment = (exudate_values[i] - 5) * 3  # Approximately 3% per exudate point
            humidity_values[i] = max(humidity_values[i], 60 + humidity_adjustment * np.random.uniform(0.9, 1.1))
    
    # 3. Oxygen Saturation and Temperature Inverse Correlation (r=-0.68)
    # For each 1°C increase, oxygen decreases by ~3-5%
    for i in range(n_samples):
        baseline_temp = 32  # Reference temperature
        if temperature_values[i] > baseline_temp:
            # Reduce oxygen based on temperature elevation
            temp_diff = temperature_values[i] - baseline_temp
            oxygen_reduction = temp_diff * np.random.uniform(3, 5)
            oxygen_values[i] = max(oxygen_values[i] - oxygen_reduction, 55)  # Ensure it doesn't go too low
    
    # 4. pH and Oxygen Saturation Inverse Correlation (r=-0.58)
    # Higher pH correlates with lower oxygen due to increased bacterial activity
    for i in range(n_samples):
        if ph_values[i] > 7.6:
            # Reduce oxygen based on pH level
            ph_diff = ph_values[i] - 7.6
            oxygen_reduction = ph_diff * 5 * np.random.uniform(0.9, 1.1)  # ~5% per 0.1 pH unit
            oxygen_values[i] = max(oxygen_values[i] - oxygen_reduction, 55)  # Ensure it doesn't go too low
    
    # Add final noise layer (smaller, 2-3%)
    ph_values *= np.random.normal(1, 0.02, n_samples)
    temperature_values *= np.random.normal(1, 0.01, n_samples)
    humidity_values *= np.random.normal(1, 0.03, n_samples)
    exudate_values *= np.random.normal(1, 0.04, n_samples)
    oxygen_values *= np.random.normal(1, 0.02, n_samples)
    
    # Ensure values stay within realistic bounds
    ph_values = np.clip(ph_values, 5.0, 9.5)
    temperature_values = np.clip(temperature_values, 30, 38)
    humidity_values = np.clip(humidity_values, 30, 100)
    exudate_values = np.clip(exudate_values, 0, 10)
    oxygen_values = np.clip(oxygen_values, 50, 100)
    
    # Handle class boundary cases - ensure parameters maintain coherent relationships
    for i in range(n_samples):
        # Example: If at class boundary for pH (around 7.8) ensure other parameters align
        if 7.7 <= ph_values[i] <= 7.9:
            # If classified as healthy but pH suggests infection
            if classifications[i] == 'healthy' and ph_values[i] > 7.8:
                # Adjust other parameters to be more consistent
                temperature_values[i] = min(temperature_values[i], 33)
                humidity_values[i] = min(humidity_values[i], 60)
                exudate_values[i] = min(exudate_values[i], 3.5)
                oxygen_values[i] = max(oxygen_values[i], 82)
            # If classified as infected but pH is borderline
            elif classifications[i] == 'infected' and ph_values[i] < 7.8:
                # Adjust other parameters to be more consistent with infection
                temperature_values[i] = max(temperature_values[i], 33.5)
                humidity_values[i] = max(humidity_values[i], 65)
                exudate_values[i] = max(exudate_values[i], 5.5)
                oxygen_values[i] = min(oxygen_values[i], 78)
    
    # Create DataFrame
    df = pd.DataFrame({
        'pH': ph_values,
        'temperature': temperature_values,
        'humidity': humidity_values,
        'exudate_level': exudate_values,
        'oxygen_saturation': oxygen_values,
        'condition': classifications
    })
    
    return df

def save_dataset(save_path='data', train_samples=1000, test_samples=250):
    """Generate and save dataset to CSV file"""
    # Create train and test datasets
    train_data = generate_dataset(n_samples=train_samples, random_state=42)
    test_data = generate_dataset(n_samples=test_samples, random_state=43)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save datasets
    timestamp = datetime.now().strftime("%Y%m%d")
    train_path = f"{save_path}/wound_data_train_{timestamp}.csv"
    test_path = f"{save_path}/wound_data_test_{timestamp}.csv"
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Datasets saved to {save_path}/")
    print(f"Training dataset: {train_path} ({train_samples} samples)")
    print(f"Test dataset: {test_path} ({test_samples} samples)")
    return train_data, test_data

if __name__ == "__main__":
    # Increased dataset size by 20x for better generalization
    train_data, test_data = save_dataset(train_samples=20000, test_samples=5000)
    
    # Print dataset statistics
    print("\nTraining Dataset Statistics:")
    print(f"Shape: {train_data.shape}")
    print(f"Class distribution: {train_data['condition'].value_counts().to_dict()}")
    
    print("\nTest Dataset Statistics:")
    print(f"Shape: {test_data.shape}")
    print(f"Class distribution: {test_data['condition'].value_counts().to_dict()}") 