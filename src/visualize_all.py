import os
import subprocess
import sys

def main():
    """Run all visualization scripts to generate comprehensive visualizations"""
    print("===== GENERATING COMPREHENSIVE VISUALIZATIONS =====")
    
    # Check if required directories exist
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
        print("Created 'data' directory")
    
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
        print("Created 'models' directory")
    
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
        print("Created 'results' directory")
    
    # Run data visualization script
    print("\n===== GENERATING DATA VISUALIZATIONS =====")
    try:
        data_viz_result = subprocess.run([sys.executable, 'src/visualize_data.py'], 
                                         capture_output=True, text=True, check=True)
        print(data_viz_result.stdout)
        if data_viz_result.stderr:
            print("Errors/Warnings:", data_viz_result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running data visualization script:")
        print(e.stderr)
        print("Exit code:", e.returncode)
    
    # Run model visualization script
    print("\n===== GENERATING MODEL VISUALIZATIONS =====")
    try:
        model_viz_result = subprocess.run([sys.executable, 'src/visualize_model.py'], 
                                          capture_output=True, text=True, check=True)
        print(model_viz_result.stdout)
        if model_viz_result.stderr:
            print("Errors/Warnings:", model_viz_result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running model visualization script:")
        print(e.stderr)
        print("Exit code:", e.returncode)
    
    print("\n===== VISUALIZATION COMPLETE =====")
    print("All visualizations have been generated and saved to the 'results' directory.")

if __name__ == "__main__":
    main() 