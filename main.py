import os
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_models

def main():
    dataset_path = os.path.join('data', 'diabetic_data.csv')
    output_folder = 'output'
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}. Please check your 'data' folder.")
        return

    # Output folder check/create karna
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[-] Created '{output_folder}' directory for graphs.")

    # Step 1: Preprocess Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    # Step 2: Train Models
    trained_models = train_models(X_train, y_train)
    
    # Step 3: Evaluate Models and Generate Graphs
    print("[-] Evaluating models and generating graphs...")
    comparison_matrix = evaluate_models(trained_models, X_test, y_test, output_dir=output_folder)
    
    print("\n### Final Model Comparison Summary ###")
    print(comparison_matrix.to_string(index=False))
    print(f"\n[+] Success! Sab graphs '{output_folder}' folder me save ho gaye hain.")

if __name__ == "__main__":
    main()