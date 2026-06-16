from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(models, X_test, y_test, output_dir="output"):
    # Output directory banayein agar exist nahi karti
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    
    print("\n================ Model Evaluation Results ================")
    for name, model in models.items():
        predictions = model.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4)
        })
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        # Confusion Matrix nikalna aur print karna
        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        print(cm)
        print("-" * 50)
        
        # Confusion Matrix ka Graph banana aur save karna
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Graph save karna
        filename = f'confusion_matrix_{name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close() # Memory free karne ke liye close karna zaroori hai
        
    comparison_df = pd.DataFrame(results)
    
    # Model Comparison ka Bar Chart banana
    if not comparison_df.empty:
        # Data ko plot ke hisaab se reshape karna
        df_melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
        plt.title('Models Performance Comparison')
        plt.ylim(0, 1.1) # Scores 0 se 1 ke darmiyan hote hain
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_model_comparison.png'))
        plt.close()
        
    return comparison_df