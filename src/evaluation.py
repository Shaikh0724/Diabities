from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def evaluate_models(models, X_test, y_test):
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
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("-" * 50)
        
    comparison_df = pd.DataFrame(results)
    return comparison_df