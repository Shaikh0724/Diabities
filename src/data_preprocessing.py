import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    print("[-] Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 1. Missing values handling ('?' ko remove/replace karna)
    df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True, errors='ignore')
    df.replace('?', 'Unknown', inplace=True)
    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True, errors='ignore')
    
    # 2. 'age' interval ko numeric midpoint me convert karna
    def age_to_numeric(age_str):
        try:
            age_str = age_str.replace('[', '').replace(')', '')
            start, end = map(int, age_str.split('-'))
            return (start + end) / 2
        except:
            return 50
            
    df['age'] = df['age'].apply(age_to_numeric)
    
    # 3. Binary classification target variable (<30 readmitted = 1, baki sab = 0)
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # 4. Categorical variables encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # 80% train aur 20% test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[+] Preprocessing complete. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test