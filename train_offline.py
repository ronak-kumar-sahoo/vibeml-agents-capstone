import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_save_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "uploads", "churn_customer_data.csv")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    # 1. Load Data
    df = pd.read_csv(dataset_path)
    
    # Drop customer ID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    X = df.drop(columns=['churn'])
    y = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 2. Preprocessing Pipeline
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # 3. Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # 4. Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Customer Churn')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # 5. Generate Feature Importance Plot
    # Get feature names from encoder
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + ohe_features
    
    importances = model.named_steps['classifier'].feature_importances_
    
    # Sort
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(all_features))
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances[indices[:top_n]], 
        y=[all_features[i] for i in indices[:top_n]], 
        palette='viridis'
    )
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance saved to {fi_path}")

if __name__ == "__main__":
    train_and_save_assets()
