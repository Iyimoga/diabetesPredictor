"""
Train the diabetes prediction model and save it as a pickle file.
Run this script before deploying the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

def train_and_save_model(csv_path):
    """
    Train the diabetes prediction model and save it.
    
    Parameters:
    -----------
    csv_path : str
        Path to the diabetes dataset CSV file
    """
    
    print("Loading dataset...")
    dataset = pd.read_csv(csv_path)
    
    print("Preprocessing data...")
    # Convert CLASS column to string and remove spaces
    dataset['CLASS'] = dataset['CLASS'].astype(str).str.strip()
    
    # Map class values
    class_mapping = {
        'N': 'Non-diabetic',
        'P': 'Prediabetic',
        'Y': 'Diabetic'
    }
    dataset['CLASS'] = dataset['CLASS'].map(class_mapping)
    
    # Convert numeric columns
    numeric_cols = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
    for col in numeric_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    # Encode categorical columns
    label_encoder = LabelEncoder()
    dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
    dataset['CLASS'] = label_encoder.fit_transform(dataset['CLASS'])
    
    # Drop unnecessary columns
    encoded_dataset = dataset.drop(["ID", "No_Pation"], axis=1)
    
    # Remove any rows with missing values
    encoded_dataset = encoded_dataset.dropna()
    
    print(f"Dataset shape after preprocessing: {encoded_dataset.shape}")
    print(f"Class distribution:\n{encoded_dataset['CLASS'].value_counts()}")
    
    # Prepare data for modeling
    X = encoded_dataset.drop('CLASS', axis=1)
    y = encoded_dataset['CLASS']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\nTraining model...")
    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-diabetic', 'Prediabetic', 'Diabetic']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model
    print("\nSaving model...")
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✅ Model saved as 'diabetes_model.pkl'")
    print("\nYou can now run the Streamlit app!")
    
    return model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path - update this to your dataset location
        csv_path = "Dataset_of_Diabetes.csv"
    
    try:
        model = train_and_save_model(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{csv_path}'")
        print("Usage: python train_model.py <path_to_dataset.csv>")
    except Exception as e:
        print(f"Error occurred: {e}")