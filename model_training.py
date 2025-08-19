import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_selected_data(file_path='selected_features_data.csv'):
    try:
        data = pd.read_csv(file_path)
        print("Selected features data loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: File not found. Please run feature_selection.py first.")
        return None

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
        print(f"{name} model saved as '{name.lower().replace(' ', '_')}_model.pkl'.")

def main():
    # Load selected features data
    df = load_selected_data()
    if df is None:
        return
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train and evaluate models
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
