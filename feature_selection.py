import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy.stats import ttest_ind

def load_processed_data(file_path='processed_data.csv'):
    try:
        data = pd.read_csv(file_path)
        print("Processed data loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: File not found. Please run etl_eda.py first.")
        return None

def chi_square_selection(X, y, k=5):
    # Apply Chi-Square test
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    print("\nChi-Square Scores:")
    print(scores.sort_values(by='Score', ascending=False))
    return X.columns[selector.get_support()].tolist()

def t_test_selection(X, y, threshold=0.05):
    # Perform T-Test for each feature
    selected_features = []
    for column in X.columns:
        group1 = X[y == 0][column]
        group2 = X[y == 1][column]
        t_stat, p_value = ttest_ind(group1, group2)
        if p_value < threshold:
            selected_features.append(column)
    print("\nT-Test Selected Features (p < 0.05):")
    print(selected_features)
    return selected_features

def main():
    # Load processed data
    df = load_processed_data()
    if df is None:
        return
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Chi-Square feature selection
    chi2_features = chi_square_selection(X, y)
    
    # T-Test feature selection
    ttest_features = t_test_selection(X, y)
    
    # Combine selected features
    selected_features = list(set(chi2_features + ttest_features))
    print("\nFinal Selected Features:")
    print(selected_features)
    
    # Save selected features dataset
    df_selected = df[selected_features + ['target']]
    df_selected.to_csv('selected_features_data.csv', index=False)
    print("Selected features data saved as 'selected_features_data.csv'.")

if __name__ == "__main__":
    main()
