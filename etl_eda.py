import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ETL: Extract, Transform, Load
def load_data(file_path='sample_data.csv'):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: File not found. Please ensure sample_data.csv exists.")
        return None

def clean_data(df):
    # Handle missing values
    df = df.dropna()
    # Convert categorical variables to numeric (example)
    if 'category_column' in df.columns:
        df = pd.get_dummies(df, columns=['category_column'], drop_first=True)
    return df

def perform_eda(df):
    # Basic EDA
    print("\nDataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

def apply_pca(df, n_components=0.3):
    # Assuming last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Variance explained
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2f}")
    
    # Create new dataframe with PCA components
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['target'] = y.reset_index(drop=True)
    
    return df_pca

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Perform EDA
    perform_eda(df_cleaned)
    
    # Apply PCA
    df_pca = apply_pca(df_cleaned)
    
    # Save processed data
    df_pca.to_csv('processed_data.csv', index=False)
    print("Processed data saved as 'processed_data.csv'.")

if __name__ == "__main__":
    main()
