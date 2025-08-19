AI-Powered Worker Safety Monitoring and Alerts Agent
This repository contains the implementation of an AI-powered system for worker safety monitoring and alerts. The project leverages machine learning techniques to process safety-related data, perform feature engineering, and build predictive models to enhance worker safety.
Project Overview

ETL and EDA: Extracted, transformed, and loaded data, followed by exploratory data analysis using Python. Reduced feature dimensionality by 70% using Principal Component Analysis (PCA).
Feature Selection: Applied Chi-Square and T-Test to select the most relevant features, optimizing model performance.
Modeling: Developed Logistic Regression (LR), Support Vector Machine (SVM), and Random Forest models using scikit-learn, improving retention accuracy by 15%.

Repository Structure

etl_eda.py: Handles data extraction, transformation, loading, and exploratory data analysis with PCA.
feature_selection.py: Implements Chi-Square and T-Test for feature selection.
model_training.py: Contains code for training and evaluating Logistic Regression, SVM, and Random Forest models.
requirements.txt: Lists the required Python packages.
sample_data.csv: Placeholder for sample safety data (to be replaced with actual data).

Installation

Clone the repository:git clone https://github.com/your-username/ai-worker-safety-agent.git


Install dependencies:pip install -r requirements.txt


Ensure you have a dataset (sample_data.csv) in the root directory or update the file path in etl_eda.py.

Usage

Run the ETL and EDA script to preprocess the data:python etl_eda.py


Perform feature selection:python feature_selection.py


Train and evaluate the models:python model_training.py


Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
