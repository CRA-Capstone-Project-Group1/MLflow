import pandas as pd
import logging

# create dummy features
def create_dummy_vars(df):
    try:
        
        # Create dummy variables for all 'object' type variables except 'Loan_Status'
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
        
        # store the processed dataset in data/processed
        df.to_csv('data/processed/Processed_Credit_Dataset.csv', index=None)

        # Separate the input features and target variable
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        return X, y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))