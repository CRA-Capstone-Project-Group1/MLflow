# function to load and process the data

import pandas as pd
import logging
data_path = "/Users/swapnilklkar/Documents/Experiment_tracking_with_MLflow/data/raw/credit.csv"
def load_and_preprocess_data(data_path):
    
    try:
        
        # Import the data from 'credit.csv'
        df = pd.read_csv(data_path)

        # Impute all missing values in all the features
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})

        # Drop 'Loan_ID' variable from the data
        df = df.drop('Loan_ID', axis=1)

        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

if __name__=="__main__":
    df = load_and_preprocess_data(data_path)   
    print(df.head())     