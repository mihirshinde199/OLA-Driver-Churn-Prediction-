import pandas as pd
from sklearn.impute import KNNImputer

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Convert date columns
    df['Date Of Joining'] = pd.to_datetime(df['Date Of Joining'])
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')
    df['MMMM-YY'] = pd.to_datetime(df['MMMM-YY'], format='%b-%y')

    return df

def knn_impute(df):
    num_df = df.select_dtypes(include='number')
    imputer = KNNImputer()
    df[num_df.columns] = imputer.fit_transform(num_df)
    return df
