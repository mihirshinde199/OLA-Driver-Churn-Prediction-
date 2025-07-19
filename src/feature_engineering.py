def add_target_column(df):
    df['target'] = df['LastWorkingDate'].notnull().astype(int)
    return df

def add_income_rating_trends(df):
    df = df.sort_values(['Driver_ID', 'MMMM-YY'])
    df['Income_Increase'] = df.groupby('Driver_ID')['Income'].diff().gt(0).astype(int)
    df['Rating_Increase'] = df.groupby('Driver_ID')['Quarterly Rating'].diff().gt(0).astype(int)
    return df
