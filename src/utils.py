def check_nan(df):
    print(f'Percentage of missing values in each column:\n\n{round(df.isnull().sum().sort_values(ascending=False) / len(df.index) * 100, 2)}\n')