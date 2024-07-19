import pandas as pd
from gemini import *

def read_file(file):
    df = pd.read_csv(file)
    df.replace(r'^\s*\?\s*$', pd.NA, regex=True, inplace=True)
    df1 = df.copy()
    return df,df1

def columns_name(data):
    column_names = data.columns.tolist()
    return column_names

def drop_unwanted_columns(data,column_name):
    if column_name is not None:
        data = data.drop(column_name,axis =1)
    else:
        pass
        

def fill_null_with_mean(column):
    mean_value = column.mean()
    return column.fillna(mean_value)

def fill_null_with_median(column):
    median_value = column.median()
    return column.fillna(median_value)

def fill_null_with_mode(column):
    mode_value = column.mode()[0] 
    return column.fillna(mode_value)

def fill_null_with_midpoint(column):
    min_value = column.min()
    max_value = column.max()
    midpoint = (min_value + max_value) / 2
    return column.fillna(midpoint)

def remove_null_values(df):
    df = df.dropna()
    return df

def handle_null_values(df,df1):
    null_values = df.isnull().sum()
    if null_values.sum() == 0:
        print('no null')
        pass
    else:
        missing_percentage = df.isnull().sum() / len(df) * 100
        total_missing_percentage = missing_percentage.sum()
        if total_missing_percentage<5:
            df = df.dropna()
        else:        
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = fill_null_with_mode(df[column])
                elif pd.api.types.is_numeric_dtype(df[column].dtype):
                    plot = create_plot(df1[column])
                    image = open(plot)
                    response = return_llm_response(image)
                    if response == 'normal distribution':
                        df[column] = fill_null_with_mean(df[column])
                    elif response == 'Uniform Distribution':
                        df[column] = fill_null_with_midpoint(df[column])
                    elif response == 'Right-skewed' or 'Left-skewed':
                        df[column] = fill_null_with_median(df[column])
                    elif response == 'Bimodal Distribution' or 'Multimodal Distribution':
                        df[column] = fill_null_with_mode(df[column])
    return df
