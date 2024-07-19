import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import pandas as pd
from gemini import *

def replace_outliers_with_mean(df,column, outliers):
    mean_value = column.mean()
    df.loc[outliers,column.name] = mean_value

def replace_outliers_with_median(df,column, outliers):
    median_value = column.median()
    df.loc[outliers,column.name] = median_value
    
def replace_outliers_with_mode(df,column, outliers):
    mode_value = column.mode()[0] 
    df.loc[outliers,column.name] = mode_value

def replace_outliers_with_midpoint(df, column, outliers):
    min_value = column.min()
    max_value = column.max()
    midpoint = (min_value + max_value) / 2
    df.loc[outliers,column.name] = midpoint

def detect_outliers_z_score(column, threshold=2.5):
    mean = np.mean(column)
    std = np.std(column)
    z_scores = (column - mean) / std
    return np.abs(z_scores) > threshold

def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

def winsorizing(df, name, upper_limit=0.02, lower_limit=0.02):
    limits = [lower_limit, upper_limit]
    df[name] = stats.mstats.winsorize(df[name], limits=limits)

def handle_outliers(df,target_column):
    for column in df.columns:
        if column == target_column:
            pass
        else:
            if pd.api.types.is_numeric_dtype(df[column].dtype):
                print(column)
                skewness = skew(df[column])
                kurtosiss = kurtosis(df[column])
                print(column, "done")
                if skewness<-1 or skewness>1 or kurtosiss>0.9:
                    winsorizing(df,column)
                else:
                    plot = create_plot(df[column])
                    image = open(plot)
                    response = return_llm_response(image)
                    if response == 'normal distribution':
                        outliers = detect_outliers_z_score(df[column])
                        replace_outliers_with_mean(df,df[column],outliers)
                    elif response == 'Uniform Distribution':
                        outliers = detect_outliers_iqr(df[column])
                        replace_outliers_with_midpoint(df,df[column],outliers)
                    elif response == 'Right-skewed' or 'Left-skewed':
                        outliers = detect_outliers_iqr(df[column])
                        replace_outliers_with_median(df,df[column],outliers)
                    elif response == 'Bimodal Distribution' or 'Multimodal Distribution':
                        outliers = detect_outliers_iqr(df[column])
                        replace_outliers_with_mode(df,df[column],outliers)
            else:
                pass
    return df


