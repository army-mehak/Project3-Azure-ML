import pandas as pd 
import numpy as np 

def clean_data(dataset):
    ### standardization
    x_df = dataset.to_pandas_dataframe().dropna() #drop any rows with null values
    y_df = x_df.pop("target")
    x_df[['trestbps', 'chol', 'thalach']] = StandardScaler().fit_transform(x_df[['trestbps', 'chol', 'thalach']])
    #print(x_df)
    #print(y_df)

    ### binning
    min_value = df['age'].min()
    max_value = df['age'].max()
    bins = np.linspace(min_value,max_value,7)
    #print(bins)
    labels = ['29-37', '38-45', '46-53', '54-61', '62-69', '70-77']
    x_df['age_bins'] = pd.cut(x_df['age'], bins=bins, labels=labels, include_lowest=True)
    drop_age = x_df.pop("age") #drop age column

    ### rearrange columns
    x_df = x_df[['age_bins', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    #pd.set_option('display.max_rows', x_df.shape[0]+1)
    #print(x_df)
    return x_df, y_df

# def main():
    


# if __name__ == '__main__':
#     main()

