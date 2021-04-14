import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.core import Workspace
from sklearn.preprocessing import StandardScaler



def clean_data(dataset):
    ### standardization
    x_df = dataset.dropna() #drop any rows with null values
    y_df = x_df.pop("target")
    x_df[['trestbps', 'chol', 'thalach']] = StandardScaler().fit_transform(x_df[['trestbps', 'chol', 'thalach']])

    ### binning
    min_value = x_df['age'].min()
    max_value = x_df['age'].max()
    x_df['age_bins'] = pd.cut(x_df['age'], bins=7, labels = False) #bins=bins, labels=labels, include_lowest=True)
    
    ### drop age column
    drop_age = x_df.pop("age") 

    ### rearrange columns
    x_df = x_df[['age_bins', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
  
    return x_df, y_df
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser(description= 'This is Training Script of Tabular Dataset')

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(max_iter=args.max_iter, C=args.C).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.pkl')

    
# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at: 
ds = pd.read_csv('heart.csv')
#ds = TabularDatasetFactory.from_delimited_files("heart.csv")
x, y = clean_data(ds)

# TODO: Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30) #Train data 70% Test data 30%
    
run = Run.get_context()



if __name__ == '__main__':
    main()

