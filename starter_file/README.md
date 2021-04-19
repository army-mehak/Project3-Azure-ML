# Capstone Project - Azure Machine Learning Engineer
The aim of the project is to classify heart disease dataset by comparing model performances of 2 models. In this project, heart disease dataset was adapted from Kaggle to perform classification using AutoML and customized model through hypertuning. The task was to use the best of the 2 model was deployed and then used as a webservice to predict data. The best of the two models was AutoML with accuracy of 95% whereas Hyperdrive best model ran with accuracy of 88%. The AutoML model was registered, deployed and then used as a webservice for prediction.


## Dataset

### Overview
The dataset consists of data record of patient from their age, gender, blood pressure level to their heart rate. It is adapted from Kaggle Open Dataset. It is a classification problem that determines whether a person has a heart disease or not. There are 14 attributes in the dataset as follows:
1. age
2. sex
3. chest pain type
4. resting blood pressure
5. serum cholestoral in mg/dl
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved
9. exercise induced angina
10. old peak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment
12. number of major vessels (0-3) colored by fluoroscopy
13. thal: 3 = normal; 6 = fixed defect; 7 = reversible defect
14. target column, 0 = heart disease not present, 1 = heart disease present

In this project, the heart dataset was used to create an AutoML model and a customized model by tuning hyperparameters. The data was taken from Kaggle and preprocessed. Some large values on some attributes were standardized and binning was performed on the age column. After preprocessing the data, the data was fed into AutoML model and the best model was Voting Ensemble with 92.X% accuracy. The preprocessed data was also fed into a custom model of Logistic Regression was created where parameter hyperparameter tuning and the best model had an accuracy of 88.X%. The Voting Ensemble model from AutoML was deployed as a webservice and the rest endpoint HTTP was available.

### Task
The classification is a binary classification with 0 representing an absence of heart disease and 1 representing presence of heart disease.
The dataset consist information of patient that might be having heart disease. The target column has values 0 and 1 which determines if the patients has heart disease. 

The following steps were adapted for processing the data:
1. On the age column, binning was performed into 7 groups from 0-7 to allow less computational cost.
2. Standardization was performed on attributes trestbps, chol and thalach to reduce the value.
3. Null value rows were dropped as part of data cleaning.

### Access
The dataset can be downloaded from Kaggle into your project workspace as blob data URI. The data can then further be broken down into train and test data for training and predicting.
For the AutoML, the data was uploaded on Azure platform and called through the notebook using available datasets under current workspace. The data was then send for preprocessing as pandas dataframe and then available for training.
For the Hyperdrive, the data was used as a csv file which was preprocessed and then used directly from the current directory.

## Automated ML
*TODO*: Give an overview of the `utoml` settings and configuration you used for this experiment
AutoML settings is defined and passed to AutoML configuration as a parameters. 
AutoML settings: 
{
    "experiment_timeout_minutes": 20", 
    max_concurrent_iterations": 5,
    "primary_metric": 'AUC_weighted'
}
In AutoML configuration, the train data was passed as parameter along with the type of problem this is i.e. classification.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best AutoML model was Voting Ensemble with an accuracy of 95%. The Voting Ensemble used multiple algorithms such as:
- 
- 
- 
- 
The AutoML settings could have been changed by decreasing the experiment timeout minutes as the simple classification was quick to run.


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
(add screenshot of run widget)
(add screenshot of best model)
(add screenshot through azure)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
Logistic Regression was chosen as a model as its best suited for binary classification problem. The two types of hyperparameters selected were the maximum number of iterations and C i.e. inverse of regularatization strength. The value for number of iteration were choice for range of value from 10 to 31 and the value for C was uniform from 0 to 10.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The best Hyperparameter model was XXXXX with an accuracy of 88%. It used multiple algorithms such as:
- 
- 
- 
- 
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The best AutoML model 'Voting Ensemble' of accuracy 95% was deployed 

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
