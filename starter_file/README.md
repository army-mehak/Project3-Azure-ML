# Capstone Project - Azure Machine Learning Engineer
The aim of the project is to compare model performances and choose the best for performance. In this project, heart disease dataset is adapted from Kaggle to perform classification using AutoML and customized model through hypertuning. In the end, the best model will get deployed as a webservice.


## Dataset
The dataset consists of data record of patient from their age, gender, blood pressure level to their heart rate. It is adapted from Kaggle Open Dataset. It is a classification problem that determines whether a person has a heart disease or not. The classification is a binary classification with 0 representing an absence of heart disease and 1 representing presence of heart disease.
The dataset consist information of patient that might be having heart disease. The target column has values 0 and 1 which determines if the patients has heart disease. There are 14 attributes in the dataset as follows:
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

### Overview
In this project, the heart dataset was used to create an AutoML model and a customized model by tuning hyperparameters. The data was taken from Kaggle and preprocessed. Some large values on some attributes were standardized and binning was performed on the age column. After preprocessing the data, the data was fed into AutoML model and the best model was Voting Ensemble with 92.X% accuracy. The preprocessed data was also fed into a custom model of Logistic Regression was created where parameter hyperparameter tuning and the best model had an accuracy of 88.X%. The Voting Ensemble model from AutoML was deployed as a webservice and the rest endpoint HTTP was available.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
