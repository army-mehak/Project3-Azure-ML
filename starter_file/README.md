# Capstone Project - Azure Machine Learning Engineer
The aim of the project is to classify heart disease dataset by comparing model performances of 2 models. In this project, heart disease dataset was adapted from Kaggle to perform classification using AutoML and customized model through hypertuning. The task was to use the best of the 2 model was deployed and then used as a webservice to predict data. The best of the two models was AutoML with accuracy of 93.8% whereas Hyperdrive best model ran with accuracy of 86.8%. The AutoML model was registered, deployed and then used as a webservice for prediction.


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

In this project, the heart dataset was used to create an AutoML model and a customized model by tuning hyperparameters. The data was taken from Kaggle and preprocessed. Some large values on some attributes were standardized and binning was performed on the age column. After preprocessing the data, the data was fed into AutoML model and the best model was Voting Ensemble with 93.8% accuracy. The preprocessed data was also fed into a custom model of Logistic Regression was created where parameter hyperparameter tuning and the best model had an accuracy of 88.X%. The Voting Ensemble model from AutoML was deployed as a webservice and the rest endpoint HTTP was available.

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
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
AutoML settings is defined and passed to AutoML configuration as a parameters.
AutoML settings:
{
    "experiment_timeout_minutes": 20",
    max_concurrent_iterations": 5,
    "primary_metric": 'AUC_weighted'
}
"experiment_timeout_minutes":20
As the dataset was just a csv file with upto 300+ rows of data- the experiment time was kept around 20 minutes to allow all iterations to finish before the experiment times out.

"primary_metric": 'AUC_weighted'
As this is aclassification problem, we chose the primary metric as 'AUC_weighted'. AUC weighted is the arithmetic mean of the score for each class, weighted by the number of true instances in each class.

AutoMLConfig:
(
    compute_target=cpu_cluster,
     task = 'classification',
     training_data = ds_train, #train data
     label_column_name = "target", #target column with 0 & 1
     path = './pipeline-project3',
     enable_early_stopping = True,
     featurization = 'auto',
     debug_log = 'automl_errors.log',
     **automl_settings
)

task: classification
As the target column has values 0 and 1 i.e binary classification so the task is assigned classification for this problem.

enable_early_stopping = True
This is enabled to initialise early stopping of the runs if the score of the different models are not improving over a period of time.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best AutoML model was Voting Ensemble with an accuracy of 93.8% while Stack Ensemble was second best with 93.6% and XGBoost Classifier being third best with 92.4% accuracy. The Voting Ensemble used SparseNormalizer and XGBoost Classifier with multiple parameters such as few mentioned below (Figure 1):
- base_Score = 0.5
- booster = 'gbtree'
- verbose = 10
- verbosity = 10

![1](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/automl-hyperparameters.png)
<p align="center">
 Figure 1: Parameters for Voting Ensemble (The Best Model)
</p>

As shown in Figure 2, the AutoML run ran with Voting Ensemble being the best model with an accuracy of 93.8%. In AutoML run, around 50 models were tested and the Voting Ensemble was top.
![2](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/a-1.PNG)

![3](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/a-2.PNG)
<p align="center">
 Figure 2: Run Widget showing the all model run

We can also see the best Model by going to Experiment -> Run -> Child Runs ad choosing the top model as its the best model for the run.
 ![alt text](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/a-3.PNG)
 <p align="center">
  Figure 3: Best Model through Azure ML UI
The AutoML could have been tried out with different settings such as enabling dnn and using a different primary metric. 

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Logistic Regression was chosen as a model as its best suited for binary classification problem. The two types of hyperparameters selected were the maximum number of iterations and C i.e. inverse of regularatization strength with Random Parameter Sampling. Random Parameter Sampling was chosen as it supports early termination of low-performance runs. 
The reason to choose C is that it allows increasing the magnitude of parameter values in order to reduce overfitting. It tries to minimize the error between predicted and actual value. The metric was assigned a float ranging from 0 to 10.  This is done to avoid too small or too large of C to not lead into overfitting or underfitting. So its a uniform value. Whereas the reason to choose max_iteration is that it allows to set a maximum number of iteration for a particular run that could avoid workspace timeout.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The best Hyperparameter model of Logistic Regression ran with an 86.8% accuracy which is 7% less than that of AutoML. The hyperpara
- 'Regularization Strength:': 0.41825683985249884,
-  'Max iterations:': 13
![1](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/h-3.PNG)
<p align="center">
 Figure 4:Parameters for Best Model
</p>

As shown in Figure 5, the Logistic Regression ran with highest accuracy around 86.6% with max total runs of 20.
![2](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/h-1.PNG)

![3](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/h-2.PNG)
<p align="center">
 Figure 6: Run Widget showing the all run

We can also see the best Model by going to Experiment -> Run -> Child Runs ad choosing the top model as its the best model for the run.
 ![alt text](https://github.com/army-mehak/Project3-Azure-ML/blob/master/starter_file/img/h-4.PNG)
 <p align="center">
  Figure 3: Best Model through Azure ML UI
The Hyperdrive model could have been tried with different types of Parameter Sampling such as Grid Sampling as it performs simple grid search over all possible value.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The best AutoML model 'Voting Ensemble' of accuracy 93.8% was deployed using ACI WebSeervice. At first the model was deployed using Local WebService to debug the score.py() code and understanding status of the service. To test the endpoint, test input data was taken through train_test_split() method where its split as 70-30% of data. This test data was converted into a dictionary of key value pair. This dictionary was sent to the run() method in score.py() to perform prediction. Moreover, HTTP Post Request was also sent to the RestEndpoint to perform prediction. (Also shown in the recording)

## Screen Recording
Screen recording can be found in this link: https://www.youtube.com/watch?v=5a1r1Z4gTi0&ab_channel=MehakShahid

## Future Works
This dataset could be further tested with more customised model and different hyperparameter settings to allow the best model. In Hyperdrive, the data can be tested with Grid Sampling and Bayesian Sampling. Other models such as KNN, SVM or Decision Trees can be tested on to provide the best model.
The best model can then be deployed as web service and can be used by external application through Authentication key.
