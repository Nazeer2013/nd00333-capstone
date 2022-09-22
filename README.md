# nd00333-capstone
Udacity AzureML Final Project

# SMS Spam Detection using Azure Machine learning

Problem of Information overload need to be recognized and addressed.

        A wealth of information creates a poverty of attention. 
                                          - Herbert Simon

One of the answers to the problem is to filter out the noise(spam) in this case from SMS.


### Context

Spam is unsolicited and unwanted messages sent electronically  and whose content may be malicious. SMS spam is typically transmitted over a mobile network. SMS mode is more dangerous in many ways one is by the fact that SMS is usually regarded by the user as a safer, more trustworthy form of communication than other sources, e. g., emails.



# Auto ML using Python SDK


[Spam Detection Auto ML Python SDK Project Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/notebook-smsspam-automl-0915-v1.ipynb)

AutoML SDK Architecture Flow

![AutoML SDK Architecture Flow](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/AutoMLArchitectuerFlow.png)

### Step 1: Initialize Workspace

Workspace is initialized form config. Make sure to have config.json in your notebook folder.


### Step 2: Create Experiment attach Compute

**Amlcompute** an Azure Machine Learning Compute is a managed-compute infrastructure that allows you to easily create a single or multi-node compute.
Virtual machine size of Standard_DS12_v2 (4 cores, 28 GB RAM, 56 GB disk) with max 4 Nodes.

Create an environment from a Conda specification. 

### Step 3: Load Dataset

For already uploaded dataset bring it to your notebook using key from workspace. 

### Kaggle SMS Spam Data View

Kaggle SMS Spam data:


![Kaggle SMS Spam data](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/images/DatasetView.png)



### Step 4: Create AutoML Config and Submit AutoML Experiment

Represents configuration for submitting an automated ML experiment in Azure Machine Learning.

This configuration object contains and persists the parameters for configuring the experiment run, as well as the training data to be used at run time.

SMS Spam detection is a classification type problem where y (v1 in this dataset) the class to be predicted has a cardinality of 2(binary classification ham or spam) and x (v2 message). Featurization is set to 'auto' for AutoMl to automatically pick and chose features and run with it. For text columns auto ml engages bag of words, pre-trained word embedding and text target encoding techniques. 

Other settings are enable early stopping. Early stopping is triggered if the absolute value of best score calculated is the same of the past n_iters iterations. i.e., if there is no improvement in score for early_stopping_n_iters iterations.

Ready to submit AutoMl experiment.

***Auto ML Experiment Running view.*** 

Once the execution is finished you can request detailed information about the results.

![Auto ML Experiment Running view](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_3.png)

![Auto ML Step 4](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_4.png)


***AutoML Experiment Completed view***

![AutoML Experiment Completed view](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1.png)

***Retrieve Best Model***

Now that you have several trained models ranked based on the metric you specified when configuring the AutoML experiment, you can retrieve the best one and use it to score and deploy as a service.

![Auto ML Best Model](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_2.png)


### Step 5: Deploy Best Model

a. Use AutoML run get output to retrive best run. Using best run get model name. 

b. Use AutoML run to register best model.

c. Download score and environment files from best run.

d. Use Model deploy to deploy model by providing InferenceConfig and AciWebservice configuration. 


### Step 6: Test API Endpoint

Test model service using scoring uri.


# Azure HyperDrive excution using Logistic Regression

** SMS Spam using HyperDrive Logistic Regression 

[Logistic Regression Hyperdrive run Notebook](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/nb-lr-0919-v7.ipynb)

![](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/LogReg_Hperdrive_Hyperparam_Results.png)

![](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/LogReg_Hperdrive_Hyperparam_BestRun.png)

![](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/LogReg_Hyperparam_BestrunPerf.png)

![](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/RegisteredBestModel.png)


## HyperParameter tuning Keras tuner

[Keras Deep Learning Model and HyperParameter tuning](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/TFKerasHyperParameterV2.ipynb)



### References

[Living in the age of information overload](https://tom-stevenson.medium.com/we-are-living-in-the-age-of-information-overload-720ea5d31afb)

[Kaggle Spam-Ham SMS classification Data](https://www.kaggle.com/code/rumbleftw/beginner-friendly-spam-ham-sms-classification/data?select=spam.csv)
