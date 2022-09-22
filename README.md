# nd00333-capstone
Udacity AzureML Final Project

# SMS Spam Detection using Azure Machine learning

Problem of Information overload need to be recognized and addressed.

        A wealth of information creates a poverty of attention. 
                                          - Herbert Simon

One of the answers to the problem is to filter out the noise(spam) in this case from SMS.


## Context and Data Selection

Spam is unsolicited and unwanted messages sent electronically  and whose content may be malicious. SMS spam is typically transmitted over a mobile network. SMS mode is more dangerous in manyways one is by the fact that SMS is usually regarded by the user as a safer, more trustworthy form of communication than other sources, e. g., emails.





## Auto ML using Python SDK


[Spam Detection Auto ML Python SDK Project Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/notebook-smsspam-automl-0915-v1.ipynb)

![AutoML SDK Architecture Flow](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/AutoMLArchitectuerFlow.png)

***Step 1: Initialize Workspace***

***Step 2: Create Experiment attach Compute***



***Step 3: Upload Dataset***

***Step 4: Create AutoML Config and Submit AutoML Experiment***

![Auto ML Experiment Running view](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_3.png)

![AutoML Experiment Completed view](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1.png)

![Auto ML Best Model](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_2.png)


***Step 5: Deploy Best Model***




***Step 6: Test API Endpoint***

![Auto ML Step 4](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_4.png)





## Azure HyperDrive excution using Logistic Regression

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
