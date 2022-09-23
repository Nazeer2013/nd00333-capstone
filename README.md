# nd00333-capstone
Udacity AzureML Final Project

# SMS Spam Detection using Azure Machine learning

Problem of Information overload need to be recognized and addressed.

        A wealth of information creates a poverty of attention. 
                                          - Herbert Simon

One of the answers to the problem is to filter out the noise(spam) in this case from SMS.


### Overview

Spam is unsolicited and unwanted messages sent electronically  and whose content may be malicious. SMS spam is typically transmitted over a mobile network. SMS mode is more dangerous in many ways one is by the fact that SMS is usually regarded by the user as a safer, more trustworthy form of communication than other sources, e. g., emails.

This problem is solved using three different approaches:

        1. Using Azure AutoML Python SDK
        2. Using Azure HyperDrive Prameter tuning
        3. Using Keras Tensorflow and Keras tunner Neural Network 

# =====================================
# Auto ML using Python SDK
# =====================================

***[Spam Detection Auto ML Python SDK Project Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/notebook-smsspam-automl-0915-v1.ipynb)***


Azure AutoML helps find the best model that suits your data FAST! 

With Automated machine learning we can focus on the testing of most accurate models and avoid testing a large range of less valuable models, as it retains only the ones we want.
                                        -- Matthieu Boujonnier [Schneider Electric]

***AutoML SDK Implementation Flow***

![AutoML SDK Architecture Flow](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/AutoMLArchitectuerFlow.png)

### Step 1: Initialize Workspace

Workspace is initialized form config. Make sure to have config.json in your notebook folder.


### Step 2: Create Experiment attach Compute

**Amlcompute** an Azure Machine Learning Compute is a managed-compute infrastructure that allows you to easily create a single or multi-node compute.
Virtual machine size of Standard_DS12_v2 (4 cores, 28 GB RAM, 56 GB disk) with max 4 Nodes.

Create an environment from a Conda specification. 

### Step 3: Load Dataset

For already uploaded dataset bring it to your notebook using key from workspace. 

***Kaggle SMS Spam Data View***

![Kaggle SMS Spam data](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/images/DatasetView.png)



### Step 4: Create AutoML Config and Submit AutoML Experiment

Represents configuration for submitting an automated ML experiment in Azure Machine Learning.

This configuration object contains and persists the parameters for configuring the experiment run, as well as the training data to be used at run time.

SMS Spam detection is a classification type problem where y (v1 in this dataset) the class to be predicted has a cardinality of 2(binary classification ham or spam) and x (v2 message). Featurization is set to 'auto' for AutoMl to automatically pick and chose features and run with it. For text columns auto ml engages bag of words, pre-trained word embedding and text target encoding techniques. 

Other settings are enable early stopping. Early stopping is triggered if the absolute value of best score calculated is the same of the past n_iters iterations. i.e., if there is no improvement in score for early_stopping_n_iters iterations.

Ready to submit AutoMl experiment.

***Auto ML Experiment Running view.*** 

Once the execution is finished you can request detailed information about the results.

![AutoML Execution](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automatedexecutionofmodels.png)



![Auto ML Experiment Running view](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_3.png)



![Auto ML Step 4](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/smsspam_aml_exp_v1_4.png)

***AutoML Run Complete: Best Run***

![Auto ML Best Model](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_bestrun1.png)

***Retrieve Best Model***

Now that you have several trained models ranked based on the metric you specified when configuring the AutoML experiment, you can retrieve the best one and use it to score and deploy as a service.




### Step 5: Deploy Best Model

a. Use AutoML run get output to retrive best run. Using best run get model name. 

b. Use AutoML run to register best model.

***Deployed Model***

![Deployed Model](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/RegisteredBestModel.png)

c. Download score and environment files from best run.

d. Use Model deploy to deploy model by providing InferenceConfig and AciWebservice configuration. 


### Step 6: Test API Endpoint

Test model service using scoring uri.

***Test Model API***

![Test Endpoint](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_test1png.png)

Postman Test

![Test Endpoint](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_postman_test2.png)

# =====================================
# Azure HyperDrive execution using Logistic Regression
# =====================================

## SMS Spam Experiment using HyperDrive Logistic Regression Notebook

***[Azure ML Hyperdrive run Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/nb-lr-0921-v12.ipynb)***



**Azure AutoML Hyperdrive Overview**

![Azure AutoML Overview](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/MicrosoftAutoML.png)

After selection of the Dataset for your usecase, define configuration and constraints followed with Feature engineering and applicable and effective Algorithm selection. You would then continue with Hyperparameter tuning experimentation to search for the best model. Select model from the best run for deployment. 


****Featurization**** is the process of applying techniques of Feature Engineering to the dataset that will help machine learning algorithm learn better.

In this case below listed feature engineering techniques are engaged

        a. Elimination of stop words

        b. Replacement of target attribute with numerics 1 for spam and 0 for ham.

        c. Clean text of any punctuations, whitespaces and web addresses.

        d. Stemming using NLTK Snowball Stemmer. A process of reducing a word to its base word,such that words of similar kind lie under a common stem.

        e. TfidfVectorizer is an important NLP feature that converts text to a matrix of TF-IDF features.
           (TF-IDF is an abbreviation of Term Frequency - Inverse Document Frequency)  
            


****Model Selection**** Logistic Regression sounds to be the most natural fit to binary Spam classification problem (citing whitepaper from International Journal in below references). 
        
Logistic funtion is a S-shaped curve (also known as a sigmoid curve) that for a given set of input variables, produces an output between 0 and 1 which can be used to represent probability and with a given threshold would classify one way or the other. 

**HyperParameters**

Regularization techniques and streangth are to avoid Overfitting or UnderFitting model training. 

Regularization techniques play a vital role in the development of machine learning models. Especially complex models, like neural networks, prone to overfitting the training data. Broken down, the word “regularize” states that we’re making something regular. In a mathematical or ML context, we make something regular by adding information which creates a solution that prevents overfitting. The “something” we’re making regular in our ML context is the  “objective function”, something we try to minimize during the optimization problem.

Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. When you train a model such as a logistic regression model, you are choosing parameters that give you the best fit to the data. This means minimizing the error between what the model predicts for your dependent variable given your data compared to what your dependent variable actually is.

The problem comes when you have a lot of parameters (a lot of independent variables) but not too much data. In this case, the model will often tailor the parameter values to idiosyncrasies in your data -- which means it fits your data almost perfectly. However because those idiosyncrasies don't appear in future data you see, your model predicts poorly.

To solve this, as well as minimizing the error as already discussed, you add to what is minimized and also minimize a function that penalizes large values of the parameters. Most often the function is λΣθj2, which is some constant λ times the sum of the squared parameter values θj2. The larger λ is the less likely it is that the parameters will be increased in magnitude simply to adjust for small perturbations in the data. In your case however, rather than specifying λ, you specify C=1/λ.

--C  The parameters are numbers that tell the model what to do with the characteristics, whereas the hyperparameters instruct the model on how to choose parameters. Regularization will penalize the extreme parameters, the extreme values in the training data leads to overfitting.
A high value of C tells the model to give more weight to the training data. A lower value of C will indicate the model to give complexity more weight at the cost of fitting the data. Thus, a high Hyper Parameter value C indicates that training data is more important and reflects the real world data, whereas low value is just the opposite of this.

HyperDrive Selected ****Best run**** model ****hyperparameters**** are --C: 1.0 and --max_iter: 10



***AzureML HyperDrive Experiment Running***


![Azure ML Hyperdrive experiment running](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_12-39-03.png)


![Azure ML Hyperdrive experiment Accuracy plot runtime](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_12-39-34.png)


***AzureML HyperDrive Experiment Run Complete***


![Azure ML Hyperdrive experiment run complete.](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-26-43.png)


***AzureML HyperDrive Experiment Accuracy plot***


![Azure ML Hyperdrive experiment complete](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-28-27.png)


***AzureML HyperDrive Experiment Selected Best Run***


![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-32-10.png)


***AzureML HyperDrive Experiment Model Deployed***

![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/RegisteredModel.png)


***AzureML HyperDrive Experiment Service Enpoint Test***

![Azure ML Hyperdrive experiment Test 1](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/HyperDriveModelTest1.png)


Postman Test

![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/HyperDriveModelTest2.png)



# =====================================
# HyperParameter tuning using Keras tuner
# =====================================



[Keras Deep Learning Model and HyperParameter tuning](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/TFKerasHyperParameterV2.ipynb)



### References

[Living in the age of information overload](https://tom-stevenson.medium.com/we-are-living-in-the-age-of-information-overload-720ea5d31afb)

[Kaggle Spam-Ham SMS classification Data](https://www.kaggle.com/code/rumbleftw/beginner-friendly-spam-ham-sms-classification/data?select=spam.csv)


[](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406#:~:text=Azure%20AutoML%20is%20a%20cloud,pre%2Dprocess%20the%20input%20dataset.)

[](https://azure.microsoft.com/mediahandler/files/resourcefiles/automated-ml/Automated%20ML%20Infographic.pdf)