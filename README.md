# nd00333-capstone
Udacity AzureML Final Project

## Feedback Response

*  [Feedback Response AutoML](#feedback-response-updates-automl-09-26-2022)

*  [Feedback Response Hyperdrive run](#feedback-response-hyperdrive-run-updates-09-26-2022)
  

# SMS Spam Detection using Azure Machine learning

Problem of Information overload need to be recognized and addressed.

        A wealth of information creates a poverty of attention. 
                                          - Herbert Simon

One of the answers to the problem is to filter out the noise(spam) in this case from SMS.

### Overview

Spam is unsolicited and unwanted messages sent electronically  and whose content may be malicious. SMS spam is typically transmitted over a mobile network. SMS mode is more dangerous in many ways one is by the fact that SMS is usually regarded by the user as a safer, more trustworthy form of communication than other sources, e. g., emails.

This problem is solved using three different approaches:

        1. Using Azure AutoML Python SDK
        2. Using Azure HyperDrive Parameter tuning
        3. Using Keras Tensorflow and Keras tuner Neural Network 

# =====================================
# Auto ML using Python SDK
# =====================================

***[Spam Detection Auto ML Python SDK Project Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/notebook-smsspam-automl-0915-v1.ipynb)***

> To execute notebook:
>       
>       a. Load dataset 'UdacityPrjEmailSpamDataSet'
> 
>       b. copy config.json to same folder and .ipynb notebook file.
> 
>       c. Update Subscription Id, Resource Group and Workspace.

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

SMS Spam detection is a classification type problem where y (v1 in this dataset) the class to be predicted has a cardinality of 2(binary classification ham or spam) and x (v2 message). Featurization is set to 'auto' for AutoMl to automatically pick and choose features and run with it. For text columns auto ml engages bag of words, pre-trained word embedding and text target encoding techniques. 

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

a. Use AutoML run get output to retrieve best run. Using best run get model name. 

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

## =================================================

## Feedback response updates AutoML 09-26-2022

Please refer to the notebook link below for the captured details:

[Link to AutoML run new notebook](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/notebook-smsspam-automl-0924-v2.ipynb)

**1. An Overview of Dataset used**

***Data is a CSV file with five columns, 5572 rows total size of 504KB. Column v1 identifies message as ham or spam, column v2 is the message and column3, column4, column5 are additional messages. Each of object type text.***

**86.6% of data is 'ham' and 13.4% data is 'spam'**

![Dataset Overview](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_dataset_overview_0926_1.png)


**2. An Overview of Best Parameters Generated by AutoML model**

***Best run - Fitted Model parameters***

![AutoML Best Model parameters](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_model_params.png)

Stacking is an ensemble machine learning algorithm that learns how to best combine the predictions from multiple well-performing machine learning models. The benefit of stacking is that it can harness the capabilities of a range of well-performing models on a classification or regression task and make predictions that have better performance than any single model in the ensemble.

Highlevel overview of AutoML generated model parameters 

__class_weight__ dict or ‘balanced’, default=None
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

__eta0__ float, default=0.01
The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.01.

__fit_intercept__ bool, default=True
Whether the intercept should be estimated or not. If False, the data is assumed to be already centered and intercept is forced to the origin (0,0)

__l1_ratio__ is a parameter in a [0,1] range weighting l1 vs l2 regularisation. Hence the amount of l1 regularisation is l1_ratio * 1./C

__learning_rate__ learning rate determines how rapidly we update the parameters. If the learning rate is too large, we may "overshoot" the optimal value. Similarly, if it is too small, we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

__max_iter__ int, default=1000
The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.

__meta-learner__ refers to machine learning algorithms that learn from the output of other machine learning algorithms. meta-learning algorithms typically refer to ensemble learning algorithms like stacking that learn how to combine the predictions from ensemble members

__LogisticRegressionCV__ is a class that implements cross-validation inside it. This class will train multiple LogisticRegression models and return the best one.

### Future Improvements

**3. A short overview of how to improve the project in the future**

***Current performance of AutoML run - Confusion matrix:***

![AutoML Confusion matrix](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/automl_confusion_matrix.png)

Question of How well your model performs on new __unseen data__? 

Better the model, better it performs on unseen data. 

The problem of __Data imbalance__ and __Overfitting__ is in the way of building good machine learning model. 

__Imbalanced Data__ refers to datasets where the target class has an unseen distribution of observations. i.e., one class (majority class) has a very high number of observations and the other (minority class) has very low number of observations.

__Overfitting__ occurs when a model fits the training data too well, and as a result can't accurately predict on new unseen test data

### Addressing the problem of Data imbalance and Overfitting of model 

Most of the Machine learning algorithms expect __balanced data__ input to perform better on unseen data.

In our example: There's only 13.4% __spam__ samples in our dataset.

Problem at hand is to detect __spam__ hence to properly train the algorithm, one can think of is to oversample 'spam' type messages. There are different techniques to do this one is by simply making more copies of minority class data (not a very perfect approach)

 __Synthetic Minority Over-sampling Technique__ or __SMOTE__. Is an advanced technique and improvement over simply duplicating minority class.  SMOTE synthesize new examples from the minority class using k-nearest neighbors synthetic technique.

Adding samples of spam to matchup to the size of ham. Using above technique can help address data imbalance problem. 

Other techniques like __Use of more data__  is a simple one that can help prevent overfitting. Usage of more data helps model reach solutions that are more flexible as it accommodates more conditions. 

__Prevent target leak__ this occurs when your model have access to data that it normally doesn't at the time of prediction. Due to this model may have inflated performance during training but poor performance when deployed to predict on real data.  

__Removing of noisy or least important features__ from the model reduces the complexity of the model and in turn help prevent overfitting.

### There are also  built in capabilities of Automated ML to help deal with imbalanced data and overfitting

A __weight column__: automated ML weights as input, causing classes in the data to be weighted up or down, which can be used to make a class more or less "important". __minority class__ will be given more weight to balance overall outcome of the model.

The algorithms used by automated ML detect imbalance when the number of samples in the minority class is equal to or fewer than 20% of the number of samples in the majority class.

Automated ML has built in __Regularization__ of minimizing a cost function to penalize complex and overfitted models.

__Cross-validation__ built in AutomatedML is the process of taking many subsets of your full training data and training a model on each subset. Pick the subset that gives high accuracy an
When doing CV, you provide validation dataset, and automated ML train your model and tune hyperparaments. 

***Azure AutoML future improvements***

*Selecting and Fine tuning using one or more effective techniques can help improve your model performance. In certain cases, it could also be totally new __innovative approach__ specific to the problem can result in improved outcomes. Learning and improving for experimentation is the key*. 

# =====================================
# Azure HyperDrive execution using Logistic Regression
# =====================================

## SMS Spam Experiment using HyperDrive Logistic Regression Notebook

***[Azure ML Hyperdrive run Notebook link](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/nb-lr-0921-v12.ipynb)***


**Azure AutoML Hyperdrive Overview**

![Azure AutoML Overview](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/automl_images/MicrosoftAutoML.png)

After selection of the Dataset for your use case, define configuration and constraints followed with Feature engineering and applicable and effective Algorithm selection. You would then continue with Hyperparameter tuning experimentation to search for the best model. Select model from the best run for deployment. 

****Featurization**** is the process of applying techniques of Feature Engineering to the dataset that will help machine learning algorithm learn better.

In this case below listed feature engineering techniques are engaged

        a. Elimination of stop words

        b. Replacement of target attribute with numeric 1 for spam and 0 for ham.

        c. Clean text of any punctuations, whitespaces and web addresses.

        d. Stemming using NLTK Snowball Stemmer. A process of reducing a word to its base word,
        such that words of similar kind lie under a common stem.

        e. TfidfVectorizer is an important NLP feature that converts text to a matrix of TF-IDF features.
           (TF-IDF is an abbreviation of Term Frequency - Inverse Document Frequency)  


****Model Selection**** Logistic Regression sounds to be the most natural fit to binary Spam classification problem (citing whitepaper from International Journal in below references). 
        
Logistic function is a S-shaped curve (also known as a sigmoid curve) that for a given set of input variables, produces an output between 0 and 1 which can be used to represent probability and with a given threshold would classify one way or the other. 

**HyperParameters**

Overfitting is a significant issue in the field of data science that needs to handled carefully in order to build a robust and accurate model. Overfitting arises when a model tries to fit the training data so well that it cannot generalize to new observations. 

Regularization techniques and strength are to avoid Overfitting or UnderFitting model training. 

***solver='liblinear'***

LIBLINEAR is an open source library for large-scale linear classification. It supports logistic
regression and linear support vector machines. Experiments demonstrate that LIBLINEAR is very efficient
on large sparse data sets.

Keywords: large-scale linear classification, logistic regression, support vector machines,
open source, machine learning

***penalty='l1'***

L1 regularization forces the weights of uninformative features to be zero by substracting a small amount from the weight at each iteration and thus making the weight zero, eventually.

L1 regularization penalizes |weight|.

Regularization techniques play a vital role in the development of machine learning models. Especially complex models, like neural networks, prone to overfitting the training data. Broken down, the word “regularize” states that we’re making something regular. In a mathematical or ML context, we make something regular by adding information which creates a solution that prevents overfitting. The “something” we’re making regular in our ML context is the  “objective function”, something we try to minimize during the optimization problem.

Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. When you train a model such as a logistic regression model, you are choosing parameters that give you the best fit to the data. This means minimizing the error between what the model predicts for your dependent variable given your data compared to what your dependent variable actually is.

The problem comes when you have a lot of parameters (a lot of independent variables) but not too much data. In this case, the model will often tailor the parameter values to idiosyncrasies in your data -- which means it fits your data almost perfectly. However because those idiosyncrasies don't appear in future data you see, your model predicts poorly.

To solve this, as well as minimizing the error as already discussed, you add to what is minimized and also minimize a function that penalizes large values of the parameters. Most often the function is λΣθj2, which is some constant λ times the sum of the squared parameter values θj2. The larger λ is the less likely it is that the parameters will be increased in magnitude simply to adjust for small perturbations in the data. In your case however, rather than specifying λ, you specify C=1/λ.

***--C***  The parameters are numbers that tell the model what to do with the characteristics, whereas the hyperparameters instruct the model on how to choose parameters. Regularization will penalize the extreme parameters, the extreme values in the training data leads to overfitting.
A high value of C tells the model to give more weight to the training data. A lower value of C will indicate the model to give complexity more weight at the cost of fitting the data. Thus, a high Hyper Parameter value C indicates that training data is more important and reflects the real world data, whereas low value is just the opposite of this.

***max_iterint*** Maximum number of iterations taken for the solvers to converge.

***Early termination policy***

**early_termination_policy = BanditPolicy(evaluation_interval=2,slack_factor=0.2)**

Bandit is an early termination policy based on slack factor/slack amount and evaluation interval. The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.

HyperDrive Selected ****Best run**** model ****hyperparameters**** are --C: 1.0 and --max_iter: 10

***Model Object:***

**model = LogisticRegression(solver='liblinear', penalty='l1', C=args.C, max_iter=args.max_iter)**

***AzureML HyperDrive Experiment Running***

![Azure ML Hyperdrive experiment running](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_12-39-03.png)

![Azure ML Hyperdrive experiment Accuracy plot runtime](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_12-39-34.png)

***AzureML HyperDrive Experiment Run Complete***

![Azure ML Hyperdrive experiment run complete.](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-26-43.png)

***AzureML HyperDrive Experiment Accuracy plot***

![Azure ML Hyperdrive experiment complete](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-28-27.png)

***AzureML HyperDrive Experiment Selected Best Run***

![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/2022-09-21_13-32-10.png)

[link to Score.py used for Hyperdrive experiment testing](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/scripts/score_v10.py)


***AzureML HyperDrive Experiment Model Deployed***

![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/RegisteredModel.png)

***AzureML HyperDrive Experiment Service Enpoint Test***

![Azure ML Hyperdrive experiment Test 1](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/HyperDriveModelTest1.png)

Postman Test

![Azure ML Hyperdrive experiment done](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/HyperDriveModelTest2.png)

# Comparision and Conclusion

***Azure AutoML run had an accuracy of 99% and AUC of 99%, whereas Hyperdrive run picked best model accuracy is 94% and AUC 95%***

Both models had excelent performance but in the wild west of Spam world my next step as standout exercies is to look into Neural Network and Deep learning.


# Link to ScreenCast

[Screen Cast](https://github.com/Nazeer2013/nd00333-capstone/tree/master/finalproject/screencast#:~:text=2%20minutes%20ago-,zoom_0.mp4,-screencast)

## =================================================

## Feedback response Hyperdrive run updates 09-26-2022

**1. Screenshot of Hyperdrive run active model endpoint**

![Screenshot of Hyperdrive run active model endpoint](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/DeployedEnpointHyperdrive.png)

**2. Hyperdrive run environment details file**

![Hyperdrive run file containing the environment details](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/hyperdrive_images/hyperdrive_conda_env_yml.png)

# =====================================
# HyperParameter tuning using Keras tuner
# =====================================

# Standout Exercies 

[Keras Deep Learning Model and HyperParameter tuning](https://github.com/Nazeer2013/nd00333-capstone/blob/master/finalproject/TFKerasHyperParameterV2.ipynb)

***Keras*** is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research.

***TensorFlow 2*** is an end-to-end, open-source machine learning platform. You can think of it as an infrastructure layer for differentiable programming. It combines four key abilities:

Efficiently executing low-level tensor operations on CPU, GPU, or TPU.

Computing the gradient of arbitrary differentiable expressions.

Scaling computation to many devices, such as clusters of hundreds of GPUs.

Exporting programs ("graphs") to external runtimes such as servers, browsers, mobile and embedded devices.

### References

[Living in the age of information overload](https://tom-stevenson.medium.com/we-are-living-in-the-age-of-information-overload-720ea5d31afb)

[Kaggle Spam-Ham SMS classification Data](https://www.kaggle.com/code/rumbleftw/beginner-friendly-spam-ham-sms-classification/data?select=spam.csv)

[Azure AutoML](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406#:~:text=Azure%20AutoML%20is%20a%20cloud,pre%2Dprocess%20the%20input%20dataset.)

[Azure AutoML Media](https://azure.microsoft.com/mediahandler/files/resourcefiles/automated-ml/Automated%20ML%20Infographic.pdf)

[Liblinear Whitepaper](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf)

[CMU Project](https://www.cs.cmu.edu/~gpekhime/Projects/CSC2515/project.pdf)

[Logistic Regression White Paper](https://ijisrt.com/assets/upload/files/IJISRT21SEP728.pdf)

[Logistic Regression scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

[Kaggle Spam and Ham classificaton](https://www.kaggle.com/code/rumbleftw/beginner-friendly-spam-ham-sms-classification)

[dentify-models-with-imbalanced-data](https://learn.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls#identify-models-with-imbalanced-data)

[LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)

