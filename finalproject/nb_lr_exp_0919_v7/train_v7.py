
# Import libraries
import argparse, joblib, os
from azureml.core import Run

import logging
import os
import csv
import string
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from nltk.stem import SnowballStemmer

from sklearn.metrics import accuracy_score

import regex as re

import pickle
import tempfile

from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

import azureml.core
from azureml.core.dataset import Dataset

# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Input dataset
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

# Hyperparameters
#parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
# parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')

parser.add_argument('--C', type=float, default=1.0, help="indicates regularization")
parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations")

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
# run.log('learning_rate',  np.float(args.learning_rate))
# run.log('n_estimators',  np.int(args.n_estimators))

run.log("Regularization Strength:", np.float(args.C))
run.log("Max iterations:", np.int(args.max_iter))

 
# load the sms spam dataset -- Get the training data from the input
print("Loading SMS Spam Data...")
df = run.input_datasets['training_data'].to_pandas_dataframe() 

#--------------------------Prepare Data-------------------------------------------------
# Cleanup and Prepare Data # Find and eliminate stop words 
nltk.download('stopwords')
stop_words= set(stopwords.words("english"))
stop_words.update(['https', 'http', 'amp', 'CO', 't', 'u', 'new', "I'm", "would"])


spam = df.query("v1=='spam'").v2.str.cat(sep=" ")
ham = df.query("v1=='ham'").v2.str.cat(sep=" ")

# convert spam to 1 and ham to 0
df = df.replace('spam', 1)
df = df.replace('ham', 0)

# Clean the text
def clean_text(text):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = text.replace('.', '')
    text = whitespace.sub(' ', text)
    text = web_address.sub('', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    return text.lower()

df.v2 = [clean_text(item) for item in df.v2]

#---------------------More Data Prep-----------#
df = df.drop(['Column3', 'Column4', 'Column5'], axis = 1)

df_msg_copy = df['v2'].copy()

# vectorizer = TfidfVectorizer(stop_words='english')

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

df_msg_copy = df_msg_copy.apply(text_preprocess)

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

df_msg_copy = df_msg_copy.apply(stemmer)
vectorizer = TfidfVectorizer(stop_words='english')
msg_mat = vectorizer.fit_transform(df_msg_copy)


# Split Train and Test
xTrain, xTest, yTrain, yTest = train_test_split(msg_mat, df.v1, test_size=0.3, random_state=20)

# --------------------------End Prepare Data--------------------------------------------

# --------------------------Start Training----------------------------------------------
# Train a Logistic Regression classification model without the specified hyperparameters
print('Training a classification model')

model = LogisticRegression(solver='liblinear', penalty='l1', C=args.C, max_iter=args.max_iter )
model.fit(xTrain, yTrain)
pred = model.predict(xTest)
acc = accuracy_score(yTest,pred)

# ---------------------------End Training------------------------------------------------

run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(xTest)
auc = roc_auc_score(yTest,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model_v7.pkl')


run.complete()
