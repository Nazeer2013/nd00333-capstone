
# Import libraries
import argparse, joblib, os
from azureml.core import Run

import logging
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import datasets
import tensorflow as tf
import regex as re
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import tempfile
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import Dense


from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import models, layers

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score, roc_curve

import azureml.core
from azureml.core import Workspace
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace

from azureml.core.dataset import Dataset


# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Input dataset
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

#hyperdrive_feature
parser.add_argument("--hyperdrive_feature", type=bool, dest='hyperdrive_feature', help='hyperdrive feature')

# Hyperparameters
parser.add_argument('--units', type=int, default=64, help="Number of nodes")
parser.add_argument('--optimizer', type=str, default='adam', help="Algorithm of Choice")

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values 
run.log("Number of Nodes:", np.int(args.units))  
run.log("Algorithm of Choice:", np.str(args.optimizer))  

# load the email spam dataset -- Get the training data from the input
print("Loading Email Spam Data...")
df = run.input_datasets['training_data'].to_pandas_dataframe() 

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

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.oov_token = '<oovToken>'
tokenizer.fit_on_texts(df.v2)
vocab = tokenizer.word_index
vocabCount = len(vocab)+1


# Split Train and Test
SPLIT = 5000

# Split data into training set and test set
xTrain = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df.v2.to_numpy()), padding='pre', maxlen=171)
yTrain = df.v1.to_numpy()

dim = xTrain.shape[1]
xTest = xTrain[SPLIT:]
yTest = yTrain[SPLIT:]

xTrain = xTrain[:SPLIT]
yTrain = yTrain[:SPLIT]

# Train a Keras Sequential classification model without the specified hyperparameters
print('Training a classification model')

#------------------------------------------------------------
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Embedding(input_dim=vocabCount+1, output_dim=64, input_length=dim))
#model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()

#--------------------------------------------------------------

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocabCount+1, output_dim=64, input_length=dim))
model.add(tf.keras.layers.GlobalAveragePooling1D())
# for i in range(args.num_layers):
model.add(tf.keras.layers.Dense(args.units, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])
model.summary()


hist = model.fit(xTrain, yTrain, batch_size=32, epochs=100, initial_epoch=6, validation_data=(xTest, yTest))

# calculate accuracy
y_hat = model.predict(xTest)
acc = np.average(y_hat == yTest)
print('Accuracy:', acc)
run.log('Accuracy', np.float64(acc))

print('hist type: ', type(hist))

# calculate AUC
# y_scores = model.predict_proba(xTest)
# auc = roc_auc_score(yTest,y_scores[:,1])
# print('AUC: ' + str(auc))
# run.log('AUC', np.float(auc))

#for x in ['acc','val_acc']:
    #run.log('Accuracy', hist.history[x[0]])
#    print (hist.history[x])
#    print (hist.history[x[0]])
#    plt.plot(hist.history[x])

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/emailspam_model09122022.pkl')
    

run.complete()
