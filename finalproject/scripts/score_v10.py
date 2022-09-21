# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

import regex as re

import pickle
import tempfile

from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# import azureml.core.authentication
#from azureml.automl.core.shared import logging_utilities, log_server
#from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.core import Workspace, Environment 
from azureml.core.authentication import ServicePrincipalAuthentication

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

# input_sample = pd.DataFrame({"v2": pd.Series(["example_value"], dtype="object"), "Column4": pd.Series(["example_value"], dtype="object"), "Column5": pd.Series(["example_value"], dtype="object"), "Column6": pd.Series(["example_value"], dtype="object")})
input_sample = pd.DataFrame({"v2": pd.Series(["example_value"], dtype="object")})
# "Column4": pd.Series(["example_value"], dtype="object"), "Column5": pd.Series(["example_value"], dtype="object"), "Column6": pd.Series(["example_value"], dtype="object")})
output_sample = np.array(["example_value"])
method_sample = StandardPythonParameterType("predict")

try:
    #log_server.enable_telemetry(INSTRUMENTATION_KEY)
    #log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    global df
    global ws
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_v10.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    #log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})


    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")

        svc_pr_password = os.environ.get("AZUREML_PASSWORD")

        svc_pr = ServicePrincipalAuthentication(
        tenant_id="db05faca-c82a-4b9d-b9c5-0f64b6755421",
        service_principal_id="ed84bd8b-0d92-4b6d-be98-b0adcbd37cc0",
        service_principal_password="*******************************")

        
        ws = Workspace(
            subscription_id="16bc73b5-82be-47f2-b5ab-f2373344794c",
            resource_group="epe-poc-nazeer",
            workspace_name="nahmed30-azureml-workspace",
            auth=svc_pr
            )

        # print("****** Found workspace {} at location {} *******".format(ws.name, ws.location))
        logger.info("WS Loading successful........................................")

    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            key = 'UdacityPrjEmailSpamDataSet'
            smsspam_ds = ws.datasets[key]
            df = smsspam_ds.to_pandas_dataframe()
            logger.info("DF Loading successful........................................")

            #-----------------------Global Corpus--------------------------
            print("Dataframe as Global")
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

            df = df.drop(['Column3', 'Column4', 'Column5'], axis = 1)

            df_msg_copy = df['v2'].copy()

            #vectorizer = TfidfVectorizer(stop_words='english')

            def text_preprocess(text):
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
                return " ".join(text)

            df_msg_copy = df_msg_copy.apply(text_preprocess)

            def stemmer (text):
                text = text.split()
                words1 = ""
                for i in text:
                    stemmer = SnowballStemmer("english")
                    words1 += (stemmer.stem(i))+" "
                return words1

            df_msg_copy = df_msg_copy.apply(stemmer)
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit_transform(df_msg_copy)

            #-----------------------Global Corpus--------------------------

            message_txt = data.get('v2')[0]
            message_txt = message_txt.translate(str.maketrans('', '', string.punctuation))
            
            # def stemmer (text):
            message_txt = message_txt.split()

            words = ""
            for i in message_txt:
                stemmer = SnowballStemmer("english")
                words += (stemmer.stem(i))+" "
            
            # vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, use_idf=True, norm='l2', smooth_idf=True)

            words = [words]

            msg_mat = vectorizer.transform(words)

            result = model.predict(msg_mat)

            # result = result.replace(1, 'spam')
            # result = result.replace(0, 'ham')

            print("******************************************************")
            print(type(result))
            print(result)
            print(result[0])

            
            if 0 == (np.int32(result[0])):
                df_result =  pd.DataFrame("Ham")
            else:
                df_result =  pd.DataFrame("Spam")
            
                  

        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(df_result, pd.DataFrame):
            df_result = df_result.values
        return json.dumps({"result": df_result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
