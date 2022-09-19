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



#import azureml.automl.core
#from azureml.automl.core.shared import logging_utilities, log_server
#from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"v2": pd.Series(["example_value"], dtype="object"), "Column4": pd.Series(["example_value"], dtype="object"), "Column5": pd.Series(["example_value"], dtype="object"), "Column6": pd.Series(["example_value"], dtype="object")})
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
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_v7.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    #log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
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
            #--------------------------------------------------------------------------
            print("Nazeer: in run 1: data type: ")
            print(type(data))
            print("Nazeer: in run 2: data type: ")
            print(data)
            print("Nazeer: in run 3: model type: ")
            print(type(model))
            df_msg_copy = data.get('v2')[0]

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

            result = model.predict(msg_mat)
            #-------------------------------------------------------------------------
            # result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
