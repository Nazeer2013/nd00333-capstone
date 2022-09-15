import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow.keras as K

import re

from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"v2": pd.Series(["example_value"], dtype="object"), "Column4": pd.Series(["example_value"], dtype="object"), "Column5": pd.Series(["example_value"], dtype="object"), "Column6": pd.Series(["example_value"], dtype="object")})
output_sample = np.array(["example_value"])
method_sample = StandardPythonParameterType("predict")

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

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a keras model
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sms_spam_check_model_v1.h5')
    print("Env :",os.getenv('AZUREML_MODEL_DIR'))
    print("Model Path: ", model_path)
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    try:
        print("*******************Loading model from path.*************************")
        # load the model
        model = K.models.load_model(model_path)
        print("Loading successful......................................")
    except Exception as e:
        print("Exception in init Model Load : ", e)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))


def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            print("data: ", data)
            print("data type: ", type(data))
            # <class 'pandas.core.frame.DataFrame'>
            print("model: ", model)
            print("model type: ", type(model))
            # <class 'keras.engine.sequential.Sequential'>
            print("****************************************************")
            text = data.get('v2')[0]
            processed_text = clean_text(text)
            print("Processed Text: ", processed_text)
            tokenizer = K.preprocessing.text.Tokenizer()
            tokenizer.oov_token = '<oovToken>'
            # tokenizer.fit_on_texts("")
            final_text = K.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([processed_text]), padding='pre', maxlen=171)
            print("Final Text: ", final_text)
            result = model.predict(final_text)
            print("************************Got Results: ********************")
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, np.ndarray):
            print("result instance of numpy ndarry: ")
            print("result shape: ", result.shape)
            print("Ham or Spam: ",np.int32(np.rint(result[0,0])))
            if 0 == (np.int32(np.rint(result[0,0]))):
                str_result = "Ham"
            else:
                str_result = "Spam"   
        return json.dumps({"result": str_result})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})