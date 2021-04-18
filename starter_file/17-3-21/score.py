import joblib
import numpy as np
import os
import pickle
import json
import pandas as pd



# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    print("HHHHH1")

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.pkl'

    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    print("Found model:",os.path.isfile(model_path))
    
    print(model_path)

    model = joblib.load(model_path)


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.

def run(data):
    # Use the model object loaded by init().
    
    data_load = json.loads(data)
    data = pd.DataFrame.from_dict(data_load['data'])
    
    result = model.predict(data)

    # You can return any JSON-serializable object.
    return result.tolist()