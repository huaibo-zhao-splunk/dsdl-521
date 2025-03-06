#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import requests
# ...
# global constants
ollama_url = "http://ollama:11434"
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[4]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[ ]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model







    
# In[ ]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[12]:


# from fit command, we will pass parameters model and prompt.
# sample prompt: You will examine if the email content given by the user is phishing. 
#                Only output **Phishing** if the content is phishing. 
#                Only output **Legit** if the email is legitimate. Do not give extra information.
def apply(model,df,param):
    try:
        X = df["text"].values.tolist()
    except:
        cols={'Result': ["ERROR: Please make sure you have a field in the search result named \'text\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns

    try:
        prompt = param['options']['params']['prompt'].strip("\"")
    except:
        cols={'Result': ["ERROR: Please make sure you set the parameter \'prompt\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns

    try:
        model_name = param['options']['params']['model_name'].strip("\"")
    except:
        cols={'Result': ["ERROR: Please make sure you set the parameter \'model_name\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns
        
    uri = f"{ollama_url}/api/chat"
    headers = {'Content-Type': 'application/json'}
    outputs_label = []
    outputs_duration = []

    
    for i in range(len(X)):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": X[i]}
        ]
        
        data = {
            "model": param['options']['params']['model_name'].strip("\""),
            "messages": messages,
            "stream": False,
        }
        
        data = json.dumps(data)
        
        response = requests.post(uri, headers=headers, data=data).json()
        try:   
            outputs_label.append(response['message']['content'])
            duration = round(int(response['total_duration']) / 1000000000, 2)
            duration = str(duration) + " s"
            outputs_duration.append(duration)
        except Exception as e:
            if response:
                outputs_label.append(response)
                outputs_duration.append("ERROR")
            else:
                outputs_label.append(f"ERROR: {e}")
                outputs_duration.append("ERROR")
        
    cols={'Result': outputs_label, 'Duration': outputs_duration}
    returns=pd.DataFrame(data=cols)
    return returns







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[21]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    try:
        X = df["text"].values.tolist()
    except:
        cols={'Result': ["ERROR: Please make sure you have a field in the search result named \'text\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns

    try:
        prompt = param['options']['params']['prompt'].strip("\"")
    except:
        cols={'Result': ["ERROR: Please make sure you set the parameter \'prompt\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns

    try:
        model_name = param['options']['params']['model_name'].strip("\"")
    except:
        cols={'Result': ["ERROR: Please make sure you set the parameter \'model_name\'"], 'Duration': ["ERROR"]}
        returns=pd.DataFrame(data=cols)
        return returns
        
    uri = f"{ollama_url}/api/chat"
    headers = {'Content-Type': 'application/json'}
    outputs_label = []
    outputs_duration = []

    
    for i in range(len(X)):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": X[i]}
        ]
        
        data = {
            "model": param['options']['params']['model_name'].strip("\""),
            "messages": messages,
            "stream": False,
        }
        
        data = json.dumps(data)
        
        response = requests.post(uri, headers=headers, data=data).json()
        try:   
            outputs_label.append(response['message']['content'])
            duration = round(int(response['total_duration']) / 1000000000, 2)
            duration = str(duration) + " s"
            outputs_duration.append(duration)
        except Exception as e:
            if response:
                outputs_label.append(response)
                outputs_duration.append("ERROR")
            else:
                outputs_label.append(f"ERROR: {e}")
                outputs_duration.append("ERROR")
        
    cols={'Result': outputs_label, 'Duration': outputs_duration}
    returns=pd.DataFrame(data=cols)
    return returns







