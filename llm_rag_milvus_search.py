#!/usr/bin/env python
# coding: utf-8


    
# In[5]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import time
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[3]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
        
    pk_type=DataType.VARCHAR        
    embedding_type=DataType.FLOAT_VECTOR
    
    try:
        collection_name=param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name="default_collection"
        print("Using default collection")
    
    print("start connecting to Milvus")
    # this hostname may need changing to a specific local docker network ip address depending on docker configuration
    connections.connect("default", host="milvus-standalone", port="19530")

    collection_exists = utility.has_collection(collection_name)
    
    if collection_exists:
        print(f"The collection {collection_name} already exists")
        collection = Collection(collection_name)
        collection.load()
    else:
        print(f"The collection {collection_name} does not exist")
        raise Exception("The collection {collection_name} does not exist. Create it by sending data to a collection with that name using the push_to_milvus algo.")
    
    model['collection']=collection
    model['collection_name']=collection_name
    return model







    
# In[12]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[13]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    try:
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local = 0
        print("Downloading embedding model by default") 
        
    try:
        embedder_name=param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        print("Using all-MiniLM-L6-v2 as embedding model by default") 
        
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints") 

    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
    except Exception as e:
        cols = {"Results": [f"Failed to load embedding model. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns
    
    try:
       top_k=int(param['options']['params']['top_k'])
    except:
        top_k=5
        print("Using top 5 results by default")
        
    try:
        splitter=param['options']['params']['splitter']
    except:
        splitter="|"
    
    try:
        text_column = df['text'].astype(str).tolist()
    except Exception as e:
        cols = {"Results": [f"Failed to read input data. ERROR: {e}. Make sure you have an input field called text"]}
        returns = pd.DataFrame(data=cols)
        return returns
    try:
        vector_column = []
        for text in text_column:
            vector_column.append(transformer_embedder.get_text_embedding(text))
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        output_fields = [item.name for item in model['collection'].schema.fields]
        output_fields.remove('embeddings')
        results = model['collection'].search(data=vector_column, anns_field="embeddings", param=search_params, limit=top_k, output_fields=output_fields)
        l = []
        f = []
        d = []
        output_fields.remove('label')
        for result in results:
            x = ''
            y = ''
            z = ''
            for r in result:
                t = {}
                for field in output_fields:
                    t[field] = r.entity.get(field)
                x += str(t)
                x += splitter
                y += r.entity.get('label')
                y += splitter
                z += str(round(r.distance,4))
                z += splitter
            xs = x.rstrip(splitter)
            ys = y.rstrip(splitter)
            zs = z.rstrip(splitter)
            l.append(ys) 
            f.append(xs)
            d.append(zs)
    except Exception as e:
        cols = {"Results": [f"Failed to conduct vectorsearch. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns
    
    
    cols = {"Results": l, "Fields": f, "Distance": d}
    returns = pd.DataFrame(data=cols)
    return returns







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[18]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    try:
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local = 0
        print("Downloading embedding model by default") 
        
    try:
        embedder_name=param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        print("Using all-MiniLM-L6-v2 as embedding model by default") 
        
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints") 

    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
    except Exception as e:
        cols = {"Results": [f"Failed to load embedding model. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns
    
    try:
       top_k=int(param['options']['params']['top_k'])
    except:
        top_k=5
        print("Using top 5 results by default")
        
    try:
        splitter=param['options']['params']['splitter']
    except:
        splitter="|"
    
    try:
        text_column = df['text'].astype(str).tolist()
    except Exception as e:
        cols = {"Results": [f"Failed to read input data. ERROR: {e}. Make sure you have an input field called text"]}
        returns = pd.DataFrame(data=cols)
        return returns
    try:
        vector_column = []
        for text in text_column:
            vector_column.append(transformer_embedder.get_text_embedding(text))
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        output_fields = [item.name for item in model['collection'].schema.fields]
        output_fields.remove('embeddings')
        results = model['collection'].search(data=vector_column, anns_field="embeddings", param=search_params, limit=top_k, output_fields=output_fields)
        l = []
        f = []
        d = []
        output_fields.remove('label')
        for result in results:
            x = ''
            y = ''
            z = ''
            for r in result:
                t = {}
                for field in output_fields:
                    t[field] = r.entity.get(field)
                x += str(t)
                x += splitter
                y += r.entity.get('label')
                y += splitter
                z += str(round(r.distance,4))
                z += splitter
            xs = x.rstrip(splitter)
            ys = y.rstrip(splitter)
            zs = z.rstrip(splitter)
            l.append(ys) 
            f.append(xs)
            d.append(zs)
    except Exception as e:
        cols = {"Results": [f"Failed to conduct vectorsearch. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns
    
    
    cols = {"Results": l, "Fields": f, "Distance": d}
    returns = pd.DataFrame(data=cols)
    return returns

















