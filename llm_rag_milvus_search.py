#!/usr/bin/env python
# coding: utf-8


    
# In[5]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import time
from llama_index.core import VectorStoreIndex, ServiceContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.bridge.pydantic import BaseModel, StrictFloat, StrictInt, StrictStr
from llama_index.core.schema import BaseComponent, BaseNode, TextNode
from app.model.llm_utils import create_llm, create_embedding_model
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
MILVUS_ENDPOINT = "http://milvus-standalone:19530"







    
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
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")

    if service == "huggingface" or service == "ollama":
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
    
        if embedder_name == 'intfloat/multilingual-e5-large':
            embedder_dimension = 1024
        elif embedder_name == 'all-MiniLM-L6-v2':
            embedder_dimension = 384
        else:
            try:
                embedder_dimension=int(param['options']['params']['embedder_dimension'])
            except:
                embedder_dimension=384
            
        if service == "huggingface" and use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints") 
    else:
        try:
            embedder_dimension=int(param['options']['params']['embedder_dimension'])
        except:
            cols = {"Results": ["Please specify the embedder_dimension parameter for the embedding model dimensions"]}
            returns = pd.DataFrame(data=cols)
            return returns

    try:
        if service == "huggingface" or service == "ollama":
            embedder, m = create_embedding_model(service=service, model=embedder_name)
        else:
            embedder, m = create_embedding_model(service=service)

        if embedder is not None:
            print(m)
        else:
            cols = {"Results": [f"ERROR in embedding model loading: {m}. "]}
            returns = pd.DataFrame(data=cols)
            return returns
    except Exception as e:
        cols = {"Results": [f"Failed to initiate embedding model. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns

    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        cols = {"Results": ["Please specify a collection_name parameter as the vector search target"]}
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
        Settings.llm = None
        Settings.embed_model = embedder
        vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
    except Exception as e:
        cols = {"Results": [f"Could not load collection. ERROR: {e}"]}
        result = pd.DataFrame(data=cols)
        return result
    
    try:
        query = df['text'].astype(str).tolist()[0]
    except Exception as e:
        cols = {"Results": [f"Failed to read input data. ERROR: {e}. Make sure you have an input field called text"]}
        returns = pd.DataFrame(data=cols)
        return returns

    retrieved_nodes = retriever.retrieve(query)
    try:
        result = pd.DataFrame([{"Score": node.score, "Result": node.text} for node in retrieved_nodes])
        meta = pd.DataFrame([node.metadata for node in retrieved_nodes])
        result[meta.columns] = meta
    except Exception as e:
        cols = {"Results": [f"ERROR: {e}"]}
        return pd.DataFrame(data=cols)
    if not len(result):
        cols = {"Results": ["ERROR: No result returned"]}
        return pd.DataFrame(data=cols)
    
    return result







    
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
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")

    if service == "huggingface" or service == "ollama":
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
    
        if embedder_name == 'intfloat/multilingual-e5-large':
            embedder_dimension = 1024
        elif embedder_name == 'all-MiniLM-L6-v2':
            embedder_dimension = 384
        else:
            try:
                embedder_dimension=int(param['options']['params']['embedder_dimension'])
            except:
                embedder_dimension=384
            
        if service == "huggingface" and use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints") 
    else:
        try:
            embedder_dimension=int(param['options']['params']['embedder_dimension'])
        except:
            cols = {"Results": ["Please specify the embedder_dimension parameter for the embedding model dimensions"]}
            returns = pd.DataFrame(data=cols)
            return returns

    try:
        if service == "huggingface" or service == "ollama":
            embedder, m = create_embedding_model(service=service, model=embedder_name)
        else:
            embedder, m = create_embedding_model(service=service)

        if embedder is not None:
            print(m)
        else:
            cols = {"Results": [f"ERROR in embedding model loading: {m}. "]}
            returns = pd.DataFrame(data=cols)
            return returns
    except Exception as e:
        cols = {"Results": [f"Failed to initiate embedding model. ERROR: {e}"]}
        returns = pd.DataFrame(data=cols)
        return returns

    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        cols = {"Results": ["Please specify a collection_name parameter as the vector search target"]}
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
        Settings.llm = None
        Settings.embed_model = embedder
        vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
    except Exception as e:
        cols = {"Results": [f"Could not load collection. ERROR: {e}"]}
        result = pd.DataFrame(data=cols)
        return result
    
    try:
        query = df['text'].astype(str).tolist()[0]
    except Exception as e:
        cols = {"Results": [f"Failed to read input data. ERROR: {e}. Make sure you have an input field called text"]}
        returns = pd.DataFrame(data=cols)
        return returns

    retrieved_nodes = retriever.retrieve(query)
    try:
        result = pd.DataFrame([{"Score": node.score, "Result": node.text} for node in retrieved_nodes])
        meta = pd.DataFrame([node.metadata for node in retrieved_nodes])
        result[meta.columns] = meta
    except Exception as e:
        cols = {"Results": [f"ERROR: {e}"]}
        return pd.DataFrame(data=cols)
    if not len(result):
        cols = {"Results": ["ERROR: No result returned"]}
        return pd.DataFrame(data=cols)
    
    return result

















