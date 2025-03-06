#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import textwrap
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
LLM_ENDPOINT = "http://ollama:11434"









    
# In[6]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param





    
# In[10]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    return model





    
# In[19]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    info = {"message": "model trained"}
    return info







    
# In[20]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    X = df["query"].values.tolist()
    # "Documents" or "Logs"
    try:
        d_type = param['options']['params']['rag_type'].strip('\"')
    except:
        d_type = "Documents"
        print("Using Document knowledge type by default")

    try:
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local=0
        print("Download embedding model by default")
        
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        print("Using all-MiniLM-L6-v2 by default")

    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        cols = {"Response": ["ERROR: no collection specified. Please specify a vectorDB collection"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
        

    if embedder_name == 'intfloat/multilingual-e5-large':
        embedder_dimension = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        embedder_dimension = 384
    else:
        try:
            embedder_dimension = int(param['options']['params']['embedder_dimension'])
        except:
            embedder_dimension = 384
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints")
    
    try:
        top_k = int(param['options']['params']['top_k'])
    except:
        top_k = 5
        print("Using top 5 results by default")
        
    if d_type == "Documents":
        qa_prompt_str = (
            "Below are the context information.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information as well as necessary prior knowledge, "
            "answer the question: {query_str}\n"
        )
        chat_text_qa_msgs = [
            (
                "system",
                "You are an expert Q&A system that is trusted around the world. Always answer the query using the provided context information and reasoning as detailed as possible",
            ),
            ("user", qa_prompt_str),
        ]
    else:
        qa_prompt_str = (
            "Past log messages below are given as context information.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information as well as necessary prior knowledge, "
            "answer the question: {query_str}\n"
        )
        chat_text_qa_msgs = [
            (
                "system",
                "You are an expert Q&A system that is trusted around the world. Always answer the query using the provided context information and reasoning as detailed as possible",
            ),
            ("user", qa_prompt_str),
        ]
    
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    try:
        model = param['options']['params']['model_name'].strip('\"')
    except:
        cols = {"Response": ["ERROR: no LLM model name specified. Please choose a LLM model."], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        url = LLM_ENDPOINT
        llm = Ollama(model=model, base_url=url, request_timeout=6000.0)
    except Exception as e:
        cols = {"Response": [f"Could not load LLM. ERROR: {e}"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        Settings.llm = llm
        Settings.embed_model = transformer_embedder
        Settings.chunk_size = 1024
    except Exception as e:
        cols = {"Response": [f"Could not load embedder. ERROR: {e}"], "References": ["ERROR"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        if d_type == "Documents":
            vector_store = MilvusVectorStore(uri="http://milvus-standalone:19530", token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        else:
            vector_store = MilvusVectorStore(uri="http://milvus-standalone:19530", token="", collection_name=collection_name, embedding_field='embeddings', text_key='label', dim=embedder_dimension, overwrite=False)
        index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store
        )
        query_engine = index.as_query_engine(similarity_top_k=top_k, text_qa_template=text_qa_template)
    except Exception as e:
        cols = {"Response": [f"ERROR: Could not load collection. ERROR: {e}"], "References": ["ERROR"]}
        result = pd.DataFrame(data=cols)
        return result
        
    l = []
    f = []
    try:
        for i in range(len(X)):
            r = query_engine.query(X[i])
            l.append(r.response)
            if d_type == "Documents":
                files = ""
                for node in r.source_nodes:
                    files += node.node.metadata['file_path']
                    files += "\n"
                    files += node.text
                    files += "\n"
                f.append(files)
            else:
                logs = ""
                for i in range(len(r.source_nodes)):
                    logs += r.source_nodes[0].text
                    logs += "\n"
                f.append(logs)  
    except Exception as e:
        cols = {"Response": [f"Failed at querying. ERROR: {e}. Please check if the knowledge type is correct"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    
    cols = {"Response": l, "References": f}
    result = pd.DataFrame(data=cols)
    return result







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
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





    
# In[27]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    X = df["query"].values.tolist()
    # "Documents" or "Logs"
    try:
        d_type = param['options']['params']['rag_type'].strip('\"')
    except:
        d_type = "Documents"
        print("Using Document knowledge type by default")

    try:
        use_local= int(param['options']['params']['use_local'])
    except:
        use_local=0
        print("Download embedding model by default")
        
    try:
        embedder_name = param['options']['params']['embedder_name'].strip('\"')
    except:
        embedder_name = 'all-MiniLM-L6-v2'
        print("Using all-MiniLM-L6-v2 by default")

    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        cols = {"Response": ["ERROR: no collection specified. Please specify a vectorDB collection"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
        

    if embedder_name == 'intfloat/multilingual-e5-large':
        embedder_dimension = 1024
    elif embedder_name == 'all-MiniLM-L6-v2':
        embedder_dimension = 384
    else:
        try:
            embedder_dimension = int(param['options']['params']['embedder_dimension'])
        except:
            embedder_dimension = 384
    if use_local:
        embedder_name = f'/srv/app/model/data/{embedder_name}'
        print("Using local embedding model checkpoints")
    
    try:
        top_k = int(param['options']['params']['top_k'])
    except:
        top_k = 5
        print("Using top 5 results by default")
        
    if d_type == "Documents":
        qa_prompt_str = (
            "Below are the context information.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information as well as necessary prior knowledge, "
            "answer the question: {query_str}\n"
        )
        chat_text_qa_msgs = [
            (
                "system",
                "You are an expert Q&A system that is trusted around the world. Always answer the query using the provided context information and reasoning as detailed as possible",
            ),
            ("user", qa_prompt_str),
        ]
    else:
        qa_prompt_str = (
            "Past log messages below are given as context information.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information as well as necessary prior knowledge, "
            "answer the question: {query_str}\n"
        )
        chat_text_qa_msgs = [
            (
                "system",
                "You are an expert Q&A system that is trusted around the world. Always answer the query using the provided context information and reasoning as detailed as possible",
            ),
            ("user", qa_prompt_str),
        ]
    
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    try:
        model = param['options']['params']['model_name'].strip('\"')
    except:
        cols = {"Response": ["ERROR: no LLM model name specified. Please choose a LLM model."], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        url = LLM_ENDPOINT
        llm = Ollama(model=model, base_url=url, request_timeout=6000.0)
    except Exception as e:
        cols = {"Response": [f"Could not load LLM. ERROR: {e}"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        Settings.llm = llm
        Settings.embed_model = transformer_embedder
        Settings.chunk_size = 1024
    except Exception as e:
        cols = {"Response": [f"Could not load embedder. ERROR: {e}"], "References": ["ERROR"]}
        result = pd.DataFrame(data=cols)
        return result
    try:
        if d_type == "Documents":
            vector_store = MilvusVectorStore(uri="http://milvus-standalone:19530", token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
        else:
            vector_store = MilvusVectorStore(uri="http://milvus-standalone:19530", token="", collection_name=collection_name, embedding_field='embeddings', text_key='label', dim=embedder_dimension, overwrite=False)
        index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store
        )
        query_engine = index.as_query_engine(similarity_top_k=top_k, text_qa_template=text_qa_template)
    except Exception as e:
        cols = {"Response": [f"ERROR: Could not load collection. ERROR: {e}"], "References": ["ERROR"]}
        result = pd.DataFrame(data=cols)
        return result
        
    l = []
    f = []
    try:
        for i in range(len(X)):
            r = query_engine.query(X[i])
            l.append(r.response)
            if d_type == "Documents":
                files = ""
                for node in r.source_nodes:
                    files += node.node.metadata['file_path']
                    files += "\n"
                    files += node.text
                    files += "\n"
                f.append(files)
            else:
                logs = ""
                for i in range(len(r.source_nodes)):
                    logs += r.source_nodes[0].text
                    logs += "\n"
                f.append(logs)  
    except Exception as e:
        cols = {"Response": [f"Failed at querying. ERROR: {e}. Please check if the knowledge type is correct"], "References": ["None"]}
        result = pd.DataFrame(data=cols)
        return result
    
    cols = {"Response": l, "References": f}
    result = pd.DataFrame(data=cols)
    return result

















