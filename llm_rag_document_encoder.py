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
from llama_index.readers.file import DocxReader, CSVReader, PDFReader, PptxReader, XMLReader, IPYNBReader 
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import textwrap
from app.model.llm_utils import create_llm, create_embedding_model
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
MILVUS_ENDPOINT = "http://milvus-standalone:19530"









    
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







    
# In[22]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    try:
        data_path = param['options']['params']['data_path'].strip('\"')
    except:
        data_path = None
        print("No file path specified.")

    try:
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")
        
    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = 384
        print("embedder_dimension not specified. Use 384 by default.")
        
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name = "default-doc-collection"
        print("collection_name not specified. Use default-doc-collection by default.")
        
    if service == "huggingface" or service == "ollama":
        try:
            embedder_name = param['options']['params']['embedder_name'].strip('\"')
        except:
            embedder_name = 'all-MiniLM-L6-v2'
            print("embedder_name not specified. Use all-MiniLM-L6-v2 by default.")

        try:
            use_local = int(param['options']['params']['use_local'])
        except:
            use_local = 0
            print("download Huggingface embedder model by default.")

        # Dimension checking for default embedders
        if embedder_name == 'intfloat/multilingual-e5-large':
            embedder_dimension = 1024
        elif embedder_name == 'all-MiniLM-L6-v2':
            embedder_dimension = 384
        else:
            embedder_dimension = int(embedder_dimension)
         
        # Using local embedder checkpoints
        if use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints")
        
    # To support pptx files, huggingface extractor needs to be downloaded. Skipping support for this version
    # Special parser for CSV data
    parser = CSVReader()
    file_extractor = {".csv": parser}
    try:
        # Create document dataloader - recursively find data from sub-directories
        # Add desired file extensions in required_exts. For example: required_exts=[".csv", ".xml", ".pdf", ".docx", ".ipynb"]
        documents = SimpleDirectoryReader(
            input_dir=data_path, recursive=True, file_extractor=file_extractor, required_exts=[".ipynb", ".csv", ".xml", ".pdf", ".txt", ".docx"]
        ).load_data()
    except Exception as e:
        documents = None
        message = f"ERROR in directory loading: {e} "
    # Create Transformers embedding model 
    ## TODO: add local loading option
    try:
        if service == "huggingface" or service == "ollama":
            embedder, m = create_embedding_model(service=service, model=embedder_name)
        else:
            embedder, m = create_embedding_model(service=service)
        # transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        if embedder is not None:
            print(m)
        else:
            message = f"ERROR in embedding model loading: {m}. "
    except Exception as e:
        embedder = None
        message = f"ERROR in embedding model loading: {e}. Check if the model name is correct. If you selected Yes for use local embedder, make sure you have pulled the embedding model to local."

    if (documents is not None) & (embedder is not None):
        try:
            # Replacing service context in legacy llama-index
            Settings.llm = None
            Settings.embed_model = embedder
            Settings.chunk_size = 1024
            vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Index document data
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            # Prepare output dataframe
            embedder_info = [m]
            vector_store = [str(vector_store)]
            document = []
            for d in documents:
                document.append(str(d.metadata['file_path']))
            document = str(list(dict.fromkeys(document)))
            cols = {"Embedder_Info": embedder_info, "Vector_Store_Info": vector_store, "Documents_Info": [document], "Message": ["Success"]}
        except Exception as e:
            message = f"ERROR in encoding: {e}."
            cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
    else:
        cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
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





    
# In[25]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns
def compute(model,df,param):
    # Datapath Example: '/srv/notebooks/data/splunk_doc/'
    try:
        data_path = param['options']['params']['data_path'].strip('\"')
    except:
        data_path = None
        print("No file path specified.")

    try:
        service = param['options']['params']['embedder_service'].strip('\"')
        print(f"Using {service} embedding service")
    except:
        service = "huggingface"
        print("Using default Huggingface embedding service")
        
    try:
        embedder_dimension = int(param['options']['params']['embedder_dimension'])
    except:
        embedder_dimension = 384
        print("embedder_dimension not specified. Use 384 by default.")
        
    try:
        collection_name = param['options']['params']['collection_name'].strip('\"')
    except:
        collection_name = "default-doc-collection"
        print("collection_name not specified. Use default-doc-collection by default.")
        
    if service == "huggingface" or service == "ollama":
        try:
            embedder_name = param['options']['params']['embedder_name'].strip('\"')
        except:
            embedder_name = 'all-MiniLM-L6-v2'
            print("embedder_name not specified. Use all-MiniLM-L6-v2 by default.")

        try:
            use_local = int(param['options']['params']['use_local'])
        except:
            use_local = 0
            print("download Huggingface embedder model by default.")

        # Dimension checking for default embedders
        if embedder_name == 'intfloat/multilingual-e5-large':
            embedder_dimension = 1024
        elif embedder_name == 'all-MiniLM-L6-v2':
            embedder_dimension = 384
        else:
            embedder_dimension = int(embedder_dimension)
         
        # Using local embedder checkpoints
        if use_local:
            embedder_name = f'/srv/app/model/data/{embedder_name}'
            print("Using local embedding model checkpoints")
        
    # To support pptx files, huggingface extractor needs to be downloaded. Skipping support for this version
    # Special parser for CSV data
    parser = CSVReader()
    file_extractor = {".csv": parser}
    try:
        # Create document dataloader - recursively find data from sub-directories
        # Add desired file extensions in required_exts. For example: required_exts=[".csv", ".xml", ".pdf", ".docx", ".ipynb"]
        documents = SimpleDirectoryReader(
            input_dir=data_path, recursive=True, file_extractor=file_extractor, required_exts=[".ipynb", ".csv", ".xml", ".pdf", ".txt", ".docx"]
        ).load_data()
    except Exception as e:
        documents = None
        message = f"ERROR in directory loading: {e} "
    # Create Transformers embedding model 
    ## TODO: add local loading option
    try:
        if service == "huggingface" or service == "ollama":
            embedder, m = create_embedding_model(service=service, model=embedder_name)
        else:
            embedder, m = create_embedding_model(service=service)
        # transformer_embedder = HuggingFaceEmbedding(model_name=embedder_name)
        if embedder is not None:
            print(m)
        else:
            message = f"ERROR in embedding model loading: {m}. "
    except Exception as e:
        embedder = None
        message = f"ERROR in embedding model loading: {e}. Check if the model name is correct. If you selected Yes for use local embedder, make sure you have pulled the embedding model to local."

    if (documents is not None) & (embedder is not None):
        try:
            # Replacing service context in legacy llama-index
            Settings.llm = None
            Settings.embed_model = embedder
            Settings.chunk_size = 1024
            vector_store = MilvusVectorStore(uri=MILVUS_ENDPOINT, token="", collection_name=collection_name, dim=embedder_dimension, overwrite=False)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Index document data
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            # Prepare output dataframe
            embedder_info = [m]
            vector_store = [str(vector_store)]
            document = []
            for d in documents:
                document.append(str(d.metadata['file_path']))
            document = str(list(dict.fromkeys(document)))
            cols = {"Embedder_Info": embedder_info, "Vector_Store_Info": vector_store, "Documents_Info": [document], "Message": ["Success"]}
        except Exception as e:
            message = f"ERROR in encoding: {e}."
            cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
    else:
        cols = {"Embedder_Info": ["No Result"], "Vector_Store_Info": ["No Result"], "Documents_Info": ["No Result"], "Message": [message]}
    result = pd.DataFrame(data=cols)
    return result







