from llama_index.llms.ollama import Ollama
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.gemini import Gemini

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

import json
import os


def create_llm(service='ollama', model=None):
    config = json.loads(os.environ['llm_config'])
    service_list = ['ollama','azure_openai','openai','bedrock','gemini']
    if service in service_list:
        print(f"Initializing LLM object from {service}")
        if not config['llm'][service]['is_configured']:
            err = f"Service {service} is not configured. Please configure the service."
            return None, err
    else:
        err = f"Service {service} is not supported. Please choose from {str(service_list)}."
        return None, err
    
    llm_config = dict(config['llm'][service])
    del llm_config["is_configured"]
    if service == 'ollama':
        if model:
            llm_config['model'] = model
        try:
            llm = Ollama(**llm_config, request_timeout=6000.0)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'azure_openai':
        try:
            llm = AzureOpenAI(**llm_config)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'openai':
        try:
            llm = OpenAI(**llm_config)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'bedrock':
        try:
            llm = Bedrock(**llm_config)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    else:
        try:
            llm = Gemini(**llm_config)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err
            
    message = f"Successfully created LLM object from {service}"
    return llm, message
    
def create_embedding_model(service='huggingface', model=None):
    config = json.loads(os.environ['llm_config'])
    service_list = ['huggingface','ollama','azure_openai','openai','bedrock','gemini']
    if service in service_list:
        print(f"Initializing Embedding model object from {service}")
        if not config['embedding_model'][service]['is_configured']:
            err = f"Service {service} is not configured. Please configure the service."
            return None, err
    else:
        err = f"Service {service} is not supported. Please choose from {str(service_list)}."
        return None, err
    
    embedding_model_config = dict(config['embedding_model'][service])
    del embedding_model_config["is_configured"]
    if service == 'huggingface':
        if model:
            embedding_model_config['model_name'] = model
        try:
            embedding_model = HuggingFaceEmbedding(**embedding_model_config)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err    
            
    elif service == 'ollama':
        if model:
            embedding_model_config['model_name'] = model
        try:
            embedding_model = OllamaEmbedding(**embedding_model_config, request_timeout=6000.0)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err

    elif service == 'azure_openai':
        try:
            embedding_model = AzureOpenAIEmbedding(**embedding_model_config)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err

    elif service == 'openai':
        try:
            embedding_model = OpenAI(**embedding_model_config)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err

    elif service == 'bedrock':
        try:
            embedding_model = BedrockEmbedding.from_credentials(**embedding_model_config)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err

    else:
        try:
            embedding_model = GeminiEmbedding(**embedding_model_config)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, err
            
    message = f"Successfully created Embedding model object from {service}"
    return embedding_model, message
    
