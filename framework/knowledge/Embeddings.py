from typing import Dict, Union
import openai
from abc import ABC, abstractmethod
import tiktoken
import pandas as pd
import json

class Embeddings():
    """
    OpenAI Embeddings model, convert multiple data set types to multiple embedding data set types
    """
    model: str
    base_url: str
    base_api_key: str
    useOpenAIBase: bool = True
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    
    def __init__(self, base_api_key, model: str = "text-embedding-ada-002", base_url :str = 'https://api.openai.com/v1', useOpenAIBase: bool = True):
        self.base_url = base_url
        self.base_api_key = base_api_key
        self.useOpenAIBase = useOpenAIBase
        self.model = model
        
        if not useOpenAIBase:
            openai.api_base = base_url
            openai.api_key =  base_api_key
        
    def get_embedding(self, text, model: str ='text-embedding-ada-002'):
        response = openai.Embedding.create(input = text, model=model)
        return response['data'][0]['embedding']
    
    def csv_to_embeds_list(self, data_path: str, model: str ='text-embedding-ada-002'):
        # load & inspect dataset
        input_datapath = data_path  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath, header=None)
        df.columns = ['topic', 'content']
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        
        #Save the DataFrame to a list of embeddingss
        embeds = list()
        for i in range(df_embeddings.index.size):
            embeds.append(set(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
        
        return embeds
    
    def csv_to_embeds_dict(self, data_path: str, model: str ='text-embedding-ada-002'):
        # load & inspect dataset
        input_datapath = data_path  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath, header=None)
        df.columns = ['topic', 'content']
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        topics = df_embeddings['topic'].tolist()
        
        #Save the DataFrame to a dict of NL topic and embedding key value pairs
        embeds = dict()
        for i in range(df_embeddings.index.size):
            embeds[topics[i]] = (set(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
        
        return embeds
    
    def csv_to_embeds_csv(self, data_path: str, model: str ='text-embedding-ada-002'):
        # load & inspect dataset
        input_datapath = data_path  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath, header=None)
        df.columns = ['topic', 'content']
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        
        # Save the DataFrame to a CSV file
        df_embeddings.to_csv('embedded_RAP.csv', index=False)
    
    def dict_to_embeds_csv(self, data_dict: dict, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        
        # Save the DataFrame to a CSV file
        df_embeddings.to_csv('embedded_RAP2.csv', index=False)
        
    def dict_to_embeds_list(self, data_dict: dict, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        
        #Save the DataFrame to a list of embeddingss
        embeds = list()
        for i in range(df_embeddings.index.size):
            embeds.append(set(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
            print(embeds[i])
        return embeds
    
    def dict_to_embeds_dict(self, data_dict: dict, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        topics = df_embeddings['topic'].tolist()
        
        #Save the DataFrame to a dict of NL topic and embedding key value pairs
        embeds = dict()
        for i in range(df_embeddings.index.size):
            embeds[topics[i]] = (set(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
        return embeds


'''from dotenv import load_dotenv
load_dotenv()
import os
from DocumentParser import PDFParser
    
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
OPEN_AI_BASE = 'https://api.naga.ac/v1' #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

bms = PDFParser.breakdown_document("RAP.pdf", 
                                    max_tokens=4000, only_alphaNumeric=False, 
                                    strip_bookmarks={'Reasoning via Planning (RAP)'})
                                    
embeddings = Embeddings(base_api_key=OPENAI_KEY,  useOpenAIBase=False)

input_datapath = "RAP.csv"  
embeddings.dict_to_embeds_list(bms, model='text-embedding-ada-002')'''