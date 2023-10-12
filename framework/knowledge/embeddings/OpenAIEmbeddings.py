from time import sleep
import openai
import pandas as pd
import chardet
from abc import ABC, abstractmethod

class OpenAIEmbeddings(ABC):
    """
    OpenAI Embeddings model, convert multiple data set types to multiple embedding data set types
    """
    model: str
    base_url: str
    base_api_key: str
    useOpenAIBase: bool = True
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    
    @abstractmethod
    def __init__(self, base_api_key, model: str = "text-embedding-ada-002", base_url :str = 'https://api.openai.com/v1', useOpenAIBase: bool = True):
        self.base_url = base_url
        self.base_api_key = base_api_key
        self.useOpenAIBase = useOpenAIBase
        self.model = model
        
        if not useOpenAIBase:
            openai.api_base = base_url
            openai.api_key =  base_api_key
        else:
            openai.api_key = base_api_key
    
    def get_embedding(self, text, model: str ='text-embedding-ada-002'):
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            response = openai.Embedding.create(input = text, model=model)
        except:
            done = False
            count = 0
            while not done or count == 5 :
                sleep(5)
                count += 1
                try:
                    response = openai.Embedding.create(input = text, model=model)
                    done = True
                except:
                    pass
        
        return response['data'][0]['embedding']
    
class TextEmbeddings(OpenAIEmbeddings):
    
    def __init__(self, base_api_key, model: str = "text-embedding-ada-002", base_url :str = 'https://api.openai.com/v1', useOpenAIBase: bool = True):
        super().__init__(base_api_key, model, base_url, useOpenAIBase)
    
    # Convert a [topic, content] CSV file to a [topic, content, embeddings] CSV file 
    def preset_csv_to_embeds_csv(self, data_path: str, output_path: str, model: str ='text-embedding-ada-002'):
        # load & inspect dataset
        input_datapath = data_path  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath)
        df.drop(columns=df.columns[0], axis=1, inplace=True)

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df["embeddings"] = df['content'].apply(lambda x: list(self.get_embedding(x, model=model)))
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_path, index=False)
    
    # Convert a {topic:content} dict to a [topic, content, embeddings] CSV file
    def dict_to_embeds_csv(self, data_dict: dict, output_path: str, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['content'])

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings['embeddings'] = df['content'].apply(lambda x: list(self.get_embedding(x, model=model)))
        
        # Save the DataFrame to a CSV file
        df_embeddings.to_csv(output_path, index=False)
    
    # Convert a [topic, content] CSV file to a {content:embeddings} dict
    def preset_csv_to_embeds_dict(self, data_path: str, model: str ='text-embedding-ada-002'):
        # load & inspect dataset
        input_datapath = data_path  # to save space, we provide a pre-filtered dataset
        df = pd.read_csv(input_datapath)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        content = df['content'].tolist()
        
        #Save the DataFrame to a dict of NL topic and embedding key value pairs
        embeds = dict()
        for i in range(df_embeddings.index.size):
            embeds[content[i]] = (list(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
        return embeds
    
    # Convert a {topic:content} dict to a {content:embeddings} dict
    def dict_to_embeds_dict(self, data_dict: dict, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        print(df)
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        content = df['content'].tolist()
        
        #Save the DataFrame to a dict of NL topic and embedding key value pairs
        embeds = dict()
        for i in range(df_embeddings.index.size):
            embeds[content[i]] = (list(str(df_embeddings.loc[i,'content']).strip('[').strip(']').split(', ')))
            print(content[i],embeds[content[i]])
        return embeds
    
    # Convert an unfiltered CSV file to a CSV file with a new embeddings cell for each contnt cell
    def unfiltered_csv_to_embeds_csv(self, data_path: str, output_path: str, model: str ='text-embedding-ada-002'):
        # Read the file with the detected encoding
        try:
            # Try common file encodings
            encoding = self.try_encodings(data_path)
            df = pd.read_csv(data_path, encoding=encoding,  header=None)
        except:
            # Detect the file's encoding
            with open(data_path, 'rb') as f:
                result = chardet.detect(f.read())
            df = pd.read_csv(data_path, encoding=result['encoding'], header=None)

        # Create a new DataFrame to store the embeddings
        df_embeddings = pd.DataFrame()

        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Handle NaN values
            df[column] = df[column].fillna('')
            # Add the original column to the new DataFrame
            df_embeddings[str(column)] = df[column]
            # Add the embeddings column to the new DataFrame
            df_embeddings['embeddings_' + str(column)] = df[column].apply(lambda x: [self.get_embedding(str(x), model=model)])

        # Save the DataFrame to a CSV file
        df_embeddings.to_csv(output_path, index=False)

    # Detect the encoding of a CSV file
    def try_encodings(data_path):
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']  # Add more encodings if necessary
        for enc in encodings:
            try:
                df = pd.read_csv(data_path, encoding=enc)
                return df
            except UnicodeDecodeError:
                pass
        raise ValueError('None of the tried encodings work')


'''from dotenv import load_dotenv
load_dotenv()
import os
from DocumentParser import PDFParser
    
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPEN_AI_BASE = 'https://api.naga.ac/v1' #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

bms = PDFParser.breakdown_document("RAP.pdf", 
                                max_tokens=4000, only_alphaNumeric=False, 
                                strip_bookmarks={'Reasoning via Planning (RAP)'})
                                 
embeddings = TextEmbeddings(base_api_key=OPENAI_API_KEY, useOpenAIBase=True)

input_datapath = "RAP.csv"  
embeddings.preset_csv_to_embeds_csv(data_path=input_datapath, output_path='output.csv', model='text-embedding-ada-002')
#embeddings.dict_to_embeds_dict(data_dict=bms, model='text-embedding-ada-002')
print(embeddings.get_embedding("Hello World", model='text-embedding-ada-002'))'''