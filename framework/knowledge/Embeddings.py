import openai
import pandas as pd
import chardet

class TextEmbeddings():
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
        
    def two_header_csv_to_embeds_list(self, data_path: str, model: str ='text-embedding-ada-002'):
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
    
    def two_header_csv_to_embeds_dict(self, data_path: str, model: str ='text-embedding-ada-002'):
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
    
    def two_header_csv_to_embeds_csv(self, data_path: str, output_path: str, model: str ='text-embedding-ada-002'):
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
        df_embeddings.to_csv(output_path, index=False)
    
    def dict_to_embeds_csv(self, data_dict: dict, output_path: str, model: str ='text-embedding-ada-002'):
        df = pd.DataFrame(list(data_dict.items()), columns=['topic', 'content'])
        
        # Create a new DataFrame with 'topic' as the first column
        df_embeddings = pd.DataFrame(df['topic'])

        # Embed the topics
        #df_embeddings['topic_embedding'] = df['topic'].apply(lambda x: str(embeddings.get_embedding(x)))

        # Apply the lambda function to get the embeddings and add them to the DataFrame
        df_embeddings = df_embeddings.join(df['content'].apply(lambda x: list(self.get_embedding(x, model=model))))
        
        # Save the DataFrame to a CSV file
        df_embeddings.to_csv(output_path, index=False)
        
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
    
    def csv_to_embeds_csv(self, data_path: str, output_path: str, model: str ='text-embedding-ada-002'):
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
        df_embeddings = df.copy()

        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Handle NaN values
            df[column] = df[column].fillna('')
            # Embed the column's values
            df_embeddings[column] = df[column].apply(lambda x: [self.get_embedding(str(x), model=model)])

        # Save the DataFrame to a CSV file
        df_embeddings.to_csv(output_path, index=False)

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
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
OPEN_AI_BASE = 'https://api.naga.ac/v1' #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

bms = PDFParser.breakdown_document("RAP.pdf", 
                                    max_tokens=4000, only_alphaNumeric=False, 
                                    strip_bookmarks={'Reasoning via Planning (RAP)'})
                                 
embeddings = TextEmbeddings(base_api_key=OPENAI_KEY,  useOpenAIBase=False)

input_datapath = "addresses.csv"  
embeddings.csv_to_embeds_csv(data_path=input_datapath, output_path="OUTPUT.csv", model='text-embedding-ada-002')'''