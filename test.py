
#Vector store + embeddings test 
from framework.blocks.knowledge.vectorStores.Pinecone import Pinecone
from framework.blocks.knowledge.embeddings.OpenAIEmbeddings import TextEmbeddings
import os
from dotenv import load_dotenv
import openai

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
  
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
OPEN_AI_BASE = 'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

openai.api_key = OPENAI_API_KEY
openai.api_base = OPEN_AI_BASE
embedding = TextEmbeddings(base_api_key=OPENAI_API_KEY, base_url=OPEN_AI_BASE, useOpenAIBase=False)

input_datapath = "RAP.csv"  
embed_dict = embedding.preset_csv_to_embeds_dict(data_path=input_datapath, model='text-embedding-ada-002')

vector_store = Pinecone(api_key=PINECONE_API_KEY, index_name='vector-store-test', environment='gcp-starter')

#vector_store.upsert_embeddings_from_dict(dict=embed_dict, metadata_name='content')

query = "What is monte carlos tree search?"

xq = embedding.get_embedding(query)

res = vector_store.query([xq], top_k=5, include_metadata=True)

#print(f"{res['matches'][0]['score']:.2f}: {res['matches'][0]['metadata']['content']}")
    
response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=([
            { "role": "system", "content": f"Here is Context: {res['matches'][0]['score']:.2f}: {res['matches'][0]['metadata']['content']}"},
            { "role": "user", "content": query },
            ]),
            temperature=0,
            stream=False) 

print(response["choices"][0]["message"]["content"])