
#Vector store + embeddings test 
from framework.knowledge.vectorStores.Pinecone import Pinecone
from framework.knowledge.embeddings.OpenAIEmbeddings import TextEmbeddings
import os
from dotenv import load_dotenv
import openai

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

embedding = TextEmbeddings(base_api_key=OPENAI_API_KEY, useOpenAIBase=True)

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