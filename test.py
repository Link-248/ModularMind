

#Algorithm of Thought test
from framework.agents.AlgorithmOfThought.AoTAgent import AoTAgent
#import openai




"""
Laird: Pure research provides us with new technologies that contribute to saving lives. Even more worthwhile than this, however, is its role in expanding our knowledge and providing new, unexplored ideas.

Kim: Your priorities are mistaken. Saving lives is what counts most of all. Without pure research, medicine would not be as advanced as it is.

Laird and Kim disagree on whether pure research

A) derives its significance in part from its providing new technologies 
B) expands the boundaries of our knowledge of medicine
C) should have the saving of human lives as an important goal
D) has its most valuable achievements in medical applications
E) has any value apart from its role in providing new technologies to save lives

chose one of the options.
"""



import os
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
HYPRLAB_API_KEY = os.getenv('HYPRLAB_API_KEY')
OPEN_AI_BASE =  "https://api.hyprlab.io/v1" #'https://api.nova-oss.com/v1' #"https://api.naga.ac/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #"https://thirdparty.webraft.in/v1" # 
NOVA_BASE = 'https://api.nova-oss.com/v1'
WEBDRAFT_BASE = 'https://thirdparty.webraft.in/v1'
"""
Laird: Pure research provides us with new technologies that contribute to saving lives. Even more worthwhile than this, however, is its role in expanding our knowledge and providing new, unexplored ideas.

Kim: Your priorities are mistaken. Saving lives is what counts most of all. Without pure research, medicine would not be as advanced as it is.

Laird and Kim disagree on whether pure research

A) derives its significance in part from its providing new technologies 
B) expands the boundaries of our knowledge of medicine
C) should have the saving of human lives as an important goal
D) has its most valuable achievements in medical applications
E) has any value apart from its role in providing new technologies to save lives

chose one of the options.
"""



task = '''If (A-B) = [1,5,7,8], (B-A) = [2,10], and (A∩B) = [3,6,9], Find the set B.'''


dfs = AoTAgent(
    model="gpt-4-32k",
    num_thoughts=2,
    max_steps=3,
    pruning_threshold=50,
    value_threshold=80,
    initial_prompt=task,
    api_base=OPEN_AI_BASE,
    api_key=HYPRLAB_API_KEY,
    valid_retry_count=1,
)



result = dfs.solve()
print(result)
'''
#Vector store + embeddings test 
from framework.blocks.knowledge.vectorStores.Pinecone import Pinecone
from framework.blocks.knowledge.embeddings.OpenAIEmbeddings import TextEmbeddings
import os
from dotenv import load_dotenv

embedding = TextEmbeddings(base_api_key=HYPRLAB_API_KEY, base_url=OPEN_AI_BASE, useOpenAIBase=False)

input_datapath = "data\RAP.csv"  
embed_dict = embedding.preset_csv_to_embeds_dict(data_path=input_datapath, model='text-embedding-ada-002')

vector_store = Pinecone(api_key=PINECONE_API_KEY, index_name='vector-store-test')

vector_store.upsert_embeddings_from_dict(dict=embed_dict, metadata_name='content')

query = "What is monte carlos tree search?"

xq = embedding.get_embedding(query)

res = vector_store.query([xq], top_k=5, include_metadata=True)
print(vector_store.get_top_k_responses(metadata_to_get='content', res=res, top_k=2))

#print(f"{res['matches'][0]['score']:.2f}: {res['matches'][0]['metadata']['content']}")'''
