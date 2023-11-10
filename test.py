'''from framework.agents.TreeOfThought.Simple_MCTSAgent import MonteCarloAgent
import os
from dotenv import load_dotenv

api_model= "gpt-3.5-turbo"
load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
  
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
HYPRLAB_API_KEY = os.getenv('HYPRLAB_API_KEY')
OPEN_AI_BASE =  'https://api.nova-oss.com/v1' #"https://api.hyprlab.io/v1" #"https://api.naga.ac/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #"https://thirdparty.webraft.in/v1" # 




# Initialize the MonteCarloTreeofThoughts class with the model
tree_of_thoughts = MonteCarloAgent(enable_ReAct_prompting=False, optimized=True, api_key=NOVA_API_KEY, api_base=OPEN_AI_BASE)

# Note to reproduce the same results from the tree of thoughts paper if not better, 
# craft an 1 shot chain of thought prompt for your task below

initial_prompt =  """Determine whether the relation R on the set of all real
numbers is reflexive, symmetric, antisymmetric, and/or
transitive, where (x, y) belong to R if and only if x=y(mod 7)"""
num_thoughts = 1
max_steps = 3
max_states = 4
pruning_threshold = 0.9




solution = tree_of_thoughts.solve(
    initial_prompt=initial_prompt,
    num_thoughts=num_thoughts, 
    max_steps=max_steps, 
    max_states=max_states, 
    pruning_threshold=pruning_threshold,
    # sleep_time=sleep_time
)

print(f"Solution: {solution}")'''


#Algorithm of Thought test
from framework.agents.AlgorithmOfThought.AoTAgent import AoTAgent
#import openai

task =''' It takes 3 hours to dry 3 shirts, How long would it take to dry 9 shirts?'''

'''If (A-B) = [1,5,7,8], (B-A) = [2,10], and (A∩B) = [3,6,9], Find the set B.'''
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



#openai.api_key = OPENAI_API_KEY
#openai.api_base = OPEN_AI_BASE

dfs = AoTAgent(
    model="gpt-4-1106-preview",
    num_thoughts=2,
    max_steps=3,
    pruning_threshold=50,
    value_threshold=80,
    initial_prompt=task,
    api_base=OPEN_AI_BASE,
    api_key=HYPRLAB_API_KEY,
    valid_retry_count=1,
)
#openai.api_base = OPEN_AI_BASE
#openai.api_key = HYPRLAB_API_KEY


result = dfs.solve()
print(result)

#Vector store + embeddings test 
'''from framework.blocks.knowledge.vectorStores.Pinecone import Pinecone
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

#openai.api_key = OPENAI_API_KEY
#openai.api_base = OPEN_AI_BASE
embedding = TextEmbeddings(base_api_key=OPENAI_API_KEY)

input_datapath = "data\RAP.csv"  
embed_dict = embedding.preset_csv_to_embeds_dict(data_path=input_datapath, model='text-embedding-ada-002')

vector_store = Pinecone(api_key=PINECONE_API_KEY, index_name='vector-store-test')

#vector_store.upsert_embeddings_from_dict(dict=embed_dict, metadata_name='content')

query = "What is monte carlos tree search?"

xq = embedding.get_embedding(query)

res = vector_store.query([xq], top_k=5, include_metadata=True)
print(vector_store.get_top_k_responses(metadata_to_get='content', res=res, top_k=2))

#print(f"{res['matches'][0]['score']:.2f}: {res['matches'][0]['metadata']['content']}")'''
