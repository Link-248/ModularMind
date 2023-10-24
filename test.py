from framework.agents.TreeOfThought.Simple_MCTSAgent import MonteCarloAgent
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

print(f"Solution: {solution}")


#Algorithm of Thought test
'''from framework.agents.AlgorithmOfThought.AoTAgent import AoTAgent

task = """


PROMPT
###################
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. When
considering the next steps, do not choose operations that will result in a
negative or fractional number. In order to help with the calculations, the
numbers in the parenthesis represent the numbers that are left after the
operations and they are in descending order.
Another thing we do is when there are only two numbers left in the parenthesis, we
check whether we can arrive at 24 only by using basic arithmetic operations
(+ - * /). Some examples regarding this idea:
(21 2) no
since 21 + 2 = 23, 21 - 2 = 19, 21 * 2 = 42, 21 / 2 = 10.5, none of which is equal
to 24.
(30 6) 30 - 6 = 24 yes
(8 3) 8 * 3 = 24 yes
(12 8) no
(48 2) 48 / 2 = 24 yes
Most importantly, do not give up, all the numbers that will be given has indeed a
solution.

OBJECTIVE:
5 10 5 2
###################
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
OPEN_AI_BASE = "https://api.naga.ac/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # 


#openai.api_key = OPENAI_API_KEY
#openai.api_base = OPEN_AI_BASE

dfs = AoTAgent(
    num_thoughts=2,
    max_steps=5,
    value_threshold=0.7,
    initial_prompt=task,
    api_base=OPEN_AI_BASE,
    api_key=CHIMERA_GPT_KEY,
    valid_retry_count=3,
)

result = dfs.solve()
print(result)'''

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
