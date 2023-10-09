from abc import ABC, abstractmethod
import openai
from termcolor import colored
import json

class AgentBase(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
class ChatAgent(AgentBase):
    """
    The ChatAgent module is the sum of multiple blocks that build up an chat agent.
    """
    main_prompt: str = ""
    model: str
    base_url: str
    base_api_key: str
    useOpenAI: bool = True
    temperature: int = 0
    stream: bool = True
    
    def __init__(self, base_api_key, main_prompt: str = "", model: str = "gpt-3.5-turbo", base_url :str = 'https://api.openai.com/v1', useOpenAI: bool = True, temperature: int = 0, stream: bool = True):
        self.main_prompt=main_prompt
        self.base_url = base_url
        self.base_api_key = base_api_key
        self.useOpenAI = useOpenAI
        self.model = model
        self.temperature = temperature
        self.stream = stream
        
        if(useOpenAI):
            openai.api_base = base_url
            openai.api_key =  base_api_key
        
    
    def run(self, query: str):
        memory = ([
        { "role": "system", "content": self.main_prompt if self.main_prompt != "" else "You are an autonomous Agent."},
        { "role": "user", "content": query },
        ])
        
        #total_session_tokens = sum([len(chatEncoding.encode(message["content"])) for message in memory])
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=memory,
            temperature=0,
            stream=self.stream,) 
        
        if(self.stream):
            tokens_used = 0
            responses = ''
            
            #process each chunk of the response
            for chunk in response:
                if "role" in chunk["choices"][0]["delta"]:
                    continue
                elif "content" in chunk["choices"][0]["delta"]:
                    tokens_used += 1
                    
                    r_text = chunk["choices"][0]["delta"]["content"]
                    responses += r_text
                    print(colored(r_text, "green"), end='', flush=True)
                
            #total_session_tokens += tokens_used
            
            '''if show_token_consumption:
                print(colored("\nTokens used this time: " + str(tokens_used), "red"))
                print(colored("\nTokens used so far: " + str(total_session_tokens), "yellow"))'''
        else:
            return response["choices"][0]["message"]["content"]
    
    
from dotenv import load_dotenv
load_dotenv()
import os
    
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
OPEN_AI_BASE = 'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

llm = ChatAgent(base_url=OPEN_AI_BASE, base_api_key=NOVA_API_KEY, model="gpt-4", useOpenAI=True, stream=False)
#llm.run("What is the weather in New York?")
print(llm.run("What can you tell me about set theory?"))