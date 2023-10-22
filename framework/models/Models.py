from abc import ABC, abstractmethod
import time
import openai
from termcolor import colored
import tiktoken
import os

class ModelBase(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
class OpenAI(ModelBase):
    """
    The OpenAI model for usage in an Agent.
    """
    model: str
    stream: bool
    chatEncoding: object
    strategy: str
    evaluation_strategy: str
        
    def __init__(self, 
                 base_api_key: str = "", 
                 chatEncoding = tiktoken.get_encoding("cl100k_base"), 
                 model: str = "gpt-3.5-turbo", 
                 base_url :str = 'https://api.openai.com/v1', 
                 useOpenAIBase: bool = True, 
                 stream: bool = True,
                 strategy="cot",
                 evaluation_strategy="value",):
        
        self.useOpenAIBase = useOpenAIBase
        self.model = model
        self.stream = stream
        self.chatEncoding = chatEncoding
        
        if base_api_key == "" or base_api_key is None:
            base_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        openai.api_base = base_url
        openai.api_key =  base_api_key
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        
    
    def run_with_token_count(self, 
                         query: str,
                         system_prompt: str = "", 
                         show_token_consumption: bool = True, 
                         total_session_tokens: int = 0,
                         temperature: int = 0,
                         max_tokens: int = 1000,
                         stop=None):
        memory = ([
        { "role": "system", "content": system_prompt if system_prompt != "" else "You are an autonomous Agent."},
        { "role": "user", "content": query },
        ])
        
        total_session_tokens = sum([len(self.chatEncoding.encode(message["content"])) for message in memory])
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=memory,
            temperature=temperature,
            stream=self.stream,
            max_tokens=max_tokens,
            stop=stop,) 
        
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
            
            if show_token_consumption:
                print(colored("\nTokens used this time: " + str(tokens_used), "red"))
                print(colored("\nTokens used so far: " + str(total_session_tokens), "yellow"))
        else:
            return response["choices"][0]["message"]["content"]
        
    def run(self, query, system_prompt: str = "", max_tokens: int = 1000, temperature: int = 0, stop=None):
        while True:
            try:
                messages = [
                    { "role": "system", "content": system_prompt if system_prompt != "" else "You are an autonomous Agent."},
                    {"role": "user", "content": query}
                    ]
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    )
                with open("openai.logs", "a") as log_file:
                    log_file.write(
                        "\n" + "-----------" + "\n" + "System Prompt : " + system_prompt + "\n" +
                        "\n" + "-----------" + "\n" + "Prompt : " + query + "\n"
                    )
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError as e:
                sleep_duration = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(
                    f"{str(e)}, sleep for {sleep_duration}s, set it by env OPENAI_RATE_TIMEOUT"
                )
                time.sleep(sleep_duration)
    
   
from dotenv import load_dotenv
load_dotenv()
import os
    
CHIMERA_GPT_KEY = os.getenv('CHIMERA_GPT_KEY')
ZUKI_API_KEY = os.getenv('ZUKI_API_KEY')
WEBRAFT_API_KEY = os.getenv('WEBRAFT_API_KEY')
NOVA_API_KEY = os.getenv('NOVA_API_KEY')
OPEN_AI_BASE = 'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://thirdparty.webraft.in/v1" #"https://zukijourney.xyzbot.net/v1"  #'https://api.nova-oss.com/v1' #"https://thirdparty.webraft.in/v1" # #"https://api.naga.ac/v1"

llm = OpenAI(model="gpt-3.5-turbo", stream=True)
#llm.run("What is the weather in New York?")
print(llm.run(query="Explain to me embeddings and vector databases", max_tokens=400, temperature=0.0, stop=None))