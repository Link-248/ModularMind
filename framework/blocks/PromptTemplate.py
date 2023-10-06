from typing import List
import Tool
from abc import ABC, abstractmethod

class PromptBase(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def format_messages(self):
        pass
    
    
class PromptTemplate(PromptBase):
     # The template to use
    template: str
    # The list of tools available
    tools: List[Tool.Tool]
    #a dictionary of input variables to be used in the prompt template and their values
    input_variables: dict[str: any]
    
    def __init__(self, template, tools, input_variables):
        self.template = template
        self.tools = tools
        self.input_variables = input_variables
    
    def format_messages(self, *args, **kwargs) -> str:
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        for key in self.input_variables.keys():
            #print(pair)
            kwargs[key] = self.input_variables[key]
        
        formatted = self.template.format(**kwargs)
        return formatted
    
    def __str__(self):
        return self.format_messages(template=self.template, tools=self.tools, input_variables=self.input_variables)
    
    
#Example usage
'''from search import search

PROMPT = """
You are an expert on problem solving with a focus of complex problems. I am going to give you the secret formula to respond to questions, and I want you to adhere to it indefinitely.
Answer the following questions and obey the following commands as best you can. 
 
You ONLY have access to the following tools: 

{tools}
 
You will receive a message, then you should start a loop and do one of two things
 
Option 1: You use one of the tools above to answer the question.
For this, you should ONLY use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Python REPL]
Action Input: the input to the action, to be sent to the tool. You will wait for the tool to respond before saying anything else.
 
After this, the human or tool will respond with an observation, and you will continue.
 
Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: your response to the human, summarizing what you did and what you learned. End your chain here.
 
Begin {Name}!

Question: {Question}
"""

searchTool = Tool(
    name="Search",
    func=search,
    description="useful for when you need to answer questions about current events and data. You should ask targeted questions"
)

promptT = PromptTemplate(template=PROMPT, 
                         input_variables={"Question": searchTool.name, "Name": "John"}, 
                         tools=[searchTool],)

print(promptT)'''