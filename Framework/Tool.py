from typing import Callable, Optional

class Tool():
    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    func: Optional[Callable[..., str]]
    description: str
    """Used to tell the model how/when/why to use the tool.
    
    You can provide few-shot examples as a part of the description.
    """
    
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
        
    def run(self, *args):
        return self.func(*args)
        
#Example usage
'''from search import search

searchTool = Tool(
    name="Search",
    func=search,
    description="useful for when you need to answer questions about current events and data. You should ask targeted questions"
)

print(searchTool.run("What is the weather in New York?"))'''