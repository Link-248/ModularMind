from framework.models.modelProcesses import AlgorithmModelProcesses
import json
from typing import List, Dict, Any, Tuple
from termcolor import colored

import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

import logging
import traceback

class CustomLogger(logging.Logger):
    def info(self, msg, *args, **kwargs):
        stack_trace = ''.join(traceback.format_stack())
        super().info(f"{msg}\nStack trace: {stack_trace}", *args, **kwargs)

# Create a custom logger
logging.setLoggerClass(CustomLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.encoding = 'utf-8'  # Set encoding to utf-8 for console handler

f_handler = logging.FileHandler('file.log', encoding='utf-8')  # Set encoding to utf-8 for file handler

# Create formatters and add it to handlers
format = logging.Formatter('%(name)s - %(message)s - %(lineno)d')
c_handler.setFormatter(format)
f_handler.setFormatter(format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Define constants
PRUNING_THRESHOLD = 0.5

class AoTAgent():
    """
    Algorithm of Thoughts
    ---------------------
    Credit to kyegomez on github as my AoTAgent implementation is 
    a heavy rework and reimplementation based on his implementation structure.
    This class implements the Algorithm of Thoughts (AoT) Agent. AoT is a
    general-purpose algorithm for solving problems. It is inspired by the
    human thought process and is based on the idea of generating thoughts and
    evaluating them.

    Parameters
    ----------
    model_type : str
        The model provider to use for the agent.
    model : str
        The model name to use for the agent.
    
    api_key : str
        The API key to use for the Agent.
    api_base : str
        The API base to use for the Agent.
        
    num_thoughts : int
        The number of thoughts to generate at each step of the Agent.
    max_steps : int
        The maximum number of steps to run the Agent for.
    value_threshold : float
        The minimum value of a thought to be considered valid.
    pruning_threshold : float
        The minimum value of a thought to be considered for caching.
    backtracking_threshold : float
        The minimum value of a thought to be considered for backtracking.
    initial_prompt : str
        The initial prompt to start the Agent with.
    thought_cache : dict
        The cache to use for the Agent.

    Returns
    -------
    solution : str
        The solution to the problem.
    """
    model: AlgorithmModelProcesses = None
    output: list 
    api_key: str
    api_base: str 
    num_thoughts: int 
    max_steps: int
    value_threshold: float
    pruning_threshold: float
    backtracking_threshold: float
    initial_prompt: str
    thought_cache: Dict[str, Any]
        
    def __init__(
        self,
        model_type: str = 'OpenAI',
        model: str = "gpt-3.5-turbo",
        api_key: str = None,
        api_base: str = 'https://api.openai.com/v1',
        num_thoughts: int = 2,
        max_steps: int = 1,
        value_threshold: float = 0.5,
        pruning_threshold: float = PRUNING_THRESHOLD,
        initial_prompt: str = None,
        thought_cache: Dict[str, Any] = None,
        valid_retry_count: int = 1,
    ):
        """Init method for AoT"""
        if thought_cache is None:
            self.thought_cache = {"accepted": {}, "pruned": {}}
        else:
            self.thought_cache = thought_cache
        self.num_thoughts = num_thoughts
        self.max_steps = max_steps
        self.value_threshold = value_threshold
        self.pruning_threshold = pruning_threshold
        self.initial_prompt = initial_prompt
        #self.output = []
        
        self.model = AlgorithmModelProcesses(model_type)
        self.model.LLM.set_api_info(base_api_key=api_key, base_url=api_base)
        self.model.LLM.model = model
        
        self.valid_retry_count = valid_retry_count
        self.evaluated_thoughts = {}
        self.last_state = ""
        self.nodeCount = 0
        self.graph = nx.DiGraph()  # Add this line to initialize the graph
        
        
        self.best_thoughts = {} # A dictionary to store the best thoughts per step
        self.state_stack = []

    def solve(self) -> str:
        """Solve the problem using AoT prompt and dfs search algorithm"""
        try:
            self.last_state = self.initial_prompt
            self.graph.add_node(0, state=self.initial_prompt, color='blue')
            #self.graph.add_node(self.nodeCount, state=self.initial_prompt)
            #self.nodeCount += 1
            # Run DFS
            self.dfs(self.initial_prompt, 1)

            self.get_best_thoughts_per_step()
            
            # Generate the final solution based on the best thought
            solution = self.model.generate_solution(initial_prompt=self.initial_prompt, best_steps=self.best_thoughts, rejected_solutions=self.thought_cache["pruned"])

            # Display and return the solution
            logger.info(f"Solution is {solution}")

            '''
            # Write cache to JSON file
            with open("./thought_cache.json", "a") as json_file:
                json.dump(self.thought_cache, json_file)'''
            # Draw the graph at the end of the solve method
           # pos = nx.spring_layout(self.graph, scale=10)  # This will calculate the positions of the nodes
            print(self.graph.edges(data=True))
            #labels = {node: data['state'] for node, data in self.graph.nodes(data=True)}
            #T = nx.dfs_tree(self.graph, source=0)  # This will create a tree from the graph
            nodes = self.graph.nodes(data=True)
            node_colors = [node['color'] for _, node in nodes]
            edges = self.graph.edges(data=True)
            edge_colors = [edge['color'] for _, _, edge in edges]
            pos = graphviz_layout(self.graph, prog='dot')
            try:
                nx.draw(self.graph, pos, with_labels=True, arrows=True, node_color=node_colors, edge_color=edge_colors)   
                plt.show() 
            except Exception as e :
                print(e) 
            ''' try:
                nx.draw(self.graph, labels=labels, with_labels=True)
                plt.show()
            except Exception as e:
                print(e)'''
            return solution

        except Exception as error:
            logger.error(f"Error in AoT_dfs: {error}")

            '''
            # Write cache to JSON file even if an error occurs
            with open("./thought_cache_error.json", "a") as json_file:
                json.dump(self.thought_cache, json_file)'''

            raise error

    def check_cache(self, state: str) -> float:
        """Check if the state is in the cache and return the corresponding value"""
        if state in self.thought_cache["accepted"]:
            value = self.thought_cache["accepted"][state]["value"]
            #print(f"Retrieved accepted thought value from cache: {value}")
        elif state in self.thought_cache["pruned"]:
            value = 0  # or whatever value you use for pruned thoughts
            #print(f"Retrieved pruned thought value from cache: {value}")
        else:
            value = None
        return value
            
    def dfs(self, state: str, step: int) -> None:
        """Depth-first search algorithm"""
        if step > self.max_steps:
            return
        
        state_node_count = self.get_node_number_from_state(state) if self.get_node_number_from_state(state) != None else self.nodeCount
        retry_count = 0
        
        # Push the current state onto the stack
        self.state_stack.append(state)
    
        while retry_count < self.valid_retry_count:
            last_state_value = self.evaluated_thoughts.get(state) if state in self.evaluated_thoughts else None
            thoughts = self.generate_and_filter_thoughts(state=state, last_score=last_state_value, current_step=step)
            # Check if any thought has a value above the threshold
            if any(self.evaluated_thoughts[thought] > self.value_threshold for thought in thoughts):
                break
            retry_count += 1
            
        print(colored("Step: " + str(step), "red"))
        for next_state in thoughts:
            next_state_value = self.evaluated_thoughts.get(next_state)
                
            # check thoughts less than the value threshold and cache pruned thoughts 
            if next_state_value <= self.value_threshold:
                self.thought_cache["pruned"][next_state] = next_state_value
                if next_state in self.thought_cache["accepted"]:
                    del self.thought_cache["accepted"][next_state]
                    
                print(colored(f"Pruned thought under {self.value_threshold}: value: {next_state_value}", "red"))  
                # Add the pruned thought to the graph with a different color
                next_state_node_count = self.get_node_number_from_state(next_state)
                if next_state_node_count == self.nodeCount:
                    next_state_node_count += 1
                    self.nodeCount += 1
                self.graph.add_node(next_state_node_count, state=next_state, color='red')
                self.graph.add_edge(state_node_count, next_state_node_count, color='black')
                
                # Add a backtracking edge to the graph
                if step > 0:  # Don't add a backtracking edge for the root node
                    previous_state = self.state_stack[-1]  # Get the state to backtrack to from the stack
                    self.graph.add_edge(self.get_node_number_from_state(next_state), self.get_node_number_from_state(previous_state), color='red')  # Add a backtracking edge
                        
                    # Pop the current state from the stack
                    self.state_stack.pop()
                child = previous_state
                
            else:           
                # Explore the next state
                next_state_node_count = self.get_node_number_from_state(next_state) 
                if next_state_node_count == self.nodeCount:
                    next_state_node_count += 1
                self.add_nodes_and_edge(state_node_count, False, state, 
                                        next_state_node_count, True, next_state)
                child = next_state
            self.dfs(child, step + 1)
        if (not thoughts):
            self.dfs(state, step + 1)
            

    def generate_and_filter_thoughts(self, state: str, last_score: float, current_step: int) -> List[str]:
        """Generate and filter thoughts"""
        state_node_count = self.get_node_number_from_state(state) if self.get_node_number_from_state(state) != None else self.nodeCount

        self.last_state = state
        
        thoughts = self.model.generate_thoughts(
                state=state, k=self.num_thoughts, initial_prompt=self.initial_prompt, accepted_solutions=self.thought_cache["accepted"], max_steps=self.max_steps, current_step=current_step
            )
        #print(thoughts)
        for thought in thoughts.copy():
            #self.add_nodes_and_edge(state_node_count, False, state, self.nodeCount + 1, True, thought)
            # Check if thoughts for this state are cached
            cached_value = self.check_cache(thought)
            if cached_value is not None:
                print(colored(f"cached state: {thought}, Cached value: {cached_value}", "cyan"))
                existing_node_number = self.get_node_number_from_state(thought)
                if existing_node_number is not None:
                    node_number = existing_node_number
                else:
                    raise Exception("Node number not found")
                    #self.nodeCount += 1
                    #node_number = self.nodeCount
                self.graph.add_node(node_number, state=thought, color='purple')
                self.graph.add_edge(state_node_count, node_number, color='purple')
                thoughts.remove(thought)
                
        self.get_best_thoughts_per_step_no_nodes()
        
        new_evaluations = self.model.evaluate_states(
            states=thoughts, initial_prompt=self.initial_prompt, previous_score=last_score, current_step=current_step, 
            previous_best_thoughts=self.best_thoughts
        )
        self.evaluated_thoughts.update(new_evaluations)
            
        filtered_thoughts = [
            thought
            for thought in thoughts
            if self.evaluated_thoughts[thought] >= self.pruning_threshold
        ]

        # Cache the filtered thoughts
        # Cache the filtered thoughts
        for thought in filtered_thoughts:
            self.thought_cache["accepted"][str(thought)] = {"value": self.evaluated_thoughts[thought], "step": current_step}
        #print(self.thought_cache["accepted"].items())
        for thought in thoughts:
            if self.evaluated_thoughts[thought] < self.pruning_threshold:
                self.thought_cache["pruned"][str(thought)] = self.evaluated_thoughts[thought]
                print(colored(f"Pruned thought under {self.pruning_threshold}: value: {thought}", "red"))
                thought_node_count = self.get_node_number_from_state(thought)
                if thought_node_count == self.nodeCount:
                    thought_node_count += 1
                    self.nodeCount += 1
                # Add the pruned thought to the graph with a different color
                self.graph.add_node(thought_node_count, state=thought, color='red')
                self.graph.add_edge(state_node_count, thought_node_count, color='black')

        #logger.info(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        #print(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        
        return filtered_thoughts
    
    def get_node_number_from_state(self, state):
        for node_number, data in self.graph.nodes(data=True):
            if 'state' in data and data['state'] == state:
                return node_number
        return self.nodeCount  
    
    def add_nodes_and_edge(self, node1_number, increment1:bool, state1, node2_number, increment2: bool, state2):
        # Check if nodes already exist
        self.graph.add_node(node1_number, state=state1, color='blue')
        self.graph.add_node(node2_number, state=state2, color='blue')
        if increment1 or increment2:
            self.nodeCount += 1
            
        # Add edge between nodes
        self.graph.add_edge(node1_number, node2_number, color='black')
        
    def get_best_thoughts_per_step(self):
        for thought, data in self.thought_cache["accepted"].items():
            step = data["step"]
            value = data["value"]
            if step not in self.best_thoughts or value > self.best_thoughts[step]["value"]:
                self.best_thoughts[step] = {"thought": thought, "value": value}

        # Change the color of the nodes that represent the best thoughts
        for node, data in self.graph.nodes(data=True):
            if 'state' in data and data['state'] in (item["thought"] for item in self.best_thoughts.values()):
                data['color'] = 'green'  # Change the color to green
                
    def get_best_thoughts_per_step_no_nodes(self):
        for thought, data in self.thought_cache["accepted"].items():
            step = data["step"]
            value = data["value"]
            if step not in self.best_thoughts or value > self.best_thoughts[step]["value"]:
                self.best_thoughts[step] = {"thought": thought, "value": value}