from framework.models.modelProcesses import AlgorithmModelProcesses
import json
from typing import List, Dict, Any, Tuple
from termcolor import colored

import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz


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
        
        self.best_state = None
        self.best_value = float('-inf')
        self.best_state_node = None
        
        self.valid_retry_count = valid_retry_count
        self.results = []
        self.evaluated_thoughts = {}
        self.last_state = ""
        self.nodeCount = 0
        self.graph = nx.DiGraph()  # Add this line to initialize the graph

    def solve(self) -> str:
        """Solve the problem using AoT prompt and dfs search algorithm"""
        try:
            self.last_state = self.initial_prompt
            
            #self.graph.add_node(self.nodeCount, state=self.initial_prompt)
            #self.nodeCount += 1
            # Run DFS
            self.dfs(self.initial_prompt, 1)

            # Find the best thought and its value
            if(not self.results):
                print(colored(f"results: {self.results}", "green"))
                best_state, best_value = max(self.results, key=lambda x: x[1])
            elif(not self.best_state):
                best_state = self.best_state
            else:
                best_state = self.initial_prompt

            # Generate the final solution based on the best thought
            solution = self.model.generate_solution(initial_prompt=self.initial_prompt, state=best_state, rejected_solutions=self.thought_cache["pruned"])

            # Display and return the solution
            logger.info(f"Solution is {solution}")

            '''
            # Write cache to JSON file
            with open("./thought_cache.json", "a") as json_file:
                json.dump(self.thought_cache, json_file)'''
            # Draw the graph at the end of the solve method
           # pos = nx.spring_layout(self.graph, scale=10)  # This will calculate the positions of the nodes
            print(self.graph.edges(data=True))
           # labels = {node: data['state'] for node, data in self.graph.nodes(data=True)}
            T = nx.dfs_tree(self.graph, source=0)  # This will create a tree from the graph
            pos = nx.drawing.nx_agraph.graphviz_layout(T, prog='dot')
            try:
                nx.draw(T, pos, with_labels=True, arrows=True)   
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
            value = self.thought_cache["accepted"][state]
            print(f"Retrieved accepted thought value from cache: {value}")
        elif state in self.thought_cache["pruned"]:
            value = 0  # or whatever value you use for pruned thoughts
            print(f"Retrieved pruned thought value from cache: {value}")
        else:
            value = None
        return value

    def dfs(self, state: str, step: int) -> None:
        """Depth-first search algorithm"""
        if step > self.max_steps:
            value = self.check_cache(state)
            if value is not None or 0:
                self.results.append((state, value))
            return

        retry_count = 0
        while retry_count < self.valid_retry_count:
            last_state_value = self.evaluated_thoughts.get(state) if state in self.evaluated_thoughts else None
            thoughts = self.generate_and_filter_thoughts(state=state, last_score=last_state_value, last_step=step, last_state=self.last_state, current_step=step)
            # Check if any thought has a value above the threshold
            if any(self.evaluated_thoughts[thought] > self.value_threshold for thought in thoughts):
                break
            retry_count += 1
        if not thoughts:
            self.last_state = state
            self.dfs(state=state, step=step + 1)
        
        print(colored("Step: " + str(step), "red"))
        for next_state in thoughts:
            print(colored(f"Next state val: {self.evaluated_thoughts.get(next_state)}, next state: {next_state}", "cyan"))
            if(next_state in self.evaluated_thoughts):
                next_state_value = self.evaluated_thoughts.get(next_state)
            else:
                print("ERROR: next_state not in evaluated_thoughts")
                continue
            #logger.info(f"Entering DFS with state: {next_state} and step: {step}")
            print(f"Entering DFS with state: {next_state} and step: {step}")
            
            # check thoughts less than the value threshold and cache pruned thoughts 
            if next_state_value <= self.value_threshold:
                self.thought_cache["pruned"][next_state] = next_state_value
                if next_state in self.evaluated_thoughts:
                    del self.evaluated_thoughts[next_state]
                print(colored(f"Pruned thought under {self.value_threshold}: value: {next_state_value}", "red"))  
                        
            
            if self.evaluated_thoughts:
                best_state, best_value = max(self.evaluated_thoughts.items(), key=lambda item: item[1])
                best_state_nodeCount = self.get_node_number_from_state(best_state)
                print(colored(f"Best state: {best_state}, best value: {best_value}", "red"))
                #backtracking
                if best_value > next_state_value:
                    print(colored(f"Backtracking to: {best_state}", "yellow"))
                    if next_state in self.evaluated_thoughts:
                        del self.evaluated_thoughts[next_state]
                    child = best_state
                    self.add_nodes_and_edge(self.get_node_number_from_state(next_state), False, next_state, 
                                            best_state_nodeCount, False, best_state)
                    self.nodeCount = best_state_nodeCount
                else: 
                    print(colored(f"continue with {next_state}", "yellow")) 
                    child = next_state
                    self.add_nodes_and_edge(self.get_node_number_from_state(state), False, state, 
                                            self.get_node_number_from_state(next_state), False, next_state)
            else:
                child = state
                self.add_nodes_and_edge(self.get_node_number_from_state(next_state), False, next_state, 
                                        self.get_node_number_from_state(state), False, state)
                  
                
               
            if step <= self.max_steps:
                self.last_state = state
                self.dfs(state=child, step=step + 1)

    def generate_and_filter_thoughts(self, state: str, last_score: float, last_step:int, last_state: str, current_step: int) -> List[str]:
        """Generate and filter thoughts"""
        thoughts = self.model.generate_thoughts(
                state=state, k=self.num_thoughts, initial_prompt=self.initial_prompt, max_steps=self.max_steps, current_step=current_step-1
            )
        
        state_node_count = self.get_node_number_from_state(state) if self.get_node_number_from_state(state) != None else self.nodeCount - 1
        
        for thought in thoughts:
            self.add_nodes_and_edge(state_node_count, False, state, self.nodeCount + 1, True, thought)
            # Check if thoughts for this state are cached
            cached_value = self.check_cache(state)
            if cached_value is not None:
                print(colored(f"cached state: {state}, Cached value: {cached_value}", "cyan"))
                thoughts.remove(thought) 
            
            
            
            
        new_evaluations = self.model.evaluate_states(
            states=thoughts, initial_prompt=self.initial_prompt, previous_score=last_score, current_step=current_step
        )
        self.evaluated_thoughts.update(new_evaluations)
            
        filtered_thoughts = [
            thought
            for thought in thoughts
            if self.evaluated_thoughts[thought] >= self.pruning_threshold
        ]

        # Cache the filtered thoughts
        for thought in filtered_thoughts:
            self.thought_cache["accepted"][str(thought)] = self.evaluated_thoughts[thought]

        for thought in thoughts:
            if self.evaluated_thoughts[thought] < self.pruning_threshold:
                self.thought_cache["pruned"][str(thought)] = self.evaluated_thoughts[thought]
                print(colored(f"Pruned thought under {self.pruning_threshold}: value: {thought}", "red"))

        #logger.info(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        #print(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        
        return filtered_thoughts
    
    def get_node_number_from_state(self, state):
        for node_number, data in self.graph.nodes(data=True):
            if data['state'] == state:
                return node_number
        return None  # return None if no matching node is found
    
    def add_nodes_and_edge(self, node1_number, increment1:bool, state1, node2_number, increment2: bool, state2):
        # Check if nodes already exist
        self.graph.add_node(node1_number, state=state1)
        self.graph.add_node(node2_number, state=state2)
        if increment1 or increment2:
            self.nodeCount += 1
            
        # Add edge between nodes
        self.graph.add_edge(node1_number, node2_number)