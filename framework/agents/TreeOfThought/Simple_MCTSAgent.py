import time
from framework.models.modelProcesses import OptimizedKyeTreeModelProcesses, KyeTreeModelProcesses
import json
from typing import Any, Dict, Union
from termcolor import colored
import numpy as np
import logging
import traceback
import json
import os

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
f_handler = logging.FileHandler('file.log')

# Create formatters and add it to handlers
format = logging.Formatter('%(name)s - %(message)s - %(lineno)d')
c_handler.setFormatter(format)
f_handler.setFormatter(format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


class Simple_MCTSAgent():
    """
    Tree of Thoughts
    ---------------------
    Credit to kyegomez on github as the ToTAgent implementation is based on his work.
    This class implements the Tree of Thoughts (AoT) Agent.

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
    Optimized : bool
        Whether to use the optimized version of the model type.

    Returns
    -------
    solution : str
        The solution to the problem.
    """
    model: OptimizedKyeTreeModelProcesses = None
    output: list 
    api_key: str
    api_base: str 
    
    def __init__(self,
        model_type: str = 'OpenAI',
        optimized: bool = False,
        model: str = "gpt-3.5-turbo",
        api_key: str = None,
        api_base: str = 'https://api.openai.com/v1'):
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = [] #added line initalize history
        if optimized:
            self.model = OptimizedKyeTreeModelProcesses(model_type)
        else:
            self.model = KyeTreeModelProcesses(model_type)
        self.model.LLM.set_api_info(base_api_key=api_key, base_url=api_base)
        self.model.LLM.model = model


    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def logNewState(self, state, evaluation):
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree['nodes']:
            self.tree['nodes'][state]['thoughts'].append(evaluation)
        else:
            self.tree['nodes'][state] = {'thoughts': [evaluation]}

    def adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile):
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0 
        return max(np.percentile(values, percentile), 0.1)
    

    def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)

class MonteCarloAgent(Simple_MCTSAgent):
    """
    Tree of Thoughts: Monte Carlo Search
    ---------------------
    Credit to kyegomez on github as the MonteCarloToTAgent implementation is based on his work.
    This class implements the MonteCarloToTAgent which is a Tree of Thought 
    prompting strategy that uses the monte carlos search algorithm.

    Parameters
    ----------
    model_type : str
        The model provider to use for the agent.
    model : str
        The model name to use for the agent.
    objective="balance",
        What parameter to optimize for.
    api_key : str
        The API key to use for the Agent.
    api_base : str
        The API base to use for the Agent.
    Optimized : bool
        Whether to use the optimized version of the model type.

    Returns
    -------
    solution : str
        The solution to the problem.
    """
    def __init__(self, 
        model_type: str = 'OpenAI',
        model: str = "gpt-3.5-turbo",
        api_key: str = None,
        api_base: str = 'https://api.openai.com/v1', 
        objective="balance",
        optimized: bool = False,
        enable_ReAct_prompting: bool = True):
        super().__init__(optimized=optimized,model_type=model_type, model=model, api_key=api_key, api_base=api_base)
        self.objective = objective
        self.solution_found = False
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
            "metrics": {"thoughts": {}, "evaluations": {}},
        }
        self.model.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.model.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."


    def optimize_params(self, num_thoughts, max_steps, max_states):
        if self.objective == 'speed':
            num_thoughts = max(1, num_thoughts - 1)
            max_steps = max(1, max_steps - 1)
            max_states = max(1, max_states - 1)
        elif self.objective == 'reliability':
            num_thoughts += 1
            max_steps += 1
            max_states += 1
        elif self.objective == 'balanace':
            if self.solution_found:
                num_thoughts = max(1, num_thoughts - 1)
                max_steps = max(1, max_steps - 1)
                max_states = max(1, max_states - 1)
            else:
                num_thoughts += 1
                max_steps += 1
                max_states += 1
        
        return num_thoughts, max_steps, max_states

    def solve(self,
              initial_prompt: str,
              num_thoughts: int,
              max_steps: int,
              max_states: int,
              pruning_threshold: float,
            #   sleep_time: float,
              ):
        self.file_name = "logs/tree_of_thoughts_output_montecarlo.json"
        return self.monte_carlo_search(
            initial_prompt=initial_prompt,
            num_thoughts=num_thoughts,
            max_steps=max_steps,
            max_states=max_states,
            pruning_threshold=pruning_threshold,
            # sleep_time,
        )
#v3
    def monte_carlo_search(self,
                        initial_prompt: str,
                        num_thoughts: int,
                        max_steps: int,
                        max_states: int,
                        pruning_threshold: float,
                        ):
        current_states = [initial_prompt]
        state_values = {}
        visit_counts = {initial_prompt: 0}
        transposition_table = {}

        best_state = None
        best_value = float('-inf')

        for step in range(1, max_steps + 1):
            print(colored(f"Step: {step}", "yellow"))
            selected_states = []

            for state in current_states:
                print(colored(f"Current state: {state}", "blue"))
                if state not in transposition_table:
                    thoughts = self.model.generate_thoughts(state=state, k=num_thoughts, initial_prompt=initial_prompt)
                    evaluated_thoughts = self.model.evaluate_states(states=thoughts, initial_prompt=initial_prompt)

                    for thought, value in evaluated_thoughts.items():
                        flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
                        transposition_table[flattened_state] = value

                for thought, value in evaluated_thoughts.items():
                    flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
                    print(colored(f"Flattened state: {flattened_state}", "green"))

                    if flattened_state not in visit_counts:
                        visit_counts[flattened_state] = 0
                        selected_states.append(flattened_state)
                        state_values[flattened_state] = value
                        print(colored(f"New state: {flattened_state}", "magenta"))
                    elif visit_counts[state] > visit_counts[flattened_state]:
                        if visit_counts[flattened_state] == 0:
                            ucb1_value = float('inf')
                        else:
                            ucb1_value = value + np.sqrt(2 * np.log(visit_counts[state]) / visit_counts[flattened_state])
                        print(colored(f"UCB1 value: {ucb1_value}", "cyan"))
                        if ucb1_value >= pruning_threshold:
                            selected_states.append(flattened_state)
                            state_values[flattened_state] = value

                    if value > best_value:
                        best_state = flattened_state
                        best_value = value
                        print(colored(f"New best state: {best_state}", "red"))

                visit_counts[state] += 1

            if len(selected_states) > max_states:
                current_states = selected_states[:max_states]

            self.save_tree_to_json(self.file_name)

        if best_state is not None:
            solution = self.model.generate_solution(initial_prompt, best_state)
            return solution
        else:
            solution = self.model.generate_solution(initial_prompt, best_state)
            return solution if solution else best_state

