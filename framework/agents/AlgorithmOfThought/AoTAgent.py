from framework.agents.AlgorithmOfThought.modelProcesses import ModelProcesses
import json
from typing import List, Dict, Any, Tuple
from framework.models import Models as model
from termcolor import colored

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
f_handler = logging.FileHandler('file.log')

# Create formatters and add it to handlers
format = logging.Formatter('%(name)s - %(message)s - %(lineno)d')
c_handler.setFormatter(format)
f_handler.setFormatter(format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Define constants
PRUNING_THRESHOLD = 0.5
BACKTRACKING_THRESHOLD = 0.4

class AoTAgent():
    """
    Algorithm of Thoughts
    ---------------------
    Credit to kyegomez on github as the AoTAgent implementation is based on his work.
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
    model: ModelProcesses = None
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
        backtracking_threshold: float = BACKTRACKING_THRESHOLD,
        initial_prompt: str = None,
        thought_cache: Dict[str, Any] = None,
    ):
        """Init method for AoT"""
        if thought_cache is None:
            self.thought_cache = {"accepted": {}, "pruned": {}}
        else:
            self.thought_cache = thought_cache
        self.num_thoughts = num_thoughts
        self.max_steps = max_steps
        self.value_threshold = value_threshold
        self.backtracking_threshold = backtracking_threshold
        self.pruning_threshold = pruning_threshold
        self.initial_prompt = initial_prompt
        self.output = []
        
        self.model = ModelProcesses(model_type)
        self.model.LLM.set_api_info(base_api_key=api_key, base_url=api_base)
        self.model.LLM.model = model
        
        self.best_state = None
        self.best_value = float('-inf')

    def solve(self) -> str:
        """Solve the problem using AoT prompt and dfs search algorithm"""
        try:
            # Run DFS
            self.dfs(self.initial_prompt, 1)

            # Check if any thoughts were generated
            if not self.output:
                logger.error("No valid thoughts were generated during DFS")
                return None

            # Find the best thought and its value
            best_state, best_value = max(self.output, key=lambda x: x[1])
            print(colored(f"Best state: {best_state}, best value: {best_value}", "red"))
            # Cache the best thought
            self.thought_cache["accepted"][best_state] = best_value

            # Generate the final solution based on the best thought
            solution = self.model.generate_solution(initial_prompt=self.initial_prompt, state=best_state, rejected_solutions=self.thought_cache["pruned"])

            # Display and return the solution
            logger.info(f"Solution is {solution}")

            '''
            # Write cache to JSON file
            with open("./thought_cache.json", "a") as json_file:
                json.dump(self.thought_cache, json_file)'''

            return solution if solution else best_state

        except Exception as error:
            logger.error(f"Error in tot_dfs: {error}")

            '''
            # Write cache to JSON file even if an error occurs
            with open("./thought_cache_error.json", "a") as json_file:
                json.dump(self.thought_cache, json_file)'''

            raise error

    def check_cache(self, state: str) -> float:
        """Check if the state is in the cache and return the corresponding value"""
        if state in self.thought_cache["accepted"]:
            value = self.thought_cache["accepted"][state]
            logger.info(f"Retrieved accepted thought value from cache: {value}")
        elif state in self.thought_cache["pruned"]:
            value = 0  # or whatever value you use for pruned thoughts
            logger.info(f"Retrieved pruned thought value from cache: {value}")
        else:
            value = None
        return value

    def dfs(self, state: str, step: int) -> None:
        """Depth-first search algorithm"""
        if step > self.max_steps:
           # Check cache before evaluating
            if state in self.thought_cache["accepted"]:
                value = self.thought_cache["accepted"][state]
            elif state in self.thought_cache["pruned"]:
                return
            else:
                thought, value = self.evaluate_thought(state)
                # Cache the evaluated thought
                self.thought_cache["accepted"][state] = value

            self.output.append((state, value))
            return
                
        logger.info(step)
        
         # Check cache before generating and filtering
        if state in self.thought_cache["accepted"]:
            thoughts = [state]
        elif state in self.thought_cache["pruned"]:
            return
        else:
            print(colored("Step: " + str(step), "red"))
            thoughts = self.generate_and_filter_thoughts(state)

         # Find the thought with the highest value and add it to the output list
        if thoughts:
            best_thought = max(thoughts, key=lambda t: self.evaluated_thoughts.get(t, 0))
            self.output.append((best_thought, self.evaluated_thoughts[best_thought]))
        
        for next_state in thoughts:
            next_state_value = self.evaluated_thoughts.get(next_state, 0)
            #logger.info(f"Entering DFS with state: {next_state} and step: {step}")
            
            # Cache pruned thoughts
            if next_state_value <= self.value_threshold:
                self.thought_cache["pruned"][next_state] = next_state_value
                continue

            child = (
                (str(state), str(next_state)) if isinstance(state, str) else (*map(str, state), str(next_state))
            )
            self.dfs(state=child, step=step + 1)

            #backtracking
            if self.output:  # Check if self.output is not empty
                best_value = max([value for _, value in self.output])

                if best_value < self.backtracking_threshold:
                    self.output.pop()
                    continue

        # Add to output regardless of whether the state is in the cache
        '''thought, value = self.evaluate_thought(state)
        if thought not in [t for t, _ in self.output]:
            self.output.append((thought, value))'''

    def generate_and_filter_thoughts(self, state: str) -> List[str]:
        """Generate and filter thoughts"""
        # Check if thoughts for this state are cached
        cached_value = self.check_cache(state)
        if cached_value is not None:
            print(colored([state] if cached_value > 0 else [], "yellow"))
            return [state] if cached_value > 0 else []

        thoughts = []
        # Else generate new thoughts
        for i in range(self.num_thoughts):
            thoughts.extend(self.model.generate_thoughts(
                state=state, initial_prompt=self.initial_prompt
            ))

        self.evaluated_thoughts = self.model.evaluate_states(
            states=thoughts, initial_prompt=self.initial_prompt
        )

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

        logger.info(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        #print(colored(f"filtered_thoughts: {filtered_thoughts}", "yellow"))
        
        return filtered_thoughts

    def evaluate_thought(self, state: str) -> Tuple[str, float]:
        """Evaluate a thought"""
        # Check if the thought is already in the cache
        cached_value = self.check_cache(state)
        if cached_value is not None:
            print(colored(f"{state}, Evaluated thought: {cached_value}", "yellow"))
            return state, cached_value

        # Otherwise, evaluate the thought
        thought = self.model.generate_thoughts(state=state, k=1, initial_prompt=self.initial_prompt)
        value = self.model.evaluate_states(states=[state], initial_prompt=self.initial_prompt)[state]

        # Update best state and value if a better state is found
        if value > self.best_value:
            self.best_value = value
            self.best_state = state
        
        # Update the cache based on the evaluation
        if value >= self.pruning_threshold:
            self.thought_cache["accepted"][str(state)] = value
        else:
            self.thought_cache["pruned"][str(state)] = value
            
        #logger.info(f"Evaluated thought: {value}")
        
        return thought, value