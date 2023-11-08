import re
from framework.models import Models as model
from typing import List, Dict
from termcolor import colored
from abc import ABC, abstractmethod
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

class AbstractModelProcesses(ABC):
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt:str = "", max_tokens: int = 1000, temperature: int = 0.5, k: int = 1):
        pass
    @abstractmethod
    def generate_thoughts(self, state: str, initial_prompt: str, k: int = 1, rejected_solutions=None):
        pass
    @abstractmethod
    def generate_solution(self, initial_prompt: str, state: str, rejected_solutions=None):
        pass
    @abstractmethod
    def evaluate_states(self, states: List[str], initial_prompt: str):
        pass

class AlgorithmModelProcesses(AbstractModelProcesses):
    LLM = None
    
    def __init__(self, model_to_use: str = 'OpenAI'):
        self.LLM = model.Models.get_Model(model_to_use)
        
    def generate_text(self, prompt: str, system_prompt:str = "", max_tokens: int = 1000, temperature: int = 0, k: int = 1) -> List[str]:
        thoughts = []
        for _ in range(k):
            response = self.LLM.run(system_prompt=system_prompt, query=prompt, max_tokens=max_tokens, temperature=temperature)
            thoughts += [response]
        return thoughts

    def generate_thoughts(self, state: str, initial_prompt: str, k: int = 1, accepted_solutions = None, rejected_solutions=None, max_steps: int = 3, current_step: int = 0) -> List[str]:
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        system_prompt = f"""
        Follow these steps to complete the task:

        1. Break down the task into {max_steps} subtasks and lay them out under ###PLAN###. DO NOT write it again if {current_step} is greater than 1.
        2. Write step {current_step} towards solving the question under ###STEP {current_step}###. NOTHING MORE 
        3. Write step {current_step} towards solving the question under ###STEP {current_step}###. NOTHING MORE 
        4. Write step {current_step} towards solving the question under ###STEP {current_step}###. NOTHING MORE  
        \n
        DO NOT PROVIDE A FINAL SOLUTION UNLESS {current_step} IS EQUAL TO {max_steps}
        DO NOT PROVIDE A FINAL SOLUTION UNLESS {current_step} IS EQUAL TO {max_steps}
        DO NOT PROVIDE A FINAL SOLUTION UNLESS {current_step} IS EQUAL TO {max_steps}
        
        #####OBJECTIVE#####
        {initial_prompt}
        ###################\n
        ###PREVIOUS STATES###
        {accepted_solutions}
        ###CURRENT STATE####\n
        {state_text}
        """
        prompt = f"Generate step {current_step} towards the solution."

        thoughts = self.generate_text(system_prompt=system_prompt, prompt=prompt, temperature=1, k=k)
        return thoughts

    #Need to add in previous best steps per stage, so highest value per stage and give it here to generate the solution.
    def generate_solution(self, initial_prompt: str, best_steps, rejected_solutions=None) -> str:
        try:
            '''if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state'''

            prompt = f"""
            Generate a solution to comply with the user's instructions, 
            you must generate a solution on the basis of determining the most reliable solution in the shortest amount of time, 
            while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{best_steps}'###\n\n
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated steps that were rejected: 
            ###{rejected_solutions}###\n\n
            Give the solution without making the same mistakes you did with the evaluated rejected steps. 
            Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
           
            answer = self.generate_text(prompt=prompt, max_tokens=2048, temperature=0)
            if not answer or answer == '':  # Check if the answer is empty
                raise ValueError("No solution generated")
            #logger.info(colored(f"Generated Solution Summary {answer}", "green"))
            return answer
        except Exception as e:
            logging.error(colored(f"Error in generate_solutions: {e}", "red"))
            return None

    def evaluate_states(self, states: List[str], initial_prompt: str, previous_score: float, current_step: int = 0, previous_best_thoughts = None) -> Dict[str, float]:
        if not states:
            return {}

        if self.LLM.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                prompt = f""" To achieve the following goal: '{initial_prompt}', 
                    pessimistically value the latest generated step and it's accuracy
                    AS A FLOAT BETWEEN 0 AND 100.\n
                    If this state has another step that is not step {current_step} in it at once, rank it lower. Having a PLAN is NOT BAD\n
                    current state to the solution:\n\n
                    {state_text}\n
                    {previous_score} was the previous score of the last state this step branches from, 
                    ###{previous_best_thoughts}### are the previous best states/steps,
                    Only rate it higher than the previous score if it is step {current_step} towards the solution.\n  
                    Again evaluate the current state AS A FLOAT BETWEEN 0 and 100:\n,  DO NOT RETURN ANYTHING ELSE, JUST THE FLOAT
                """
                # If the solutions is not making fast progress in achieving the goal, give it a lower score.
                response = self.LLM.run(query=prompt, max_tokens=10, temperature=1)
                match = re.search(r'[-+]?[0-9]*\.?[0-9]+', response)
                if match:
                    value = float(match.group())
                    print(colored(f"Evaluated Thought Value: {value} at step: {current_step} with context being {state_text}", "green"))
                    state_values[state] = value
                else:
                    print(colored(f"No float value found in response: {response}", "red"))
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
        
