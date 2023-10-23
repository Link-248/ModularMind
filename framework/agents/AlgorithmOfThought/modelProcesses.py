from framework.models import Models as model
from typing import List, Dict, Any, Tuple
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

class ModelProcesses():
    LLM = None
    
    def __init__(self, model_to_use: str = 'OpenAI'):
        self.LLM = model.Models.get_Model(model_to_use)
        
    def generate_text(self, prompt: str, system_prompt:str = "", max_tokens: int = 1000, temperature: int = 0.5, k: int = 1) -> List[str]:
        thoughts = []
        for _ in range(k):
            response = self.LLM.run_with_streaming(system_prompt=system_prompt, query=prompt, max_tokens=max_tokens, temperature=temperature, k=k)
            thoughts += [response]
        return thoughts

    def generate_thoughts(self, state: str, initial_prompt: str, k: int = 1, rejected_solutions=None) -> List[str]:
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        system_prompt = f"""
        Please follow these steps to complete the task:

        1. Break down the task into minimal subtasks.
        2. Use markers like '1', '2', '3' to guide the exploration of the OBJECTIVE.
        3. Generate and evaluate potential next steps.
        4. If a step doesn't progress towards a solution, explore another path.
        5. Provide a solution for each subtask and summarize the final result.

        Remember, all tasks have solutions. Keep your responses concise and complete.
        """
        prompt = f"""
        (DO NOT INCLUDE THIS IN YOUR RESPONSE)
        #####OBJECTIVE#####
        {initial_prompt}
        ###################
        ###CURRENT STATE###
        {state_text}
        ###################
    """
        thoughts = self.generate_text(system_prompt=system_prompt, prompt=prompt, k=k)
        return thoughts

    def generate_solution(self, initial_prompt: str, state: str, rejected_solutions=None) -> str:
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""
            Generate a series of solutions to comply with the user's instructions, 
            you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, 
            while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. 
            Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
           
            answer = self.generate_text(prompt=prompt, max_tokens=2048, temperature=0)
            if not answer:  # Check if the answer is empty
                raise ValueError("No solution generated")
            logger.info(colored(f"Generated Solution Summary {answer}", "green"))
            return answer
        except Exception as e:
            logging.error(colored(f"Error in generate_solutions: {e}", "red"))
            return None

    def evaluate_states(self, states: List[str], initial_prompt: str) -> Dict[str, float]:
        if not states:
            return {}

        if self.LLM.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                response = self.LLM.run(query=prompt, max_tokens=10, temperature=1)
                try:
                    value = float(response)
                    logger.info(colored(f"Evaluated Thought Value: {value}", "green"))
                except ValueError:
                    value = 0
                state_values[state] = value
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")