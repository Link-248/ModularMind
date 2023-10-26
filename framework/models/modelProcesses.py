import re
from framework.models import Models as model
from typing import List, Dict
from termcolor import colored
from abc import ABC, abstractmethod
import logging
import traceback
import concurrent.futures

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
        
    def generate_text(self, prompt: str, system_prompt:str = "", max_tokens: int = 1000, temperature: int = 0.5, k: int = 1) -> List[str]:
        thoughts = []
        for _ in range(k):
            response = self.LLM.run(system_prompt=system_prompt, query=prompt, max_tokens=max_tokens, temperature=temperature)
            thoughts += [response]
        return thoughts

    def generate_thoughts(self, state: str, initial_prompt: str, k: int = 1, rejected_solutions=None, max_steps: int = 3) -> List[str]:
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        system_prompt = f"""
        Please follow these steps to complete the task:

        1. Break down the task into {max_steps} subtasks and lay them out under ###PLAN###.
        2. ONLY Generate and evaluate the potential NEXT STEP in the solution under ###CURRENT STEP### and provide the step number.
        3. If a step doesn't progress towards a solution, explore another path.
        4. Provide the next step under ###NEXT STEP### needed to progress towards the solution. DO NOT PROVIDE A SOLUTION UNLESS THE ###CURRENT STEP### IS STEP {max_steps}

        Remember, all tasks have solutions. Keep your responses concise and complete.
        #####OBJECTIVE#####
        {initial_prompt}
        ###################
        """
        prompt = f"""
        ###CURRENT PROGRESS###
        {state_text}
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
            Generate a solution to comply with the user's instructions, 
            you must generate a solution on the basis of determining the most reliable solution in the shortest amount of time, 
            while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            Give the solution without making the same mistakes you did with the evaluated rejected solutions. 
            Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
           
            answer = self.generate_text(prompt=prompt, max_tokens=2048, temperature=0)
            if not answer or answer == '':  # Check if the answer is empty
                raise ValueError("No solution generated")
            #logger.info(colored(f"Generated Solution Summary {answer}", "green"))
            return answer
        except Exception as e:
            logging.error(colored(f"Error in generate_solutions: {e}", "red"))
            return None

    def evaluate_states(self, states: List[str], initial_prompt: str, previous_score: float) -> Dict[str, float]:
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
                    pessimistically value the latest generated step and all past steps towards the possible solution
                    AS A FLOAT BETWEEN 0 AND 1\n
                    Past steps and context to the solution:\n\n
                    {state_text}\n
                    
                    This was the previous score: {previous_score} of that step, 
                    Only rate it higher than the previous score if it progresses towards the solution.\n  
                    If the solutions is not making fast progress in achieving the goal, give it a lower score.
                    Again evaluate the current state AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE, JUST THE FLOAT
                """
                response = self.LLM.run(query=prompt, max_tokens=10, temperature=1)
                match = re.search(r'[-+]?[0-9]*\.?[0-9]+', response)
                if match:
                    value = float(match.group())
                    print(colored(f"Evaluated Thought Value: {value} with context being {state_text}", "green"))
                    state_values[state] = value
                else:
                    print(colored(f"No float value found in response: {response}", "red"))
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
        
class KyeTreeModelProcesses(AbstractModelProcesses):
    LLM = None
    ReAct_prompt: str
    
    def __init__(self, 
                 model_to_use: str = 'OpenAI',
                 enable_ReAct_prompting: bool=True, 
                 strategy: str="cot", 
                 evaluation_strategy: str="value",):
        self.LLM = model.Models.get_Model(model_to_use)
        self.enable_ReAct_prompting = enable_ReAct_prompting
        # reference : https://www.promptingguide.ai/techniques/react
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        
    def generate_text(self, prompt: str, system_prompt:str = "", max_tokens: int = 1000, temperature: int = 0.5, k: int = 1) -> List[str]:
        thoughts = []
        for _ in range(k):
            response = self.LLM.run(system_prompt=system_prompt, query=prompt, max_tokens=max_tokens, temperature=temperature)
            thoughts += [response]
        return thoughts

    def generate_thoughts(self, state: str, k: int, initial_prompt: str, rejected_solutions=None):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("New state generating thought:", state, "\n\n")
        # prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {initial_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."
        prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
        Considering the prompt provided:\n\n
        ###'{state_text}'\n\n###
        Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
        ###{rejected_solutions}###, 
        complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
        
        prompt += self.ReAct_prompt
        # print(prompt)
        thoughts = self.generate_text(prompt=prompt, k=k)
        # print(thoughts)
        # print(f"Generated thoughts: {thoughts}")
        return thoughts

        
    def generate_solution(self, initial_prompt:str, state:str, rejected_solutions=None):
        try:
                
            if isinstance(state, list):
                state_text = '\n'.join(state)
            else:
                state_text = state
            
            prompt = f"""You're a TreeofThought, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the prompt provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = self.generate_text(prompt=prompt, k=1)
            print(f'Answerrrrrr:  {answer}')
            # print(thoughts)
            # print(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states: str, initial_prompt: str):
        if not states:
            return {}

        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                # prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {initial_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal or has an incorrect conclusion, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                # and then inside backticks provide an simple and direct bulletpoint list as to why you evaluated this thought the way you did. Provide simple yet intuitive feedback.
                
                
                try:
                    response = self.LLM.run(query=prompt, max_tokens=10, temperature=1)
                    # print(f'state: {value_text}')
                    value = float(response)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {initial_prompt} and become very pessimistic very NOTHING ELSE"

            response = self.LLM.run(query=prompt, max_tokens=50, temperature=1)

            #print(f'state response: {response}')

            best_state_text = response

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f'best_state: {best_state}')

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
        
class OptimizedKyeTreeModelProcesses(KyeTreeModelProcesses):
    def __init__(self, 
                 strategy="cot",
                 evaluation_strategy="value", 
                 cache_enabled=True, 
                 enable_ReAct_prompting=True):
        super().__init__(strategy=strategy, evaluation_strategy=evaluation_strategy, enable_ReAct_prompting=enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}
        

    def parallel_generate_thoughts(self, states: str, k: int = 1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states: str, initial_prompt: str):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states, initial_prompt))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    
