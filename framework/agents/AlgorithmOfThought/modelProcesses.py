from framework.models import Models as model
from termcolor import colored

LLM = model.Models

class ModelProcesses():
    LLM = None
    
    def __init__(self, model_to_use: str = 'OpenAI'):
        self.LLM = LLM.get_Model(model_to_use)
        
    def generate_text(self, prompt: str, max_tokens: int = 400, temperature: int = 0, k: int = 1):
        thoughts = []
        for _ in range(k):
            response = self.LLM.run(query=prompt, max_tokens=max_tokens, temperature=temperature, k=k)
            thoughts += [response]
            # print(f'thoughts: {thoughts}')
        return thoughts

    def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        print("New state generating thought:", state_text, "\n\n")
        prompt = f"""
        Accomplish the task below by decomposing it as many very explicit subtasks as possible, be very explicit and thorough denoted by 
        a search process, highlighted by markers ‘1’,..., ‘3’ as “first operations” guiding subtree exploration for the OBJECTIVE, 
        focus on the third subtree exploration. Produce prospective search steps (e.g., the subtree exploration ‘5. 11 + 1’) 
        and evaluates potential subsequent steps to either progress
        towards a solution or retrace to another viable subtree then be very thorough 
        and think atomically then provide solutions for those subtasks, 
        then return the definitive end result and then summarize it


        ########## OBJECTIVE
        {initial_prompt}
        ###################
        """
        thoughts = self.generate_text(prompt, k)
        # print(f"Generated thoughts: {thoughts}")
        return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
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
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = self.generate_text(prompt, 1)
            print(colored(f"Generated Solution Summary {answer}", "green"), end='', flush=True)
            return answer
        except Exception as e:
            print(colored(f"Error in generate_solutions: {e}", "red"), end='', flush=True)
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.LLM.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    print(f"state: {state}")
                    state_text = "\n".join(state)
                print(
                    "We receive a state of type",
                    type(state),
                    "For state: ",
                    state,
                    "\n\n",
                )
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                response = self.LLM.run(prompt, 10, 1)
                try:
                    value_text = response
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(colored(f"Evaluated Thought Value: {value}", "green"), end='', flush=True)
                except ValueError:
                    value = 0
                state_values[state] = value
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")