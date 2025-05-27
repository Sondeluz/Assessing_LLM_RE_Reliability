import textwrap
import re
import random
import torch
import ast
from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLM:
    # 2B
    GEMMA_2B = "google/gemma-2-2b-it"

    # 3B
    LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
    PHI_3B = "microsoft/Phi-3.5-mini-instruct"

    # 8B
    LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"

    # 9B
    GEMMA_9B = "google/gemma-2-9b-it"


    models_for_testing = [GEMMA_2B,
                          LLAMA_3B,
                          PHI_3B,
                          LLAMA_8B,
                          GEMMA_9B]

    max_new_tokens = 1024

    chosen_model_name = None
    model = None
    tokenizer = None
    pipe = None

    def __init__(self, model_name, on_gpu = True, hf_token=None):
        self.chosen_model_name = model_name

        if model_name is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.__get_torch_dtype(),
                trust_remote_code=True,
                token=hf_token
            )

            if on_gpu:
                self.model = self.model.to(torch.cuda.current_device())

            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                           token=hf_token,
                                                           trust_remote_code=True)

            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def __get_torch_dtype(self):
        match self.chosen_model_name:
            case self.GEMMA_2B:
                return torch.bfloat16
            case self.LLAMA_3B:
                return torch.bfloat16
            case self.PHI_3B:
                return "auto"
            case self.LLAMA_8B:
                return torch.bfloat16
            case self.GEMMA_9B:
                return torch.bfloat16
            case _:
                return "unknown_model"

    def get_human_readable_model_name(self):
        match self.chosen_model_name:
            case self.GEMMA_2B:
                return "gemma_2b"
            case self.LLAMA_3B:
                return "llama_3b"
            case self.PHI_3B:
                return "phi_3b"
            case self.LLAMA_8B:
                return "llama_8b"
            case self.GEMMA_9B:
                return "gemma_9b"
            case _:
                return "unknown_model"

    def set_max_new_tokens(self, new_max):
        self.max_new_tokens = new_max

    def __clean_prompt(self, p):
        """
        Returns a clean prompt without whitespace on newlines
        (usually due to using triple-quoted strings)
        """
        return re.sub(r'\n[ \t]+', '\n', p)

    def get_text_response(self, prompt):
        """
        Returns the raw text response for the provided prompts
        """

        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            # Enforce greedy decoding
            do_sample=False
        )

        return outputs[0]["generated_text"][-1]["content"]

    def get_triples(self, system_prompt: str, user_prompt: str, allow_bad_triples=False):
        """
        Returns a list of tuples of 3 str elements, representing a list of triples, for the provided prompts

        In case of error (LLM hallucination causing an incorrect output format...), an empty list will
        be returned. In these cases, the model can be prompted again by re-executing the function.
        """

        raw_response = self.get_text_response(self.get_formatted_prompt(system_prompt, user_prompt))

        try:
             # Force triple elements containing apostrophes inside to have double quotes
            raw_response = re.sub(r"(\b\w+)'(\w+)", r"\1\\'\2", raw_response)

            triples = ast.literal_eval(self.__extract_list_from_text(textwrap.dedent(raw_response)))

            if isinstance(triples, list) and (allow_bad_triples or all(self.__is_valid_triple(t) for t in triples)):
                if allow_bad_triples:
                    return triples
                else: # Clean them
                    return list(map(self.__clean_triple, triples))
            else:
                return []
        except Exception as e:
            return []

    def get_triples_from_existing_response(self, raw_response: str, allow_bad_triples=False):
        """
        Returns a list of tuples of 3 str elements, representing a list of triples, for the provided prompts

        In case of error (LLM hallucination causing an incorrect output format...), an empty list will
        be returned. In these cases, the model can be prompted again by re-executing the function.
        """

        try:
             # Force triple elements containing apostrophes inside to have double quotes
            raw_response = re.sub(r"(\b\w+)'(\w+)", r"\1\\'\2", raw_response)

            triples = ast.literal_eval(self.__extract_list_from_text(textwrap.dedent(raw_response)))

            if isinstance(triples, list) and (allow_bad_triples or all(self.__is_valid_triple(t) for t in triples)):
                if allow_bad_triples:
                    return triples
                else: # Clean them
                    return list(map(self.__clean_triple, triples))
            else:
                return []
        except Exception as e:
            return []

    def get_entities(self, text: str):
        """Returns a list of relevant entities contained in the text. Filtering
        for the most relevant entities will be performed internally"""
        system_prompt, user_prompt, entities = self.__get_unfiltered_list_of_entities(text)
        if len(entities) == 0:
            system_prompt, user_prompt, entities = self.__get_unfiltered_list_of_entities(text)  # Try again
            if len(entities) == 0:
                return []  # Give up


        filtered_entities = self.__filter_entities(system_prompt, user_prompt, entities)
        if len(filtered_entities) == 0:
            filtered_entities = self.__filter_entities(system_prompt, user_prompt, entities)  # Try again
            if len(filtered_entities) == 0:
                return entities  # Give up, but at least return the unfiltered ones

        return filtered_entities

    def __is_valid_triple(self, t):
        return isinstance(t, tuple) and len(t) == 3

    def __clean_triple(self, t):
        s, p, o = t
        if not isinstance(s, str):
            s = str(s)
        if not isinstance(p, str):
            p = str(p)
        if not isinstance(o, str):
            o = str(o)

        return (s, p, o)

    def allows_system_prompt(self):
        return self.chosen_model_name not in [self.GEMMA_2B, self.GEMMA_9B]

    def get_formatted_prompt(self,
                               system_prompt: str,
                               user_prompt: str):
        if not self.allows_system_prompt():
            messages = [
                {"role": "user", "content": system_prompt+"\n"+user_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        return messages

    def get_formatted_twoturn_prompt(self,
                                       system_prompt_t0: str,
                                       user_prompt_t0: str,
                                       response_t0: str,
                                       system_prompt_t1: str,
                                       user_prompt_t1: str):
        if not self.allows_system_prompt():
            messages = [
                {"role": "user", "content": system_prompt_t0+"\n"+user_prompt_t0},
                {"role": "assistant", "content": response_t0},
                {"role": "user", "content": system_prompt_t1+"\n"+user_prompt_t1},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt_t0},
                {"role": "user", "content": user_prompt_t0},
                {"role": "assistant", "content": response_t0},
                {"role": "system", "content": system_prompt_t1},
                {"role": "user", "content": user_prompt_t1},
            ]

        return messages


    def __extract_list_from_text(self, generated_text: str):
        """ Given a LLM response, extract everything between the
        first '[' and the last ']' characters found (included).
        If they are not found, returns an empty string.

        It will also attempt to extract an enumeration of triples"""

        start_index = generated_text.find('[')
        end_index = generated_text.rfind(']')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            result = generated_text[start_index:end_index + 1]
            return result
        else:
            triples = re.findall(r"\('([^']*)', '([^']*)', '([^']*)'\)", generated_text)

            return [(s, p, o) for s, p, o in triples]

    def __filter_entities(self,
                          system_prompt_t0: str,
                          user_prompt_t0: str,
                          entities: str):
        """Filters out irrelevant entities by performing a multi-turn
         conversation based on the previously retrieved entities and prompt"""
        system_prompt_t1 = """From those entities you provided, output the list in the same format 
        but only containing the most relevant ones,  such as not commonly known entities or very technical terms."""
        user_prompt_t1 = ""

        prompt = self.get_formatted_twoturn_prompt(system_prompt_t0,
                                                     user_prompt_t0,
                                                     entities,
                                                     system_prompt_t1,
                                                     user_prompt_t1)

        raw_response = get_text_response(prompt)

        filtered_entities = self.__extract_list_from_text(raw_response)

        try:
            ents = eval(textwrap.dedent(filtered_entities))
            # The LLM can now hallucinate by including entities from the examples into the filtered list,
            # so we add a simple length check
            if isinstance(ents, list) and all(isinstance(ent, str) for ent in ents) and len(ents) <= len(entities):
                return ents
            else:
                return []

        except Exception as e:
            return []

    def __get_unfiltered_list_of_entities(self, text: str):
        """
        Returns a list of entities found in the text, alongside the system and user prompts if you want to filter
        it (via `filter_entities(system_prompt, user_prompt, entities)`)
        """
        system_prompt = f"""
        Output a Python-formatted list of strings containing the most relevant entities found in last provided text.
        If an entity has an acronym, it should be included between parentheses.
        You should only include technical and complex entities, excluding those that refer to too broad or widely known terms.

        Text: Chronic Obstructive Pulmonary Disease (COPD) is a disabling respiratory pathology. We want to detect the emphysema subtype of its clinical phenotypes.
        Entities: ["Chronic Obstructive Pulmonary Disease (COPD)", "emphysema", "clinical phenotypes"]"""

        user_prompt = f"""
        Text: {text}
        Entities:"""

        raw_response = get_text_response(self.get_formatted_prompt(system_prompt, user_prompt))

        try:
            entities = eval(textwrap.dedent(raw_response))
            if isinstance(entities, list) and all(isinstance(ent, str) for ent in entities):
                return system_prompt, user_prompt, entities
            else:
                return system_prompt, user_prompt, []

        except Exception as e:
            return system_prompt, user_prompt, []

