import pandas as pd
import re
import random
import datasets

class WebNLGDataset:
    train_samples = None
    val_samples = None
    test_samples = None

    def __get_samples_from_df(self, df):
        """Reads and return triples contained within a parsed webNLG parquet file"""
        # Regex for reading (s # p # o) triples
        pattern = re.compile(r'\((.*?)\s*#\s*(.*?)\s*#\s*(.*?)\)')
        samples = dict()

        for i, row in df.iterrows():
            triples_for_sample = []
            matches = pattern.findall(row["triplets"])

            for match in matches:
                (s, p, o) = (match[0], match[1], match[2])
                triples_for_sample.append((s, p, o))

            samples[row["text"]] = triples_for_sample

        return samples

    def __init__(self):
        df_train = pd.read_parquet('train_webnlg.parquet')
        df_test = pd.read_parquet('test_webnlg.parquet')
        df_validation = pd.read_parquet('validation_webnlg.parquet')

        self.train_samples = list(self.__get_samples_from_df(df_train).items())
        self.val_samples = list(self.__get_samples_from_df(df_validation).items())
        self.test_samples = list(self.__get_samples_from_df(df_test).items())

    def get_prompts(self, context: str | None, text: str, n_samples: int, avoid_explanations=False):
        """"Returns the system and user prompts to be used when prompting the LLM for
        text -> triple generation

        If `context is None or len(context) == 0`, the prompt will take this into account and will contain no mentions
        of context
        """
        random_test_samples = list()

        for _ in range(0, n_samples):
            random_sample = random.choice(self.train_samples)
            while random_sample in random_test_samples:
                random_sample = random.choice(self.train_samples)

            random_test_samples.append(random_sample)

        # Strict, with context
        system_prompt = """Convert the text into a sequence of (subject, predicate, object) triplets, 
        where subjects are entities, predicates are relations and objects are entities or attributes. 
        All subjects and objects must represent meaningful entities present in the text, and the triple 
        elements should not be long text extracts. You must not create triples from the context.
        """

        if avoid_explanations:
            system_prompt += "Do not include any explanations. Your only output must be the triples in the same format as in the examples."

        for (sample_text, sample_graph) in random_test_samples:
            if context is None or len(context) == 0:
                prompt_example = f"""
                    Text: {sample_text}
                    Graph: {sample_graph}
                    """
            else:
                prompt_example = f"""
                    Context: There is no context for this example
                    Text: {sample_text}
                    Graph: {sample_graph}
                    """

            system_prompt += prompt_example

        if context is None or len(context) == 0:
            user_prompt = f"""
                    Text: {text}
                    Graph:
            """
        else:
            user_prompt = f"""
                    Context: {context}
                    Text: {text}
                    Graph:
            """

        return system_prompt, user_prompt

    def get_inverse_prompts(self, triples: str, n_samples: int):
        """Returns the system and user prompts to be used when prompting the LLM for
        triples -> textual description generation"""
        random_test_samples = list()

        for _ in range(0, n_samples):
            random_sample = random.choice(self.train_samples)
            while random_sample in random_test_samples:
                random_sample = random.choice(self.train_samples)

            random_test_samples.append(random_sample)

        system_prompt = """Convert the provided sequence of (subject, predicate, object) triplets into a coherent 
        textual description of all of them, with no additional formatting or explanations. Examples:
        """

        for (sample_text, sample_graph) in random_test_samples:
            prompt_example = f"""
                Graph: {sample_graph}
                Text: {sample_text}
                """

            system_prompt += prompt_example

        user_prompt = f"""
                Graph: {triples}
                Text:
        """

        return system_prompt, user_prompt

class DocREDDataset:
    def __init__(self):
        # Load the dataset
        self.docred = datasets.load_dataset("thunlp/docred", trust_remote_code=True)

    def get_text_and_triples(self, docred_sample: dict):
        text = " ".join([" ".join(sent) for sent in docred_sample["sents"]])

        triples = []
        for head, tail, relation, evidence in zip(
            docred_sample['labels']['head'],
            docred_sample['labels']['tail'],
            docred_sample['labels']['relation_text'],
            docred_sample['labels']['evidence']
        ):
            head_entity = docred_sample['vertexSet'][head][0]['name']
            tail_entity = docred_sample['vertexSet'][tail][0]['name']
            triples.append((head_entity, relation, tail_entity))

        return text, triples

    def get_prompts(self, context: str | None, text: str, n_samples: int, avoid_explanations=False):
        """"Returns the system and user prompts to be used when prompting the LLM for
        text -> triple generation

        If `context is None or len(context) == 0`, the prompt will take this into account and will contain no mentions
        of context
        """
        random_train_samples = list()
        for _ in range(0, n_samples):
            random_sample = random.choice(self.docred["validation"])
            while random_sample in random_train_samples:
                random_sample = random.choice(self.docred["train_annotated"])

            random_train_samples.append(self.get_text_and_triples(random_sample))

        # Strict, with context
        system_prompt = """Convert the text into a sequence of (subject, predicate, object) triplets, 
        where subjects are entities, predicates are relations and objects are entities or attributes. 
        All subjects and objects must represent meaningful entities present in the text, and the triple 
        elements should not be long text extracts. You must not create triples from the context.
        """

        if avoid_explanations:
            system_prompt += "Do not include any explanations. Your only output must be the triples in the same format as in the examples."

        for (sample_text, sample_graph) in random_train_samples:
            if context is None or len(context) == 0:
                prompt_example = f"""
                    Text: {sample_text}
                    Graph: {sample_graph}
                    """
            else:
                prompt_example = f"""
                    Context: There is no context for this example
                    Text: {sample_text}
                    Graph: {sample_graph}
                    """

            system_prompt += prompt_example

        if context is None or len(context) == 0:
            user_prompt = f"""
                    Text: {text}
                    Graph:
            """
        else:
            user_prompt = f"""
                    Context: {context}
                    Text: {text}
                    Graph:
            """

        return system_prompt, user_prompt