import numpy as np
import spacy

class PromptAnalyzer():
    """
    A class for analyzing prompts using NLP, based
    on the current framework configuration
    """
    #def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, prompt_model='en_core_web_sm'):
        #self.model_config = model_config
        #self.training_config = training_config
        #self.recognizable_parts = list(map(lambda t: t[0], model_config.segmentation_attrs[model_config.model_to_segmentation[training_config.model_name]]))
    
    def __init__(self, training_params, prompt_model='en_core_web_sm'):
        self.recognizable_parts = list(map(lambda t: t[0], training_params['segmentation_parts']))
        self.model = spacy.load(prompt_model) #SentenceTransformer(prompt_model)

    def __call__(self, prompt, expansion_factor = 10, print_analysis=True):
        """
        Analyze a prompt:
            - Find to which semantic of the segmentation model the prompt is related 
            - Retrieve the relevant noun phrase from the prompt
            - Expand the noun phrase `expansion_factor` times within the prompt (used for a better CLIP score)
        """
        #part_embeddings = self.model.encode(self.recognizable_parts, convert_to_tensor=True)
        #prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)

        #similarities = util.pytorch_cos_sim(part_embeddings, prompt_embedding)
        prompt_embedding = self.model(prompt)
        similarities = [prompt_embedding.similarity(self.model(part)) for part in self.recognizable_parts]
        self.part_idx = np.argmax(similarities)
        self.final_part = self.recognizable_parts[self.part_idx]


        #doc = self.model(prompt)

        part_embedding = self.model(self.final_part)
        # Iterate over words or chunks ?
        chunk_similarities = [self.model(chunk.text).similarity(part_embedding) for chunk in prompt_embedding.noun_chunks]
        best_chunk_idx = np.argmax(chunk_similarities)
        best_chunk = None

        new_prompt_chunks = []
        for idx, noun_chunk in enumerate(prompt_embedding.noun_chunks):
            new_prompt_chunks.append(noun_chunk.text)
            if (idx == best_chunk_idx):
                best_chunk = noun_chunk.text
                new_prompt_chunks.extend([noun_chunk.text] * (expansion_factor - 1))

        self.new_prompt = ' '.join(new_prompt_chunks)

        self.prompt_dir = best_chunk.lower().replace("'", " ").replace(" ", "_")

        #attrs = self.model_config.segmentation_attrs[self.model_config.model_to_segmentation[self.training_config.model_name]]
        #related_attrs = attrs[self.part_idx][1]
        #related_idxs = []
        #for idx, el in enumerate(attrs):
        #    name, related = el
        #    if name in related_attrs:
        #        related_idxs.append(idx)

        return self.part_idx, self.new_prompt, self.prompt_dir