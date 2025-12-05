import sys
sys.path.insert(0, '..')  # or the path to your project root
from training.Inference_Wrapper_Class import SuperModelWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
from typing import Callable, Dict
import torch
import gc

class HFModel(SuperModelWrapper):
    def __init__(self):
        self._tokenizer = None
        self._model = None
        def prompt(ex, labels):
            out = f"Classify the sentiment as {', '.join(list(labels.values())[:-1])}, or {list(labels.values())[-1]}.\n\nText: {ex}\nSentiment:"
            return out
        self._prompt = prompt
        self._labels = None
        self._reversed_labels = None
        # self._train_df = None

    def set_labels(self, labels: Dict[int, str]):
        """Provided a dictionary of labels it will se the labels. The keys are the integer labels in the dataset and the values of the dictionary are the labels for the prompt into the models.

        Args:
            labels (Dict[int, str]): The labels to be saved

        Raises:
            ValueError: A dictionary must be provided as input otherwise an error will be risen.
            ValueError: If not all the keys are integers it will cause issues.
            ValueError: If not all the values are strings it will raise an error.
        """# NOTE: May want to change this so that the string label representations are the keys and the values are the integer labels. Or as an array, where the index is the integer label and the value is the string label.
        # if self._train_df is None or self._test_df is None:
        #     raise ValueError("The train and test dataframes have not be set yet. You must set to ensure that each of the labels in the dataframe have been set.")
        if not isinstance(labels, dict):
            raise ValueError("Labels must be a dictionary")
        if not all(isinstance(k, int) for k in labels.keys()):
            raise ValueError("Label keys must be integers")
        if not all(isinstance(v, str) for v in labels.values()):
            raise ValueError("Label values must be strings")
        label_keys = set(labels.keys())
        self._labels = labels
        self._reversed_labels = {v: k for k, v in self._labels.items()}

    def load_model(self, path: str):
        """
        Loads the model and tokenizer from the specified path url on hugging face.

        Args:
            path (str): The path to the model directory or the Hugging Face model ID.
        """
        if not isinstance(path, str):
            raise ValueError("A model name must be provided as a string")
        if self._model is not None or self._tokenizer is not None:
            print(f"Unloading current model and tokenizer from device {self._model.device}")
            # Unload the current model and tokenizer before loading a new one
            del self._tokenizer
            # Ensure the model is moved to CPU before deleting to free GPU memory
            self._model.cpu()
            del self._model
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        self._tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            # quantization_config=bnb_config,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"Model loaded from {path} on device {self._model.device}")
    
    def set_prompt(self, prompt_func: Callable[[str, dict], str]):
        """ Claude Sonnet 4.5
        Set the prompt function for generating prompts for the model.
        Args:
            prompt_func (Callable[[str, dict], str]): A callable that takes a string to label 
                and label options as a dictionary, and returns a formatted prompt string.
                The function signature should be: f(text: str, labels: dict) -> str
        Returns:
            None
        Example:
            >>> def my_prompt(text, labels):
            ...     return f"Classify the following texts with {labels}.\nText: {text}"
            >>> model.set_prompt(my_prompt)
        """

        # Prompt function takes in as such f(string to label, label options, example dataframe) -> prompt string
        self._prompt = prompt_func

    def predict(self, input_text):
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model and Tokenizer must be set")
        if self._prompt is None:
            raise ValueError("Prompt must be set.")
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model and Tokenizer have not been set yet.")

        # Run through the model in inference mode
        with torch.inference_mode():
            prompt = self._prompt(input_text, self._labels )
            model_inputs = self._tokenizer(prompt, return_tensors="pt").to(
                self._model.device
            )
            # Input into the model and get the output
            model_outputs = self._model(**model_inputs)
            # Get the last token output
            next_token_logits = model_outputs.logits[:, -1, :]
            # Get the probabilities of the values
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)[0]
            # Iterate through the labels and get the probability of it
            label_probs = torch.zeros(max(self._labels.keys()) + 1)
            for label in self._labels.values():
                # For simplicity, use first token probability
                label_tokens = self._tokenizer.encode(f" {label}", add_special_tokens=False)
                token_id = label_tokens[0]
                prob = probs[token_id].item()
                label_probs[self._reversed_labels[label]] = prob
            # Normalize the probabilities of the values
            return label_probs / label_probs.sum()
    
    def predict_batch(self, batch_input):
        # Predict batch
        results = []
        for input_text in batch_input:
            results.append(self.predict(input_text))
        return torch.stack(results)