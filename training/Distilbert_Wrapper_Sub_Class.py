import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from Inference_Wrapper_Class import SuperModelWrapper   # adjust import path if needed


class DistilbertWrapper(SuperModelWrapper):

    def __init__(self, device: str = None):
        """
        Wrapper for a DistilBERT sentiment model.
        Args:
            device (str, optional): 'cpu' or 'cuda'. Auto-selects if not provided.
        """
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    def load_model(self, path: str):
        """
        Loads a DistilBERT model + tokenizer from a directory created by:
            model.save_pretrained(...)
            tokenizer.save_pretrained(...)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def predict_batch(self, batch_input: List[str]) -> torch.Tensor:
        """
        Predicts probabilities for a batch of input texts.
        Returns:
            Tensor shape: (batch_size, num_classes)
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Call load_model(path) first.")

        # Tokenize
        encoded = self.tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1)

        return probs.cpu()

    # ------------------------------------------------------------------
    def predict(self, input_text: str) -> torch.Tensor:
        """
        Predicts probabilities for ONE piece of text.
        Returns Tensor shape: (num_classes,)
        """
        probs = self.predict_batch([input_text])
        return probs[0]


