import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score


class SuperModelWrapper(Object):

    def load_model(self):    
        raise Exception("Don't call me, call my subclasses")

    def predict(self, input_text: List[str], batch_size: int = 16):
        raise Exception("Don't call me, call my subclasses")

    def export_results(self, input_text: List[str], predictions: List[int], output_path: str):
"""
Args:
    model_path (str): Path to stored model
"""
class ModelWrapper:
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None


    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def predict(self, texts, batch_size: int = 16):
        """
        Make predictions on a list of texts.
        Args:
            texts (list[str]): Input texts
            batch_size (int): Batch size for inference
        Returns:
            list: Predicted labels
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
        return all_preds


    def export_results(self, texts, predictions, output_path: str):
        """
        Export results to CSV.
        Args:
            texts (list[str]): Input texts
            predictions (list[int]): Model predictions
            output_path (str): Path to save CSV
        """
        df = pd.DataFrame({"text": texts, "prediction": predictions})
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

if __name__ == "__main__":
    wrapper = ModelWrapper(model_path="../models/distilbert_baseline_IMDB")
    wrapper.load_model()
    
    #test_texts = ["I love this movie!", "This is terrible."]

    test_df = pd.read_csv("../data/test.csv")
    test_texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()
    preds = wrapper.predict(test_texts)

    # check accuracy
    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy on test set: {acc:.4f}")


    
    
    wrapper.export_results(test_texts, preds, "predictions.csv")




        
