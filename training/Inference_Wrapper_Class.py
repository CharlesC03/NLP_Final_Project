import torch
from typing import List


class SuperModelWrapper(object):
    """ This class should be a super class for all of the model classes so inference is easy.
    """
    def load_model(self, path: str):
        """
        Load a trained model from the specified path.
        This method must be implemented by subclasses and should not be called directly
        on the base class.
        Args:
            path (str): The file path to the saved model that should be loaded.
        Raises:
            Exception: Always raised when called on the base class, indicating that
                       this method should be implemented by subclasses.
        Note:
            This is an abstract method that serves as a template for subclass
            implementations. Each subclass should override this method with
            its own model loading logic.
        """
        raise Exception("Don't call me, call my subclasses")
    
    def predict_batch(self, batch_input: List[str]) -> torch.Tensor:
        """
        Predict labels for a batch of input texts.
        This method should be implemented by subclasses to perform batch inference
        on the provided input strings.
        Args:
            batch_input (List[str]): A list of input text strings to be processed
                and classified.
        Returns:
            torch.Tensor: A tensor containing the probability distribution of the classes for each of the inputs in the batch.
        Raises:
            Exception: This base class method raises an exception as it must be
                overridden by subclasses.
        """
        # Note maybe later we want to implement a method here where if not implemented it can default to the predicted class and just iterate through there.
        raise Exception("Don't call me, call my subclasses")

    def predict(self, input_text: str) -> torch.Tensor:
        """
        Predict the output for the given input text.
        This is an abstract method that must be implemented by subclasses.
        Raises an exception if called directly on the base class.
        Args:
            input_text (str): The input text to generate predictions for.
        Returns:
            torch.Tensor: The predicted output tensor as a probability distribution.
        Raises:
            Exception: Always raises an exception indicating this method should be 
                       implemented in subclasses rather than called directly.
        """

        raise Exception("Don't call me, call my subclasses")



        
