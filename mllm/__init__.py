from abc import ABC, abstractmethod


class BaseMLLM(ABC):
    """
    Abstract base class for all MLLM models.
    All MLLM models should inherit from this class.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated text response.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Load the model from the specified path.

        Args:
            model_path (str): The path to the model file.
        """
        pass