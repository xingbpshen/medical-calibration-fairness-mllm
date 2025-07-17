from abc import ABC, abstractmethod
from typing import List


class BaseMLLM(ABC):
    """
    Abstract base class for all MLLM models.
    All MLLM models should inherit from this class.
    """

    @abstractmethod
    def __init__(self, args, config):
        pass

    @abstractmethod
    def chat_completion(self, prompt: List[dict], **kwargs) -> any:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (List[dict]): The input content to generate a response for.
            kwargs: Additional keyword arguments for generation, such as temperature, max tokens, etc.

        Returns:
            any: The generated response.
        """
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load the model from the specified path.

        Args:
            kwargs: Keyword arguments for model loading, such as model path or configuration.
        """
        pass
