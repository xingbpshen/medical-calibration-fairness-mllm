from abc import ABC, abstractmethod
from typing import List


class BasePrompt(ABC):
    """
    Abstract base class for all prompt templates.
    All prompt templates should inherit from this class.
    """

    @abstractmethod
    def build(self, **kwargs) -> List[dict]:
        """
        Format the prompt with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the prompt.

        Returns:
            List[dict]: A list of formatted prompt messages.
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the prompt template.

        Returns:
            bool: True if the prompt is valid, False otherwise.
        """
        pass

    @abstractmethod
    def parse_response(self, response: any) -> dict:
        """
        Parse the response from the model.

        Args:
            response (any): The response from the model.

        Returns:
            dict: Parsed response data.
        """
        pass
