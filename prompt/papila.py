from prompt import BasePrompt
from typing import List
from util import my_number_to_words, parse_content


class PapilaPrompt(BasePrompt):
    """
    PAPILA prompt template for image-based question answering.
    This class inherits from BasePrompt and implements the required methods.
    """

    def __init__(self, get_image_url, parsing_namespace, exemplar_data_points: List[dict] = None):
        """
        Initialize the PAPILA prompt template.
        This constructor can be extended to include any necessary initialization logic.
        """
        super().__init__()
        # Additional initialization
        self.get_image_url = get_image_url  # Function to get image URL
        self.exemplar_data_points = exemplar_data_points
        self.answer_options = parsing_namespace.answer_options    # e.g., ["negative", "positive"]
        self.attributes_namespace = parsing_namespace.attributes
        self.attributes_list = list(vars(self.attributes_namespace).keys())

        reversed_answer_options_in_text = " or ".join(option.capitalize() for option in reversed(self.answer_options))
        self.developer_content = [
            {
                'type': 'text',
                'text': f'You are a helpful multimodal medical assistant that can analyze the image. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question choose from {reversed_answer_options_in_text}.'
            }
        ]

    def get_attributes_value_in_text(self, data_point: dict, attributes_list: List[str]) -> str:
        """
        Get the attributes' values in text format.

        Args:
            data_point (dict): The data point containing attributes.
            attributes_list (List[str]): List of attributes to extract values from.

        Returns:
            str: A string representation of the attributes' values.
        """
        attributes_value_in_text = []
        for attribute in attributes_list:
            if attribute in data_point.keys():
                value_idx = data_point[attribute]
                value_list = getattr(self.attributes_namespace, attribute)
                attributes_value_in_text.append(value_list[value_idx])
        return ' '.join(attributes_value_in_text)

    def build_exemplars_content(self) -> List[dict]:
        """
        Build the exemplar prompts for the PAPILA template.

        Returns:
            List[dict]: A list of formatted exemplar messages.
        """
        if not self.exemplar_data_points:
            return []

        exemplars_content = []
        cnt = 1  # Counter for numbering the exemplars
        for data_point in self.exemplar_data_points:
            # get attributes' value in text, e.g., "age <60" and "female" -> "age <60 female"
            attributes_value_in_text = self.get_attributes_value_in_text(data_point, self.attributes_list)
            image_url = self.get_image_url(data_point['image_path'])
            answer_idx = data_point['gt_answer']
            text = f"Does the {my_number_to_words(cnt)} fundus image of {attributes_value_in_text} show sign of Glaucoma? Answer: {str(self.answer_options[answer_idx]).capitalize()}."
            exemplars_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
            exemplars_content.append({'type': 'text', 'text': text})
            cnt += 1

        return exemplars_content

    def build(self, data_point: dict, marginalize_out: List[str], **kwargs) -> List[dict]:
        """
        Format the prompt with the provided keyword arguments.

        Args:
            data_point (dict): The data point containing the image path and other attributes.
            marginalize_out (List[str]): List of assets to be marginalized out. e.g., ["image", "b_age"]
            **kwargs: Keyword arguments to format the prompt.

        Returns:
            List[dict]: A list of formatted prompt messages.
        """
        # some misc
        number_in_word = my_number_to_words(len(self.exemplar_data_points) + 1)
        exemplars_content = self.build_exemplars_content()

        # logic for marginalization
        remaining_attributes = [attr for attr in self.attributes_list if attr not in marginalize_out]
        remaining_attributes_value_in_text = self.get_attributes_value_in_text(data_point, remaining_attributes)
        if remaining_attributes_value_in_text != '':
            remaining_attributes_value_in_text = 'of ' + remaining_attributes_value_in_text
        if 'image' in marginalize_out:
            text = f"Does an arbitrary fundus image {remaining_attributes_value_in_text} show sign of Glaucoma? Answer:"
            user_content = exemplars_content + [{'type': 'text', 'text': text}]
        else:
            image_url = self.get_image_url(data_point['image_path'])
            text = f"Does the {number_in_word} fundus image {remaining_attributes_value_in_text} show sign of Glaucoma? Answer:"
            user_content = exemplars_content + [{'type': 'image_url', 'image_url': {'url': image_url}},
                                                {'type': 'text', 'text': text}]

        messages = [
            {
                'role': 'developer',
                'content': self.developer_content
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]
        return messages

    def validate(self) -> bool:
        """
        Validate the prompt template.

        Returns:
            bool: True if the prompt is valid, False otherwise.
        """
        # Implementation of validation logic goes here
        pass

    def parse_response(self, completion: any) -> (int, dict):
        """
        Parse the response from the model.

        Args:
            completion (any): The completion from the model.

        Returns:
            dict: Parsed response data.
        """
        # use conventional parsing, returns top prob answer option index and {answer option indexes: probs}
        return parse_content(top_logprobs=completion.choices[0].logprobs.content[0].top_logprobs,
                             options=self.answer_options)
