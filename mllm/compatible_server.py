from mllm import BaseMLLM
from openai import OpenAI, AzureOpenAI
from typing import List


class CompatibleServer(BaseMLLM):

    def __init__(self, args, config):
        """
        Initialize the CompatibleServer class.
        This constructor can be extended to include any necessary initialization logic.
        """
        super().__init__()
        # Additional initialization can be added here if needed
        self.client, self.model = self.load_model(args=args, config=config)

    def generate(self, prompt: List[dict], **kwargs) -> any:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (List[dict]): The input content to generate a response for.

        Returns:
            any: The generated response.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=kwargs.get('temperature'),
            stop=None,
            stream=False,
            n=kwargs.get('n'),
            max_completion_tokens=kwargs.get('max_completion_tokens'),
            logprobs=True,
            top_logprobs=kwargs.get('top_logprobs'))
        return completion

    def load_model(self, **kwargs):
        """
        Load the model

        Args:
            kwargs: Keyword arguments for model loading, such as model path or configuration.
        """
        args = kwargs.get('args')
        config = kwargs.get('config')
        if args.azure_openai:
            client = AzureOpenAI(azure_endpoint=config.azure_openai.endpoint_url,
                                 api_key=config.azure_openai.api_key,
                                 api_version=config.azure_openai.api_version)
            model = config.azure_openai.deployment_name
        elif args.local_mllm and config.local_mllm.use_vllm_serve:
            client = OpenAI(api_key='EMPTY',
                            base_url=f'http://{config.local_mllm.host}:{config.local_mllm.port}/v1',
                            timeout=30.0,
                            max_retries=0)
            models = client.models.list()
            model = models.data[0].id
        else:
            raise ValueError
        return client, model
