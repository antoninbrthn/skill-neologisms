"""Load and run models via AzureOpenAI."""

from typing import Union
from openai import AzureOpenAI
import os
import asyncio
import aiohttp
import time

MODEL_TO_NAME = {
    "gpt-4o": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o"),
    "gpt-4o-mini": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI", "gpt-4o-mini"),
    "gpt-5": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5", "gpt-5"),
}

API_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5"]


def load_api_model(model_config):
    model_name = MODEL_TO_NAME[model_config.name]
    model = OpenAIModel(model_name=model_name, **model_config)
    return model


class OpenAIModel:
    def __init__(self, model_name, max_tokens=1024, use_batch=True, **kwargs):
        """
        Initializes the model with an AzureOpenAIClient object.

        Args:
            model_name (str): The name of the model.
            max_tokens (int): Maximum tokens for the model's response.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.use_batch = use_batch
        self.client = AzureOpenAIClient(self.model_name)
        self.total_price = 0

    def _generate(self, *, input_text: Union[str, list] = None, messages=None) -> list:
        """Generate a response using the model.
        Can take either a string (single prompt) or a list of strings (multiple prompts) as input,
        but will always return a list of responses.
        """
        if type(input_text) == str:  # single sample
            input_text = [input_text]
        assert messages is not None or input_text is not None, "Either messages or input_text must be provided for batch processing"

        # Process input_text into messages format if provided
        if messages is None:
            messages = [[{"role": "user", "content": text}] for text in input_text]

        if self.use_batch:  # batch processing
            print(f"Using batch to generate {len(messages)} inputs")
            try:
                outputs = asyncio.run(self._batch_generate(messages_list=messages))
            except KeyError as e:
                print("WARNING: Probably exceeded OpenAI API rate limit. Retrying in 60 seconds..")
                print("error", e)

                time.sleep(60)
                return self._generate(input_text=input_text, messages=messages)
            return outputs
        else:  # process one by one
            output = []
            # different max_tokens arg name for gpt-5
            args = {"max_completion_tokens": self.max_tokens} if "gpt-5" in self.model_name else {"max_tokens": self.max_tokens}
            for message in messages:
                completion = self.client.get_completion(messages=message, **args)
                self.update_price(completion)
                output.append(completion.choices[0].message.content)
            return output[0] if len(output) == 1 else output

    def update_price(self, completion):
        price = get_price(completion)
        self.total_price += price
        if self.total_price > 10:
            print(f"Total price: {self.total_price}")
        if self.total_price > 100:
            print("WARNING: Price exceeds 100 dollars. " * 100)

    # async calls
    async def _generate_async(self, input_text=None, messages=None):
        """Async version of _generate using AzureOpenAIClient."""
        assert messages is not None or input_text is not None, "Either messages or input_text must be provided for batch processing"
        assert (input_text is None) != (messages is None), "Provide either messages_list or input_list, not both"

        if messages is None:
            messages = [{"role": "user", "content": input_text}]

        try:
            # max_completion_tokens for gpt-5, otherwise max_tokens
            args = {"max_completion_tokens": self.max_tokens} if "gpt-5" in self.model_name else {"max_tokens": self.max_tokens}
            if "gpt-5" in self.model_name:
                args["reasoning_effort"] = "minimal"
            completion = await self.client.get_completion_async(messages=messages, args=args)
            return completion
        except Exception as e:
            print("Error during API call:", e)
            return f"Error: {e}"

    async def _batch_generate(self, input_list=None, messages_list=None):
        """Executes multiple requests in parallel using asyncio."""
        assert messages_list is not None or input_list is not None, "Either messages_list or input_list must be provided for batch processing"
        assert (messages_list is None) != (input_list is None), "Provide either messages_list or input_list, not both"

        if input_list is not None:
            tasks = [self._generate_async(input_text=text) for text in input_list]
        else:
            tasks = [self._generate_async(messages=messages) for messages in messages_list]
        results = await asyncio.gather(*tasks)
        output = []
        for completion in results:
            self.update_price(completion)
            output.append(completion["choices"][0]["message"]["content"])
        return output[0] if len(output) == 1 else output


class AzureOpenAIClient:
    def __init__(self, model="gpt-4o-mini"):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = model

    def get_completion(self, messages, max_tokens=1000, args={}, **kwargs):
        """Synchronous API call for backward compatibility."""
        # if 'gpt-5' not in self.model:
        if "max_completion_tokens" not in kwargs.keys() and "max_completion_tokens" not in args.keys():
            args["max_tokens"] = max_tokens
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=123,
            **args,
            **kwargs,
        )

    async def get_completion_async(self, messages, args={"max_tokens": 1000}):
        """Asynchronous API call to support parallel processing."""
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01"
        headers = {
            "Authorization": f"Bearer {os.getenv('AZURE_OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages, **args}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                result = await response.json()
                return result


### PRICING
# from https://openai.com/api/pricing/
openai_api_rates_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},
    "gpt-5": {"input": 1.50, "output": 10.00},
    "gpt-5-2025-08-07": {"input": 1.50, "output": 10.00},
}


def get_price(completion):
    completion = to_dict(completion)
    try:
        model = completion["model"]
        usage = completion["usage"]
    except KeyError as e:
        print("Failed to get model and usage from completion object.")
        print(f"KeyError: {e}")
        print("Completion object:", completion)
        print("Returning 0 as price.")
        return 0
    input_price = openai_api_rates_1M[model]["input"] * usage["prompt_tokens"] / 1_000_000
    output_price = openai_api_rates_1M[model]["output"] * usage["completion_tokens"] / 1_000_000
    return input_price + output_price


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return to_dict(vars(obj))
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(v) for v in obj)
    return obj
