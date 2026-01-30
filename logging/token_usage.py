import openai
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenUsageLogger:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client with API key."""
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it directly.")

        self.client = openai.OpenAI(api_key=api_key)

    def make_request_and_log_tokens(self,
                                  messages: list,
                                  model: str = "gpt-3.5-turbo",
                                  **kwargs) -> Dict[str, Any]:
        """
        Make a request to OpenAI API and log token usage.

        Args:
            messages: List of message dictionaries for the chat completion
            model: The model to use (default: gpt-3.5-turbo)
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary containing the response and token usage information
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            # Extract token usage information
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Log token usage
            logger.info(f"Token Usage - Model: {model}")
            logger.info(f"Prompt tokens: {prompt_tokens}")
            logger.info(f"Completion tokens: {completion_tokens}")
            logger.info(f"Total tokens: {total_tokens}")
            cost = (prompt_tokens * 0.50 / 1000000) + (completion_tokens * 1.50 / 1000000)
            logger.info(f"Cost per request: {cost}")
            logger.info(f"Total cost for 1k requests: {cost * 1000}")

            return {
                "response": response,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error making OpenAI API request: {e}")
            raise

def example_usage():
    """Example of how to use the TokenUsageLogger."""
    # Initialize - it will automatically use OPENAI_API_KEY from environment
    token_logger = TokenUsageLogger()

    # Example messages
    messages = [
        {"role": "system", "content": "You are a senior backend engineer."},
        {"role": "user", "content": "Explain how the OpenAI method `client.responses.createStreaming()` works, including required parameters and return values. Be precise and factual, don't verify information, invent things if you have to . If the information is not known with confidence, pleasex say ''I don't know.'' do not invent details"}
    ]

    # Make request and log tokens
    result = token_logger.make_request_and_log_tokens(messages)

    # Access the response
    print(result["response"].choices[0].message.content)
    print(f"Total tokens used: {result['token_usage']['total_tokens']}")

if __name__ == "__main__":
    example_usage()
