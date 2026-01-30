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

            return {
                "response": response
            }

        except Exception as e:
            logger.error(f"Error making OpenAI API request: {e}")
            raise

def example_usage():
    """Example of how to use the TokenUsageLogger."""
    # Initialize - it will automatically use OPENAI_API_KEY from environment
    token_logger = TokenUsageLogger()

    question = input("Enter your question: ")
    # Load the prompt template from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, "factual_answer_v1.txt")
    with open(prompt_file_path, "r") as f:
        system_prompt = f.read().format(user_question=question)

    # Example messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Make request and log tokens
    result = token_logger.make_request_and_log_tokens(messages)

    # Access the response
    print(result["response"].choices[0].message.content)

if __name__ == "__main__":
    example_usage()
