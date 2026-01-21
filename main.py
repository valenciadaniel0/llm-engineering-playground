import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def chat_completion(messages, model="gpt-3.5-turbo"):
    """
    Send a chat completion request to OpenAI API

    Args:
        messages (list): List of message objects with 'role' and 'content'
        model (str): The model to use for completion

    Returns:
        str: The AI response content
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0
        )
        # Extract the response content and usage information
        content = response.choices[0].message.content
        usage = response.usage

        # Calculate approximate cost (prices as of 2024)
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        if model == "gpt-3.5-turbo":
            # $0.0015 per 1K input tokens, $0.002 per 1K output tokens
            cost = (input_tokens * 0.0015 / 1000) + (output_tokens * 0.002 / 1000)
        elif model.startswith("gpt-4"):
            # $0.03 per 1K input tokens, $0.06 per 1K output tokens (gpt-4)
            cost = (input_tokens * 0.03 / 1000) + (output_tokens * 0.06 / 1000)
        else:
            cost = 0  # Unknown model

        print(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {usage.total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

        return content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Explain in one paragraph, what Saatva is"}
    ]

    response = chat_completion(messages)
    if response:
        print(f"AI Response: {response}")
