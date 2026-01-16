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
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    response = chat_completion(messages)
    if response:
        print(f"AI Response: {response}")
