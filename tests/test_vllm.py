import openai
from openai import OpenAI
import os

api_key = "EMPTY"
base_url = "http://localhost:8000/v1" 

# instantiate client
client = OpenAI(base_url=base_url,
                api_key=api_key)

def chat_with_server(prompt):
    """
    Sends a prompt to the OpenAI Chat API and returns the response.
    """
    try:
        completion = client.chat.completions.create(
            model="Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat_with_server(user_input)
        print("AI:", response)
