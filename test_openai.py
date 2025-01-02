import os
from dotenv import load_dotenv

from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key=os.getenv('api_key')
)
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)