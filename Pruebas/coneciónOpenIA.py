import openai
import os
from openai import OpenAI




# Configura tu clave API directamente
client = OpenAI(api_key=openai_api_key)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Hola, ¿cómo estás?"}
    ]
)

print(completion.choices[0].message.content)