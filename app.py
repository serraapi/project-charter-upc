import chainlit as cl
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Carrega el system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Client Azure OpenAI — GPT-5-mini (Sweden Central)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-04-01-preview",
    azure_endpoint="https://project-charter-gpt5-resource.services.ai.azure.com"
)

MODEL = "gpt-5-mini"


@cl.on_chat_start
async def start():
    cl.user_session.set("historial", [])


@cl.on_message
async def main(message: cl.Message):
    historial = cl.user_session.get("historial")
    historial.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    contingut = ""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *historial
        ],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            contingut += token
            await msg.stream_token(token)

    await msg.update()

    historial.append({"role": "assistant", "content": contingut})
    cl.user_session.set("historial", historial)
