import chainlit as cl
from openai import AzureOpenAI
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Carrega el system prompt des del fitxer
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Client Azure OpenAI — GPT-5-mini desplegat a Sweden Central
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-04-01-preview",
    azure_endpoint="https://project-charter-gpt5-resource.services.ai.azure.com"
)

DEPLOYMENT_NAME = "gpt-5-mini"


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    # Filtra només comptes UPC (@estudiantat.upc.edu i @upc.edu)
    email = raw_user_data.get("email", "")
    if email.endswith("@estudiantat.upc.edu") or email.endswith("@upc.edu"):
        return default_user
    return None  # Rebutja qualsevol altre compte


@cl.on_chat_start
async def start():
    cl.user_session.set("historial", [])


@cl.on_message
async def main(message: cl.Message):
    historial = cl.user_session.get("historial")
    historial.append({"role": "user", "content": message.content})

    # Missatge buit que anirem omplint amb streaming
    msg = cl.Message(content="")
    await msg.send()

    contingut = ""

    # Crida a Azure OpenAI amb streaming
    stream = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *historial
        ],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = delta.content
        if token:
            contingut += token
            await msg.stream_token(token)

    await msg.update()

    historial.append({"role": "assistant", "content": contingut})
    cl.user_session.set("historial", historial)