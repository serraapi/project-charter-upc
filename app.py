import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from openai import AzureOpenAI
import os
from typing import Optional
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


# ────── Persistència: Supabase (PostgreSQL) ──────
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))


# ────── Autenticació OAuth: només comptes UPC ──────
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    email = raw_user_data.get("email", "")
    if email.endswith("@estudiantat.upc.edu") or email.endswith("@upc.edu"):
        return default_user
    return None  # Rebutja qualsevol altre compte


@cl.on_chat_start
async def start():
    cl.user_session.set("historial", [])


# ────── Reprendre una conversa guardada ──────
@cl.on_chat_resume
async def resume(thread: ThreadDict):
    historial = []
    for step in thread["steps"]:
        if step["type"] == "user_message":
            historial.append({"role": "user", "content": step["output"]})
        elif step["type"] == "assistant_message":
            historial.append({"role": "assistant", "content": step["output"]})
    cl.user_session.set("historial", historial)


@cl.on_message
async def main(message: cl.Message):
    historial = cl.user_session.get("historial")
    historial.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

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
            await msg.stream_token(token)

    # send() al final tanca l'stream i persisteix el missatge al data layer
    await msg.send()

    historial.append({"role": "assistant", "content": msg.content})