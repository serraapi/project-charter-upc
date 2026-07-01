import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from openai import AzureOpenAI
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-04-01-preview",
    azure_endpoint="https://project-charter-gpt5-resource.services.ai.azure.com"
)

MODEL = "gpt-5.4-mini"   # canvia aquí per provar: "gpt-5-mini" / "gpt-5.4-mini"
EFFORT = "medium"        # canvia aquí per provar: "low" / "medium" / "high"

# ────── Eina que el model pot decidir invocar ──────
TOOLS = [{
    "type": "function",
    "function": {
        "name": "iniciar_questionari_projecte",
        "description": (
            "Llança un qüestionari interactiu de botons per conèixer el projecte de "
            "l'estudiant. Invoca-la NOMÉS quan l'estudiant expressi clarament que vol "
            "començar a redactar o muntar la Carta del SEU projecte (ex: 'va, comencem', "
            "'vull fer la meva carta', 'tinc un projecte i vull redactar-lo'). NO la "
            "invoquis si només fa una pregunta teòrica o un dubte concret, ni si el "
            "qüestionari ja s'ha fet abans en aquesta conversa."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}]


# ────── Persistència ──────
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))


@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: dict,
                   default_user: cl.User) -> Optional[cl.User]:
    email = raw_user_data.get("email", "")
    if email.endswith("@estudiantat.upc.edu") or email.endswith("@upc.edu"):
        return default_user
    return None


# ────── Desa el context del projecte als TAGS del thread (invisible a l'usuari) ──────
async def desar_context_thread(context_projecte: str):
    thread_id = cl.context.session.thread_id
    if not thread_id:
        return
    dl = cl_data.get_data_layer()
    if dl:
        try:
            await dl.update_thread(thread_id, tags=[f"ctx::{context_projecte}"])
        except Exception as e:
            print(f"[avis] no s'ha pogut desar el context al thread: {e}")


# ────── L'arbre de preguntes ramificat ──────
async def executar_arbre() -> Optional[str]:
    parts = []

    r1 = await cl.AskActionMessage(
        content="Genial! Per situar-me, quin tipus de projecte plantejes?",
        actions=[
            cl.Action(name="prod", payload={"v": "Producte/Prototip"}, label="⚙️ Producte o prototip"),
            cl.Action(name="rec", payload={"v": "Recerca/R+D"}, label="🔬 Recerca / R+D"),
            cl.Action(name="mill", payload={"v": "Millora/Optimització"}, label="📈 Millora o optimització"),
        ],
        timeout=300,
    ).send()
    if not r1:
        return None
    tipus = r1["payload"]["v"]
    parts.append(f"Naturalesa: {tipus}")

    if tipus == "Producte/Prototip":
        opcions2 = [
            cl.Action(name="hw", payload={"v": "Hardware/físic"}, label="⚡ Objecte físic / hardware"),
            cl.Action(name="sw", payload={"v": "Programari"}, label="💻 Programari / digital"),
            cl.Action(name="mix", payload={"v": "Mixt (hw+sw)"}, label="🔀 Sistema mixt"),
        ]
        preg2 = "I el que vols desenvolupar, què és principalment?"
    elif tipus == "Recerca/R+D":
        opcions2 = [
            cl.Action(name="exp", payload={"v": "Experimental"}, label="🧪 Experimental"),
            cl.Action(name="teo", payload={"v": "Teòrica/disseny"}, label="📐 Anàlisi o disseny teòric"),
        ]
        preg2 = "Quin tipus de recerca és?"
    else:
        opcions2 = [
            cl.Action(name="cost", payload={"v": "Costos/eficiència"}, label="📉 Costos / eficiència"),
            cl.Action(name="qual", payload={"v": "Prestacions/qualitat"}, label="📈 Prestacions / qualitat"),
            cl.Action(name="sost", payload={"v": "Sostenibilitat"}, label="🌱 Sostenibilitat / impacte"),
        ]
        preg2 = "Què vols millorar principalment?"

    r2 = await cl.AskActionMessage(content=preg2, actions=opcions2, timeout=300).send()
    if r2:
        parts.append(f"Detall: {r2['payload']['v']}")

    r3 = await cl.AskActionMessage(
        content="I per acabar: quin és el principal condicionant del projecte?",
        actions=[
            cl.Action(name="temps", payload={"v": "Temps/terminis"}, label="⏱️ Temps"),
            cl.Action(name="diner", payload={"v": "Pressupost"}, label="💰 Pressupost"),
            cl.Action(name="norma", payload={"v": "Normativa/especificacions"}, label="📋 Normativa"),
            cl.Action(name="nose", payload={"v": "Per determinar"}, label="❓ Encara no ho sé"),
        ],
        timeout=300,
    ).send()
    if r3:
        parts.append(f"Condicionant: {r3['payload']['v']}")

    return " | ".join(parts)


def construir_system(context_projecte: Optional[str]) -> str:
    s = system_prompt
    if context_projecte:
        s += (f"\n\n# CONTEXT DEL PROJECTE DE L'ESTUDIANT\n{context_projecte}\n"
              "Adapta tota la teva guia a aquest perfil concret.")
    return s


async def respondre_streaming(historial, context_projecte):
    """Crida normal amb streaming i persisteix la resposta."""
    msg = cl.Message(content="")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": construir_system(context_projecte)}, *historial],
        stream=True,
        reasoning_effort=EFFORT,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            await msg.stream_token(token)
    await msg.send()
    historial.append({"role": "assistant", "content": msg.content})


@cl.on_chat_start
async def start():
    # Sense missatge de benvinguda: la conversa la inicia l'usuari
    cl.user_session.set("historial", [])
    cl.user_session.set("context_projecte", None)
    cl.user_session.set("arbre_fet", False)


@cl.on_chat_resume
async def resume(thread: ThreadDict):
    historial = []
    context_projecte = None
    arbre_fet = False

    # Recupera el context del projecte des dels TAGS del thread
    for tag in (thread.get("tags") or []):
        if tag.startswith("ctx::"):
            context_projecte = tag.replace("ctx::", "", 1)
            arbre_fet = True

    for step in thread["steps"]:
        out = step.get("output", "") or ""
        if step["type"] == "user_message":
            historial.append({"role": "user", "content": out})
        elif step["type"] == "assistant_message":
            historial.append({"role": "assistant", "content": out})

    cl.user_session.set("historial", historial)
    cl.user_session.set("context_projecte", context_projecte)
    cl.user_session.set("arbre_fet", arbre_fet)


@cl.on_message
async def main(message: cl.Message):
    historial = cl.user_session.get("historial")
    context_projecte = cl.user_session.get("context_projecte")
    arbre_fet = cl.user_session.get("arbre_fet")
    historial.append({"role": "user", "content": message.content})

    # Pas 1: el model decideix si cal llançar el qüestionari (només si no s'ha fet ja)
    vol_questionari = False
    if not arbre_fet:
        decisio = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": construir_system(context_projecte)}, *historial],
            tools=TOOLS,
            tool_choice="auto",
            reasoning_effort="low",
        )
        tool_calls = decisio.choices[0].message.tool_calls
        if tool_calls and tool_calls[0].function.name == "iniciar_questionari_projecte":
            vol_questionari = True

    # Pas 2a: si toca, llança l'arbre i genera la transició amb la IA
    if vol_questionari:
        context_projecte = await executar_arbre()
        if context_projecte:
            cl.user_session.set("context_projecte", context_projecte)
            cl.user_session.set("arbre_fet", True)
            # Desa el context de forma INVISIBLE als tags del thread
            await desar_context_thread(context_projecte)
            # La IA genera la transició natural cap a l'Objecte
            historial.append({
                "role": "user",
                "content": f"[Sistema: l'estudiant ha completat el qüestionari. Perfil del projecte: "
                           f"{context_projecte}. Saluda breument que ja et situes i convida'l a "
                           f"començar per l'Objecte explicant què vol fer el projecte. To natural i variat.]"
            })
            await respondre_streaming(historial, context_projecte)
            return

    # Pas 2b: resposta normal en streaming
    await respondre_streaming(historial, context_projecte)