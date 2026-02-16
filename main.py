# import packages
import chainlit as cl #type:ignore
from langchain_core.prompts import ChatPromptTemplate #type:ignore
from langchain_groq import ChatGroq #type:ignore
from langchain_core.output_parsers import StrOutputParser #type:ignore
from langchain_core.runnables import Runnable, RunnableConfig #type:ignore
from typing import cast, List, Tuple
from dotenv import load_dotenv #type:ignore
import os

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant who gives straightforward, in-depth responses. If you are explaining code related stuff, you provide well-explained code examples. Use the given chat history: {chat_history}'),
        ('human', '{question}')
    ])

    llm = ChatGroq(model="qwen/qwen3-32b", reasoning_format="parsed")

    chain = prompt | llm | StrOutputParser()

    # add chain and chat history to the current session
    cl.user_session.set('runnable', chain)
    cl.user_session.set('chat_history', [])

@cl.on_message
async def on_message(message: cl.Message):
    # let chainlit types to expect
    runnable = cast(Runnable, cl.user_session.get('runnable'))
    chat_history: List[Tuple[str, str]] = cl.user_session.get('chat_history')

    # start conversation with empty message
    msg = cl.Message(content='')
    await msg.send()

    # strictly handle string of messages
    chat_history_str = format_chat_history(chat_history)

    # stream responses
    async for chunk in runnable.astream({
        'chat_history': chat_history_str,
        'question': message.content,
    }, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await msg.stream_token(chunk)
    
    # update states
    await msg.update()
    chat_history.append((msg.content, message.content))
    chat_history = chat_history[-10:] # remember only last 10 messages
    cl.user_session.set('chat_history', chat_history)


def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    formatted_history = ""
    for human, ai in chat_history:
        formatted_history += f"Human: {human}\nAI: {ai}\n\n"
    return formatted_history.strip()


if __name__ == '__main__':
    cl.run()
