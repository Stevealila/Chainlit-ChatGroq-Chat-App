from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import chainlit as cl
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, List, Tuple
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

@cl.on_chat_start
async def on_chat_start():
    model = ChatGroq(model='gemma2-9b-it', streaming=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who provides in-depth, well-thought out answers. If it is code, you give well-explained examples. Always maintain context from the chat history."),
        ("human", "{chat_history}"),
        ("human", "{question}")
    ])

    runnable = prompt | model | StrOutputParser()
    cl.user_session.set('runnable', runnable)
    cl.user_session.set('chat_history', [])

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get('runnable'))
    chat_history: List[Tuple[str, str]] = cl.user_session.get('chat_history')

    msg = cl.Message(content="")
    await msg.send()

    chat_history_str = format_chat_history(chat_history)
    
    async for chunk in runnable.astream(
        {
            "question": message.content,
            "chat_history": chat_history_str
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ): 
        await msg.stream_token(chunk)

    await msg.update()

    # Append the new interaction to the chat history
    chat_history.append((message.content, msg.content))
    
    # Keep only the last 5 interactions to manage context length
    chat_history = chat_history[-5:]
    
    cl.user_session.set('chat_history', chat_history)

def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    formatted_history = ""
    for human, ai in chat_history:
        formatted_history += f"Human: {human}\nAI: {ai}\n\n"
    return formatted_history.strip()

if __name__ == "__main__":
    cl.run()