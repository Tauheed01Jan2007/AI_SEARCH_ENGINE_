import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import  WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv() 
import os

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 500)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 500)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)

search = DuckDuckGoSearchRun(name = "Search")

st.title("Langchain - Chat with search")
"""
In this example, we're  using 'StreamlitCallbackHandler' to display the throughts and actions
try more langchain streamlit Agent examples at [https://github.com/langchain-ai/streamlit-agent]
"""

## Sidebar for setting
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key", type = "password")

if "messages" not in st.session_state:
    st.session_state["messages"]= [
        {"role":"assistant", "content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

if prompt := st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key = api_key, model_name = "openai/gpt-oss-20b", streaming=True)
    search_agent = create_agent(tools = [search, wiki, arxiv], model = llm)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts= False)
        response = search_agent.invoke({
                "messages" : [{"role": "user", "content": prompt}]} ,
                config = {"callbacks" :[st_cb]}   
        )
        answer = response["messages"][-1].content
        st.session_state.messages.append({"role":'assistant', "content":answer})
        st.write(answer)
        