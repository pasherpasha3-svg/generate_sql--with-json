import streamlit as st
import os, json
from dotenv import load_dotenv
from function import SQLAssistantEngine 

st.set_page_config(page_title="AI SQL Assistant", page_icon="🤖", layout="centered")


@st.cache_resource
def init_engine():
    load_dotenv()
    engine = SQLAssistantEngine()
    engine.load_memory()
    engine.fetch_db_schema() # سحب السكيما عند البداية
    return engine

assistant = init_engine()

# --- واجهة المستخدم (UI) ---
st.title("🤖 SQL Assistant")
st.markdown("aske your question and I generate sql statment")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "SELECT" in message["content"].upper():
            st.code(message["content"], language="sql")
        else:
            st.markdown(message["content"])

if user_query := st.chat_input("Write your question here..."):
    
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing and Generating SQL..."):
            
            cached_sql = assistant.search_memory(user_query)
            
            if cached_sql:
                response = cached_sql
                st.caption("✨ Found in Memory")
            else:
                if assistant.check_relevance(user_query):
                    response = assistant.generate_sql(user_query)
                    
                    if "SELECT" in response.upper():
                        assistant.save_memory(user_query, response)
                else:
                    response = "This question is not related to the database schema."

            if "SELECT" in response.upper():
                st.code(response, language="sql")
            else:
                st.info(response)
            
            # حفظ الرد في تاريخ المحادثة
            st.session_state.chat_history.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat Window"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("Database Schema")
    with st.expander("Show Connected Tables"):
        st.text(assistant.full_schema)