import streamlit as st
import pandas as pd
import json
import os
import uuid
from dotenv import load_dotenv
from function import SQLAssistantEngine

HISTORY_FILE = "full_chats_history.json"
MY_ACTIVE_TABLES = ["FinanceTransaction", "Biller","BillerAggregator","BillerCategory"] 

st.set_page_config(page_title="AI SQL Assistant", page_icon="🤖", layout="wide")

# --- وظائف الحفظ والتحميل ---
def save_all_chats():
    data_to_save = {}
    for chat_id, messages in st.session_state.all_chats.items():
        clean_messages = []
        for msg in messages:
            m = msg.copy()
            if "df" in m: del m["df"]  # لا نحفظ الجداول لتقليل الحجم
            clean_messages.append(m)
        data_to_save[chat_id] = clean_messages
        
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

def load_all_chats():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def init_engine():
    load_dotenv()
    engine = SQLAssistantEngine()
    engine.fetch_db_schema()
    engine.load_memory()
    return engine

assistant = init_engine()

# --- إدارة الحالة (Session State) ---
if "all_chats" not in st.session_state:
    saved_history = load_all_chats()
    if saved_history:
        st.session_state.all_chats = saved_history
        st.session_state.current_chat_id = list(saved_history.keys())[0]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.all_chats = {new_id: []}
        st.session_state.current_chat_id = new_id

# --- الـ Sidebar (يجب أن يكون خارج الـ if) ---
with st.sidebar:
    st.title("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.all_chats[new_id] = []
        st.session_state.current_chat_id = new_id
        save_all_chats()
        st.rerun()
    
    st.divider()
    # عرض قائمة المحادثات
    for chat_id in list(st.session_state.all_chats.keys()):
        messages = st.session_state.all_chats[chat_id]
        label = messages[0]["content"][:25] + "..." if messages else "New Chat 📄"
        
        is_active = (chat_id == st.session_state.current_chat_id)
        if st.button(label, key=chat_id, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.divider()
    st.info(f"📍 Active Tables:\n{', '.join(MY_ACTIVE_TABLES)}")

# --- واجهة المحادثة الرئيسية ---
st.title("🤖 AI SQL Assistant")

current_chat_messages = st.session_state.all_chats[st.session_state.current_chat_id]

# عرض الرسائل
for message in current_chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "df" in message:
            st.dataframe(message["df"])

# إدخال السؤال
if user_query := st.chat_input("Ask about your data..."):
    current_chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # متغير لحمل النتيجة النهائية
        clean_sql = None
        cached_sql = None

        with st.spinner("Checking memory..."):
            cached_sql = assistant.search_memory(user_query, MY_ACTIVE_TABLES)

        if cached_sql:
            st.caption("✨ Answer retrieved from memory (Offline)")
            clean_sql = cached_sql.replace("```sql", "").replace("```", "").strip()
        else:
            with st.spinner("AI Analyzing (Online)..."):
         
                if assistant.check_relevance(user_query, MY_ACTIVE_TABLES):
                    response = assistant.generate_sql(user_query, MY_ACTIVE_TABLES)
                    clean_sql = response.replace("```sql", "").replace("```", "").strip()
                else:
                    error_msg = "Sorry, this question is not related to the selected tables."
                    st.warning(error_msg)
                    current_chat_messages.append({"role": "assistant", "content": error_msg})

        if clean_sql and "SELECT" in clean_sql.upper():
            st.code(clean_sql, language="sql")
            
            with st.spinner("Executing query..."):
                result_df = assistant.execute_query(clean_sql)
    
                if isinstance(result_df, pd.DataFrame):
                    if result_df.empty:
                        st.warning("The query returned no data.")
                        current_chat_messages.append({"role": "assistant", "content": clean_sql})
                    else:
                        st.success(f"✅ Found {len(result_df)} rows.")
                        display_df = result_df.head(10)
                        st.dataframe(display_df, use_container_width=True)
                        
                        if not cached_sql:
                            assistant.save_memory(user_query, clean_sql, MY_ACTIVE_TABLES)
                        
                        current_chat_messages.append({
                            "role": "assistant", 
                            "content": clean_sql, 
                            "df": display_df
                        })
                else:
                    st.error(f"SQL Error: {result_df}")
                    current_chat_messages.append({"role": "assistant", "content": f"Query Error: {result_df}"})
        
        elif clean_sql: 
            st.info(clean_sql)
            current_chat_messages.append({"role": "assistant", "content": clean_sql})
    
    # حفظ التاريخ وعمل تحديث
    save_all_chats()
    st.rerun()