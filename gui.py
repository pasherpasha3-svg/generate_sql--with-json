import streamlit as st
import os, json
from dotenv import load_dotenv
from function import SQLAssistantEngine 
import pandas as pd 
import uuid

# --- 1. الإعدادات اليدوية (حدد جداولك هنا) ---
MY_ACTIVE_TABLES = ["FinanceTransaction", "Biller","BillerAggregator","BillerCategory"] 

st.set_page_config(page_title="AI SQL Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def init_engine():
    load_dotenv()
    engine = SQLAssistantEngine()
    engine.fetch_db_schema() # يسحب السكيما بالكامل في الخلفية
    engine.load_memory()
    return engine

assistant = init_engine()

# --- 2. إدارة المحادثات (Chat History Management) ---
if "all_chats" not in st.session_state:
    # بننشئ قاموس لكل المحادثات، وبنبدأ بمحادثة فاضية
    st.session_state.all_chats = {str(uuid.uuid4()): []}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]

# --- 3. Sidebar (القائمة الجانبية) ---
with st.sidebar:
    st.header("💬 Chats")
    
    # زرار New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.all_chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()
    
    st.divider()
    
    # عرض قائمة المحادثات السابقة
    st.subheader("History")
    for chat_id in list(st.session_state.all_chats.keys()):
        messages = st.session_state.all_chats[chat_id]
        # عنوان الشات هو أول سؤال تسأل، لو مفيش بيبقى New Chat
        label = messages[0]["content"][:20] + "..." if messages else "New Chat"
        
        # تمييز الشات المفتوح حالياً
        is_active = (chat_id == st.session_state.current_chat_id)
        if st.button(label, key=chat_id, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.divider()
    st.info(f"📍 Active Tables:\n{', '.join(MY_ACTIVE_TABLES)}")

# --- 4. واجهة المحادثة الرئيسية ---
st.title("🤖 SQL Assistant")
st.caption(f"Querying tables: {', '.join(MY_ACTIVE_TABLES)}")

# استرجاع رسائل المحادثة الحالية فقط
current_chat_messages = st.session_state.all_chats[st.session_state.current_chat_id]

# عرض الرسائل السابقة
for message in current_chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "df" in message:
            st.dataframe(message["df"])

# منطقة إدخال السؤال
if user_query := st.chat_input("Write your question here..."):
    
    # حفظ سؤال المستخدم
    current_chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            
            # الخطوة 1: فحص الصلة بناءً على الجداول المحددة يدوياً
            if assistant.check_relevance(user_query, MY_ACTIVE_TABLES):
                
                # الخطوة 2: البحث في الذاكرة (بشرط توافق الجداول)
                cached_sql = assistant.search_memory(user_query, MY_ACTIVE_TABLES)
                
                if cached_sql:
                    response = cached_sql
                    st.caption("✨ Found in Memory")
                else:
                    # الخطوة 3: توليد SQL جديد
                    response = assistant.generate_sql(user_query, MY_ACTIVE_TABLES)
                    
                clean_sql = response.replace("```sql", "").replace("```", "").strip()

                if clean_sql.upper().startswith("SELECT"):
                    st.code(clean_sql, language="sql")
                    
                    with st.spinner("Executing..."):
                        result_df = assistant.execute_query(clean_sql)
            
                        if isinstance(result_df, pd.DataFrame):
                            if result_df.empty:
                                st.warning("No results found.")
                                current_chat_messages.append({"role": "assistant", "content": clean_sql})
                            else:
                                st.success(f"✅ Showing top 10 rows.")
                                display_df = result_df.head(10)
                                st.dataframe(display_df, use_container_width=True)
                                
                                # حفظ في الذاكرة وفي تاريخ المحادثة
                                if not cached_sql:
                                    assistant.save_memory(user_query, clean_sql, MY_ACTIVE_TABLES)
                                
                                current_chat_messages.append({
                                    "role": "assistant", 
                                    "content": clean_sql, 
                                    "df": display_df
                                })
                        else:
                            st.error("⚠️ SQL Error")
                            st.expander("Details").write(result_df)
                            current_chat_messages.append({"role": "assistant", "content": f"Error: {result_df}"})
                else:
                    st.info(response)
                    current_chat_messages.append({"role": "assistant", "content": response})
            
            else:
                # لو السؤال مش تبع الجداول المحددة
                error_msg = f"This question is not related to the currently selected tables: {', '.join(MY_ACTIVE_TABLES)}"
                st.warning(error_msg)
                current_chat_messages.append({"role": "assistant", "content": error_msg})
    
    # تحديث الصفحة لعرض النتائج وحفظ الـ History
    st.rerun()