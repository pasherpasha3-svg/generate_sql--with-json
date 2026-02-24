import os, json, faiss, numpy as np
import time
from dotenv import load_dotenv
from google import genai
from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer
#import chromadb
#from chromadb.utils import embedding_functions
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

DIM = 384
index = faiss.IndexFlatL2(DIM)
history = []
VECTOR_FILE = "my_vector_db.index"
TEXT_FILE = "my_text_db.json"
SEED_FILE = "initial_question.json"

# 1.API Keys

def get_embedding(text):
    embedding = embed_model.encode([text])
    return np.array(embedding).astype('float32').reshape(1, -1)
# 2. Gemini Client Vector DB
def search_memory(user_question):
    if index.ntotal == 0:
        return None
    # علشان يكون في تطابق في المعني مش لازم يكون السؤال بالظبط 
    q_embedding = get_embedding(user_question)
    distances, indices = index.search(q_embedding, 1)
    if distances[0][0] < 0.5:
        matched_idx = indices[0][0]
        return history[matched_idx]["sql"]
    return None


def save_memory(user_question, sql_query):
    q_embedding = get_embedding(user_question)
    index.add(q_embedding)
    history.append({"user_question": user_question, "sql": sql_query})
    
    # حفظ على الهارد ديسك فوراً
    faiss.write_index(index, VECTOR_FILE)
    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    print(" saved in memory ")
#لتخزين الدائم حتي اذا تم قفل البرنامج 
def load_persistence():
    global index, history
    if os.path.exists(VECTOR_FILE) and os.path.exists(TEXT_FILE):
        index = faiss.read_index(VECTOR_FILE)
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        return True
    return False

def seed_memory_from_file():
    if index.ntotal == 0 and os.path.exists(SEED_FILE):
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            seeds = json.load(f)
            for item in seeds:
                save_memory(item["user_question"], item["sql"])

def get_db_schema():
    """يسحب أسماء الجداول والأعمدة أوتوماتيكياً من SQL Server"""
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    
    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    schema_text = ""
    
    for table_name in inspector.get_table_names():
        schema_text += f"\nTable: {table_name}\nColumns: "
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        schema_text += ", ".join(columns) + "\n"
    
    return schema_text

def check_relevance(user_question, full_schema):
    """Agent cheack question on schema or no"""
    prompt = (
        f"You are a gatekeeper for a database. Schema:\n{full_schema}\n"
        f"User Question: {user_question}\n"
        "Can this question be answered using the tables above? Answer ONLY 'YES' or 'NO'."
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip().upper() == "YES"

def generate_sql(user_question, full_schema):
    """Agent generat SQL"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={
            "system_instruction": "You are a SQL expert. Output ONLY the raw SQL code for SQL Server. No markdown, no explanation."
        },
        contents=f"Database Schema:\n{full_schema}\n\nQuestion: {user_question}"
    )
    return response.text.strip()

def main():
    if load_persistence():
        print("Existing memory loaded successfully.")
    seed_memory_from_file()

    print("Checking database and schema...")
    try:
        full_schema = get_db_schema()
        print("Schema successfully retrieved.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    while True:
        user_question = input("\nAsk a question (or type 'exit'): ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting... Goodbye!")
            break

        try:
            # 2. البحث في الذاكرة
            cached = search_memory(user_question)
            if cached:
                print("Result found in Vector DB.")
                print(f" [Memory]: {cached}")
                continue  # يرجع لأول الـ while عشان ياخد سؤال جديد

            # 3. لو مش موجود في الذاكرة، نبدأ نفكر
            print("Thinking...")

            # التأكد من صلة السؤال بقاعدة البيانات
            if check_relevance(user_question, full_schema):
                sql = generate_sql(user_question, full_schema)
                
                print(f"SQL Query generated:\n{sql}")
                
                # حفظ السؤال الجديد في الذاكرة
                save_memory(user_question, sql)
            else:
                print("This question is not related to the database schema.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()