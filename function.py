import os, json, faiss, numpy as np
from google import genai
from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer
import pandas as pd 

load_dotenv()

class SQLAssistantEngine:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.DIM = 384
        self.index = faiss.IndexFlatL2(self.DIM)
        self.history = []
        self.VECTOR_FILE = "my_vector_db.index"
        self.SEED_FILE = "initial_question.json"
        self.full_schema = ""

        server = os.getenv("DB_SERVER")
        database = os.getenv("DB_DATABASE")
        username = os.getenv("DB_USERNAME")
        password = os.getenv("DB_PASSWORD")
        driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        
        self.conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
        self.engine = create_engine(self.conn_str)
        self.all_tables_dict = {} 
        self.fetch_db_schema()

    def get_embedding(self, text):
        embedding = self.embed_model.encode([text])
        return np.array(embedding).astype('float32').reshape(1, -1)

    def load_memory(self):
        if os.path.exists(self.SEED_FILE):
            with open(self.SEED_FILE, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            
            if self.history:
                print(f" Loading knowledge from {self.SEED_FILE}...")
                for item in self.history:
                    embedding = self.get_embedding(item["user_question"])
                    self.index.add(embedding)
                print(f" Loaded {len(self.history)} examples.")
                return True
        return False

    def search_memory(self, user_question, current_active_tables):
        if self.index.ntotal == 0:
            return None
        
        q_embedding = self.get_embedding(user_question)
        distances, indices = self.index.search(q_embedding, 1)
        
        # إذا وجدنا سؤال مشابه (المسافة أقل من 0.7)
        if distances[0][0] < 0.5:
            matched_item = self.history[indices[0][0]]
            saved_tables = matched_item.get("tables", [])

            if set(saved_tables).issubset(set(current_active_tables)) and len(saved_tables) > 0:
                return matched_item["sql"]
        else:
            return None
    
        return None
    def save_memory(self, question, sql, active_tables):
        embedding = self.get_embedding(question)
        self.index.add(embedding)
        # 
        # بنخزن السؤال والـ SQL وقائمة الجداول اللي استُخدمت
        self.history.append({
            "user_question": question, 
            "sql": sql, 
            "tables": active_tables 
        })
        with open(self.SEED_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        faiss.write_index(self.index, self.VECTOR_FILE)

    def fetch_db_schema(self):  

        from sqlalchemy import inspect 
        inspector = inspect(self.engine) 
        
        self.all_tables_dict = {}  
        self.full_schema = ""

        for table_name in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns(table_name)]

            self.all_tables_dict[table_name] = columns
            
            self.full_schema += f"\nTable: {table_name}\nColumns: " + ", ".join(columns) + "\n"
        
        return self.full_schema

    def get_filtered_schema(self, active_tables):
        if not self.all_tables_dict: 
            self.fetch_db_schema()
            
        schema_text = ""
        for table in active_tables:
            columns_list = self.all_tables_dict.get(table, [])
            if columns_list:
                cols = ", ".join(columns_list)
                schema_text += f"Table: {table}\nColumns: {cols}\n"
        return schema_text
    
    def check_relevance(self, user_question,active_tables):
        current_schema = self.get_filtered_schema(active_tables)
        allowed_list = ", ".join(active_tables)
        prompt = f"""
    You are a strict database gatekeeper.
    
    STRICT RULES:
    1. You ONLY have access to these tables: [{allowed_list}]
    2.Available Tables and Columns:
        {current_schema}
    
    USER QUESTION: "{user_question}"
    
    TASK:
    - Can this question be answered using ONLY the provided tables and columns?
    - If the question needs a table NOT in the list above, answer NO.
    - If the question is about weather, sports, or general chat, answer NO.
    
    ANSWER ONLY 'YES' OR 'NO'.
    """
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = response.text.strip().upper()
        return "YES" in answer
    
    def generate_sql(self, user_question, active_tables):

        current_schema = self.get_filtered_schema(active_tables)
        instructions = (
            "You are a SQL expert and a read-only assistant. "
            "Output ONLY the raw SQL code for SQL Server. No markdown, no explanation. "
            "STRICT RULE: Generate ONLY SELECT statements. If a user asks to modify, "
            "delete, truncate, or drop anything, refuse politely."
            "DO NOT assume any other tables exist (like Product, Orders, etc). "
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": instructions},
            contents=f"Database Schema:\n{current_schema}\n\nQuestion: {user_question}")
        return response.text.strip()
    
    def execute_query(self, sql):
        try:
            df = pd.read_sql(sql, self.engine)
            return df
        except Exception as e:
            return f"Error executing SQL: {str(e)}"