import os, json, faiss, numpy as np
from google import genai
from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer

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

    def search_memory(self, user_question):
        if self.index.ntotal == 0:
            return None
        
        q_embedding = self.get_embedding(user_question)
        distances, indices = self.index.search(q_embedding, 1)
        
        if distances[0][0] < 0.7:
            matched_idx = indices[0][0]
            return self.history[matched_idx]["sql"]
        return None

    def save_memory(self, question, sql):
        """حفظ سؤال جديد في الذاكرة والملفات"""
        embedding = self.get_embedding(question)
        self.index.add(embedding)
        
        self.history.append({"user_question": question, "sql": sql})
        with open(self.SEED_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
            
        faiss.write_index(self.index, self.VECTOR_FILE)
        print("   saved in memory   ")

    def fetch_db_schema(self):
        """سحب هيكل البيانات من SQL Server"""
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
        
        self.full_schema = schema_text
        return schema_text

    def check_relevance(self, user_question):
        prompt = (
            f"You are a gatekeeper for a database. Schema:\n{self.full_schema}\n"
            f"User Question: {user_question}\n"
            "Can this question be answered using the tables above? Answer ONLY 'YES' or 'NO'."
        )
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip().upper() == "YES"

    def generate_sql(self, user_question):
        instructions = (
            "You are a SQL expert and a read-only assistant. "
            "Output ONLY the raw SQL code for SQL Server. No markdown, no explanation. "
            "STRICT RULE: Generate ONLY SELECT statements. If a user asks to modify, "
            "delete, truncate, or drop anything, refuse politely."
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": instructions},
            contents=f"Database Schema:\n{self.full_schema}\n\nQuestion: {user_question}"
        )
        return response.text.strip()