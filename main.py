import os
from dotenv import load_dotenv
from function import SQLAssistantEngine

load_dotenv()

def main():
    assistant = SQLAssistantEngine()
    assistant.load_memory()
    print("Checking database and schema...")
    try:
        assistant.fetch_db_schema()
        print("Schema successfully retrieved.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    while True:
        user_question = input("\nAsk a question (or type 'exit'): ").strip()

        if not user_question:
            print(" You didn't ask anything. Please type a question.")
            continue
        
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting... Goodbye!")
            break

        try:
            cached = assistant.search_memory(user_question)
            if cached:
                print("Result found in Vector DB.")
                print(f" [Memory]: {cached}")
                continue
            print("Thinking...")
            if assistant.check_relevance(user_question):
                sql = assistant.generate_sql(user_question)
                
                print(f"SQL Query generated:\n{sql}")
                
                if "SELECT" in sql.upper():
                    assistant.save_memory(user_question, sql)
                else:
                    print(f"Notice: {sql}")
            else:
                print("This question is not related to the database schema.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()