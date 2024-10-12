import streamlit as st
import sqlite3
import os

# Function to connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect('./db/questions.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database
def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                question TEXT UNIQUE NOT NULL
            )
        ''')
    conn.close()

# Function to insert a question into the database
def insert_question(question):
    conn = get_db_connection()
    with conn:
        try:
            conn.execute('INSERT INTO questions (question) VALUES (?)', (question,))
        except sqlite3.IntegrityError:
            pass  # Ignore if the question already exists
    conn.close()

# Function to find same questions
def find_same_questions(question):
    conn = get_db_connection()
    same_questions = []
    query = 'SELECT question FROM questions WHERE question = ?'
    cursor = conn.execute(query, (question,))
    for row in cursor.fetchall():
        same_questions.append(row['question'])
    conn.close()
    return same_questions

# Initialize the database on app start
init_db()

st.title("Question Similarity Checker")

# Input box for the user to insert a question
user_question = st.text_input("Insert a question:")

if st.button("Submit"):
    if user_question:
        # Check for same questions
        same_questions = find_same_questions(user_question)

        # If no same questions found, add the new question to the database
        if not same_questions:
            insert_question(user_question)
            st.success("Question added successfully!")
            same_questions.append(user_question)  # Include the submitted question as well

        st.write("Questions that are the same:")
        for question in same_questions:
            st.write(f"- {question}")
    else:
        st.error("Please enter a question.")
