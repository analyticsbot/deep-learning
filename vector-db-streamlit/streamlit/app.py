import ssl
# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['CURL_CA_BUNDLE'] = ''

import streamlit as st
from weaviate import Client
import weaviate
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Weaviate client
client = Client("http://weaviate:8080")  # Replace with your actual Weaviate URL

# Initialize Sentence Transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to create schema
def create_schema(client):
    class_obj = {
        "class": "Question",
        "properties": [
            {
                "name": "text",
                "dataType": ["text"]
            },
            {
                "name": "embedding",
                "dataType": ["number[]"]
            }
        ]
    }
    
    # Check if the schema already exists
    existing_classes = client.schema.get()["classes"]
    class_names = [cls["class"] for cls in existing_classes]
    
    if "Question" not in class_names:
        client.schema.create_class(class_obj)


# Function to fetch similar questions based on embeddings
def fetch_similar_questions(question_input, threshold=0.8):
    # Encode the input question to embedding
    input_embedding = model.encode([question_input])
    
    # Fetch all existing questions
    query = client.query.get("Question", ["text", "embedding"]).do()
    questions = query['data']['Get']['Question']
    
    similar_questions = []
    
    for question in questions:
        question_embedding = np.array(question['embedding']).reshape(1, -1)
        similarity = cosine_similarity(input_embedding, question_embedding)[0][0]
        if similarity >= threshold:
            similar_questions.append(question['text'])
    
    return similar_questions


# Function to insert new question
def insert_question(question_input):
    embedding = model.encode([question_input]).tolist()[0]
    question_object = {
        "text": question_input,
        "embedding": embedding
    }
    client.data_object.create(question_object, "Question")


# Function to fetch all questions
def fetch_all_questions():
    query = client.query.get("Question", ["text"]).do()
    questions = query['data']['Get']['Question']
    return [q['text'] for q in questions]


# Initialize Weaviate schema
create_schema(client)

# Streamlit Tabs for UI
tab1, tab2 = st.tabs(["Insert Question", "View Questions"])

# Tab for Inserting Questions
with tab1:
    st.header("Insert Question")

    question_input = st.text_input("Enter a question")

    if st.button("Check Similar Questions"):
        if question_input:
            similar_questions = fetch_similar_questions(question_input)
            if similar_questions:
                st.write("Similar Questions found:")
                for q in similar_questions:
                    st.write(f"- {q}")
            else:
                st.write("No similar questions found.")
    
    if st.button("Insert Question"):
        similar_questions = fetch_similar_questions(question_input)
        if not similar_questions:
            insert_question(question_input)
            st.write("Question inserted successfully!")
        else:
            st.write("Question is too similar to existing ones, not inserted.")

# Tab for Viewing All Questions
with tab2:
    st.header("View All Questions")
    
    if st.button("Refresh"):
        all_questions = fetch_all_questions()
        st.write("All Inserted Questions:")
        for q in all_questions:
            st.write(f"- {q}")