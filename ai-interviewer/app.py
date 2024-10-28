import streamlit as st
import speech_recognition as sr
import requests
import random
import textstat
from openai import OpenAI
import sqlite3
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os

# Setup database connection for session persistence
conn = sqlite3.connect("session_data.db")
cursor = conn.cursor()

# Create a table for asked questions if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS asked_questions (question TEXT)''')
conn.commit()

# Fetch questions from the database for persistent session storage
def fetch_asked_questions():
    cursor.execute("SELECT question FROM asked_questions")
    return [row[0] for row in cursor.fetchall()]

# Insert questions into the database
def add_asked_question(question):
    cursor.execute("INSERT INTO asked_questions (question) VALUES (?)", (question,))
    conn.commit()

# Function to play text as audio
def speak_text(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    tts.save("temp_audio.mp3")
    
    # Play the audio file
    audio = AudioSegment.from_mp3("temp_audio.mp3")
    play(audio)
    
    # Remove the temporary audio file
    os.remove("temp_audio.mp3")

# Initialize session state
st.session_state['asked_questions'] = fetch_asked_questions()
if 'selected_questions' not in st.session_state:
    st.session_state['selected_questions'] = []

if 'reordered_questions' not in st.session_state:
    st.session_state['reordered_questions'] = []


# UI Elements
st.title("AI Interviewer")

# Input Fields
# Pre-defined questions
default_questions = """Tell me about yourself.
Why do you want to join Meta?
What are your strengths?
Describe a challenging project you've worked on.
Where do you see yourself in five years?
"""

# Text area with default questions
question_input = st.text_area("Enter questions, each on a new line:", value=default_questions)
order_choice = st.radio("Select question order:", ["In Order", "Random", "Prioritize Based on Role"])
follow_up = st.checkbox("Allow LLM to ask follow-up questions if applicable")
behavior_role = st.text_input("Define the role of the LLM (e.g., HR recruiter):", "HR recruiter")
response_criteria = st.text_area(
    "Describe the type of response needed (e.g., brevity, time taken, quality):", 
    "Provide feedback on time taken, brevity, filler words, repetition, and answer quality. Keep it brief and bulleted"
)

# UI for company and role prioritization
company_name = st.text_input("Company Name:", "Meta")
job_role = st.text_input("Job Role:", "Software Engineering Manager")

# Split and clean questions
questions = question_input.split("\n")
questions = [q.strip() for q in questions if q.strip()]

# Reorder questions based on the selected order choice
if order_choice == "Random":
    random.shuffle(questions)
elif order_choice == "Prioritize Based on Role":
    def prioritize_questions(questions, job_role, company_name):
        # Send questions, role, and company to the LLM to prioritize
        prompt = f"Prioritize these interview questions for a candidate applying for the role '{job_role}' at '{company_name}':\n\n"
        prompt += "\n".join([f"- {q}" for q in questions])

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=True,
        )

        # Assuming the response will return prioritized questions as plain text
        # Collect the response chunks
        prioritized_questions = []
        for chunk in response:
            if "choices" in chunk and "delta" in chunk.choices[0] and "content" in chunk.choices[0].delta:
                prioritized_questions.append(chunk.choices[0].delta.content)

        # Join the collected chunks into a single response string
        full_response = "".join(prioritized_questions)
        return [q.strip("- ") for q in full_response.split("\n") if q.strip()]
    
    questions = prioritize_questions(questions, job_role, company_name)

st.session_state['reordered_questions'] = questions

# Display reordered questions after prioritization
st.text_area(
    "Questions (After Reordering):",
    value="\n".join(st.session_state['reordered_questions']),
    height=200
)

# Adding multiselect for repeated questions
repeat_questions = st.multiselect("Select questions to ask again:", st.session_state['asked_questions'])

# Define feedback function using LM Studio
def analyze_response(response_text, behavior_role, company, role, response_criteria):
    feedback = {
        "brevity": f"Reading ease score: {textstat.flesch_reading_ease(response_text)}",
        "filler_words": "No obvious filler words detected"  # Placeholder for a better detection mechanism
    }

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    prompt = f"As a {behavior_role} for {company}, interviewing for {role}, analyze this answer: '{response_text}'. Focus on: {response_criteria}"
    
    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=True,
    )
    
    new_message = {"role": "assistant", "content": ""}
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            new_message["content"] += chunk.choices[0].delta.content

    feedback['LLM_feedback'] = new_message['content'].strip()
    
    return feedback

# Add default starting questions
questions = ["tell me about yourself", "why do you want to join meta"]
user_questions = question_input.split("\n")
user_questions = [q.strip() for q in user_questions if q.strip()]
questions += user_questions

# Track selected repeat questions
if repeat_questions:
    st.session_state['selected_questions'] = repeat_questions

# Select the question order
if order_choice == "Random":
    random.shuffle(questions)
elif order_choice == "Prioritize Based on Role":
    # Use LLM prioritization based on job role
    ordered_questions = []
    for question in questions:
        prompt = f"As a {behavior_role} hiring for {job_role}, would you prioritize asking the question: '{question}'?"
        response = analyze_response(prompt, behavior_role, company_name, job_role, response_criteria)
        if "prioritize" in response.get("LLM_feedback", "").lower():
            ordered_questions.insert(0, question)
        else:
            ordered_questions.append(question)
    questions = ordered_questions

# Filter in repeat questions if any are selected
questions = st.session_state['selected_questions'] + questions    

# Initialize recognizer for capturing audio responses
recognizer = sr.Recognizer()

start_recording = st.button("Start Recording")
end_recording = st.button("End Recording")

for question in questions:
    st.write(f"**Question:** {question}")
    # speak_text(question)
    
    if start_recording:
        st.write("Recording started... Click 'End Recording' when done.")
        # Ask questions and capture responses
    
    response_text = ""
    st.write("Listening for a response...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        audio = recognizer.listen(source, timeout=500)
        try:
            response_text = recognizer.recognize_google(audio)
            st.write("**Your Response:**", response_text)
            print (1, response_text)
            
            # End Response Button
            if end_recording:
                feedback = analyze_response(response_text, behavior_role, company_name, job_role, response_criteria)
                print (2, response_text)
                st.write("**Feedback:**")
                for k, v in feedback.items():
                    st.write(f"{k.capitalize()}: {v}")
                
                add_asked_question(question)

                if follow_up:
                    follow_up_prompt = f"As a {behavior_role} at {company_name}, suggest a follow-up question based on: '{response_text}' only if you feel so"
                    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                    follow_up_response = client.chat.completions.create(
                        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                        messages=[
                            {"role": "system", "content": "You are an intelligent assistant."},
                            {"role": "user", "content": follow_up_prompt}
                        ],
                        temperature=0.7,
                        stream=True,
                    )
                    follow_up_question = ""
                    for chunk in follow_up_response:
                        if chunk.choices[0].delta.content:
                            follow_up_question += chunk.choices[0].delta.content

                    st.write("**Follow-Up Question:**", follow_up_question)
                    add_asked_question(follow_up_question)

        except sr.UnknownValueError:
            st.write("**Error:** Could not understand the response.")
        except Exception as e:
            st.write(f"**Error:** {str(e)}")

# Optional: Display questions asked
if st.session_state['asked_questions']:
    st.write("**Questions Asked:**", st.session_state['asked_questions'])

conn.close()
