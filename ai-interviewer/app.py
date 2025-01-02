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

# Functions for session persistence
def fetch_asked_questions():
    cursor.execute("SELECT question FROM asked_questions")
    return [row[0] for row in cursor.fetchall()]

def add_asked_question(question):
    cursor.execute("INSERT INTO asked_questions (question) VALUES (?)", (question,))
    conn.commit()

# Play text as audio
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp_audio.mp3")
    audio = AudioSegment.from_mp3("temp_audio.mp3")
    play(audio)
    os.remove("temp_audio.mp3")

# Initialize session state
st.session_state['asked_questions'] = fetch_asked_questions()
if 'selected_questions' not in st.session_state:
    st.session_state['selected_questions'] = []
if 'reordered_questions' not in st.session_state:
    st.session_state['reordered_questions'] = []

# Function to make LLM call
def call_llm(prompt, model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=True,
    )
    response_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content
    return response_text.strip()

# Reorder questions based on order choice
def reorder_questions(questions, order_choice, company_name, job_role):
    if order_choice == "Random":
        random.shuffle(questions)
    elif order_choice == "Prioritize Based on Role":
        prompt = f"Prioritize these interview questions for a candidate applying for '{job_role}' at '{company_name}':\n" + "\n".join([f"- {q}" for q in questions])
        reordered = call_llm(prompt)
        return [q.strip("- ") for q in reordered.split("\n") if q.strip()]
    return questions

# Initialize the app UI
st.title("AI Interviewer")
default_questions = """Tell me about yourself.
Why do you want to join Google?
What are your strengths?
Describe a challenging project you've worked on.
Where do you see yourself in five years?"""
question_input = st.text_area("Enter questions, each on a new line:", value=default_questions)
order_choice = st.radio("Select question order:", ["In Order", "Random", "Prioritize Based on Role"])
follow_up = st.checkbox("Allow LLM to ask follow-up questions if applicable")
behavior_role = st.text_input("Define the role of the LLM (e.g., HR recruiter):", "HR recruiter")
response_criteria = st.text_area("Describe response criteria (e.g., brevity, quality):", "Provide feedback on brevity, clarity, and relevance.")
# UI for company and role prioritization
company_name = st.text_input("Company Name:", "Google")
job_role = st.text_input("Job Role:", "Software Engineer")

# Prepare questions
questions = [q.strip() for q in question_input.split("\n") if q.strip()]
questions = reorder_questions(questions, order_choice, company_name, job_role)
st.session_state['reordered_questions'] = questions
st.text_area("Questions (After Reordering):", "\n".join(st.session_state['reordered_questions']), height=200)

# Track selected repeat questions
repeat_questions = st.multiselect("Select questions to ask again:", st.session_state['asked_questions'])
if repeat_questions:
    st.session_state['selected_questions'] = repeat_questions

# Record and analyze response
# Initialize recognizer for capturing audio responses
recognizer = sr.Recognizer()

# Track recording state in session state
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False
if 'response_text' not in st.session_state:
    st.session_state['response_text'] = ""

start_recording = st.button("Start Recording")
end_recording = st.button("End Recording")

# Function to listen and capture audio
# Function to capture audio segments continuously
def continuous_listening():
    with sr.Microphone() as source:
        st.write("Listening... (click 'End Recording' to stop)")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while st.session_state['is_recording']:
            try:
                audio_segment = recognizer.listen(source, timeout=50, phrase_time_limit=100)  # Capture each segment
                text_segment = recognizer.recognize_google(audio_segment)
                st.session_state['response_text'] += " " + text_segment
                # Optionally display live segments
                st.write("Captured segment:", text_segment)  
            except sr.UnknownValueError:
                st.write("Could not understand segment, continuing...")
            except Exception as e:
                st.write(f"Error: {str(e)}")
                break


# Initialize the question index in session state if it doesn't exist
if 'question_index' not in st.session_state:
    st.session_state['question_index'] = 0  # Start with the first question
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False
if 'response_text' not in st.session_state:
    st.session_state['response_text'] = ""

# Main logic to display one question at a time
if st.session_state['question_index'] < len(questions):
    current_question = questions[st.session_state['question_index']]
    st.write(f"**Question:** {current_question}")

    # Check if recording should start
    if start_recording and not st.session_state['is_recording']:
        st.session_state['is_recording'] = True
        st.session_state['response_text'] = ""
        speak_text(current_question)  # Speak the question
        st.write("Recording started... Click 'End Recording' when done.")
        continuous_listening()

    # Check if recording should end
    if end_recording:
        st.session_state['is_recording'] = False
        if st.session_state['response_text']:
            st.write("**Your Response:**", st.session_state['response_text'])

            # Call the LLM for feedback
            feedback = call_llm(
                f"As a {behavior_role} at {company_name}, interviewing for {job_role}, imagine you asked the question '{current_question}', analyze this answer: '{st.session_state['response_text']}'. Focus on: {response_criteria}. Keep the response brief and use bullet points."
            )
            st.write("**Feedback:**", feedback)

            # Move to the next question
            st.session_state['question_index'] += 1
            st.session_state['response_text'] = ""  # Reset response text for the next question
else:
    st.write("Interview complete! Thank you for your responses.")

# # Display questions asked
# if st.session_state['asked_questions']:
#     st.write("**Questions Asked:**", st.session_state['asked_questions'])

conn.close()

