import streamlit as st
import speech_recognition as sr
import requests
import random
import textstat
from openai import OpenAI  # Importing OpenAI to connect with LM Studio

# Initialize a session state to track asked questions
if 'asked_questions' not in st.session_state:
    st.session_state['asked_questions'] = []

if 'selected_questions' not in st.session_state:
    st.session_state['selected_questions'] = []

# UI Elements
st.title("Interactive Question-Answering Application")

# Input Fields
question_input = st.text_area("Enter questions, each on a new line:")
order_choice = st.radio("Select question order:", ["In Order", "Random", "Prioritize Based on Role"])
follow_up = st.checkbox("Allow LLM to ask follow-up questions if applicable")
behavior_role = st.text_input("Define the role of the LLM (e.g., HR recruiter):", "HR recruiter")
response_criteria = st.text_area(
    "Describe the type of response needed (e.g., brevity, time taken, quality):", 
    "Provide feedback on time taken, brevity, filler words, repetition, and answer quality."
)

# Adding multiselect for repeated questions
repeat_questions = st.multiselect("Select questions to ask again:", st.session_state['asked_questions'])

# UI for company and role prioritization
company_name = st.text_input("Company Name:", "Meta")
job_role = st.text_input("Job Role:", "Software Engineering Manager, Machine Learning")

# Initialize recognizer for capturing audio responses
recognizer = sr.Recognizer()

# Define feedback function using LM Studio
def analyze_response(response_text):
    # Analyze brevity, filler words, and repetition using textstat
    feedback = {
        "brevity": f"Reading ease score: {textstat.flesch_reading_ease(response_text)}",
        "filler_words": "No obvious filler words detected"  # Placeholder for a better detection mechanism
    }

    # Request feedback from LM Studio
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    prompt = f"As a {behavior_role}, analyze this answer: '{response_text}'. Focus on: {response_criteria}"
    
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

# Process questions input
questions = question_input.split("\n")
questions = [q.strip() for q in questions if q.strip()]  # Clean and ignore blank lines

# Track selected repeat questions
if repeat_questions:
    st.session_state['selected_questions'] = repeat_questions

# Select the question order
if order_choice == "Random":
    random.shuffle(questions)
elif order_choice == "Prioritize Based on Role":
    questions = sorted(questions, key=lambda x: (company_name.lower() in x.lower(), job_role.lower() in x.lower()), reverse=True)

# Filter in repeat questions if any are selected
questions = st.session_state['selected_questions'] + questions

# Track asked questions between sessions
if questions:
    st.session_state['asked_questions'].extend(questions)

# Ask questions and capture responses
for question in questions:
    st.write(f"**Question:** {question}")
    
    # Capture response using microphone by default
    response_text = ""
    st.write("Listening for a response...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        audio = recognizer.listen(source, timeout=5)  # Add timeout to avoid hanging
        try:
            response_text = recognizer.recognize_google(audio)
            st.write("**Your Response:**", response_text)  # Display the recognized text
            
            # End Response Button
            if st.button("End Response"):
                # Generate feedback
                feedback = analyze_response(response_text)
                st.write("**Feedback:**")
                for k, v in feedback.items():
                    st.write(f"{k.capitalize()}: {v}")

                # Optional follow-up question based on LLM feedback
                if follow_up:
                    follow_up_prompt = f"As a {behavior_role}, suggest a follow-up question to deepen this response: '{response_text}'"
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
                    st.session_state['asked_questions'].append(follow_up_question)

        except sr.UnknownValueError as e:
            st.write("**Error:** Could not understand the response." + str(e))
        except Exception as e:
            st.write(f"**Error:** {str(e)}")  # Capture any other exceptions

# Optional: Display questions asked
if st.session_state['asked_questions']:
    st.write("**Questions Asked:**", st.session_state['asked_questions'])
