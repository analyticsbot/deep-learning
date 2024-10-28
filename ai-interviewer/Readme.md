# AI Interviewer

This Streamlit application creates an AI-powered interview experience. It allows you to:

- Define interview questions
- Reorder questions based on role and company
- Record candidate responses
- Analyze responses using a Large Language Model (LLM)

## Features

- Speech recognition to capture user responses.
- Option to prioritize questions based on user-defined criteria.
- Feedback generation from an LLM (Language Model) based on the user's answers.
- Capability to ask follow-up questions based on the responses.
- Track previously asked questions and responses.

## Requirements

To run this application, you need the following libraries:

- Streamlit
- SpeechRecognition
- OpenAI (for connecting with the LLM, or alternative LLM provider)
- Textstat (for analyzing the text)
- gtts
- pydub
- sqlite3

You can install the required libraries using pip:

```bash
pip install streamlit speech_recognition openai gtts pydub sqlite3
```

### Usage
1. Clone the repository:
```bash
git clone https://github.com/analyticsbot/deep-learning.git
cd ai-interviewer
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and go to http://localhost:8501.
4. Enter your questions in the text area, and select your preferences for question order and follow-up questions.
5. Click on the microphone icon to start capturing your response.
6. After capturing the response, the application will display your answer and generate feedback.
7. If applicable, a follow-up question will be suggested based on the LLM feedback.

### Functionality Overview
- Speech Recognition: The application listens for audio input using a microphone and converts it to text using Google’s speech recognition service.
- Feedback Analysis: The recognized text is analyzed for brevity, filler words, and overall quality using a language model, and the results are displayed to the user.
- Question Management: Users can define the order of questions and keep track of previously asked questions.

### Further Development:
This is a basic implementation and can be further customized.

Feel free to modify and extend the code to suit your specific needs. Explore features like:
- Advanced LLM prompts for more comprehensive feedback
- Candidate profile management and evaluation scoring
- Better Speech Recognition

### TODO & Bugs
- speak the question
- runs into loop and doesnt listen to user
- writes all the question to the screen
- recording doesn't input all the