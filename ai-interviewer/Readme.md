# AI Interviewer

This application allows users to interactively ask questions and receive responses through speech recognition. It utilizes the `SpeechRecognition` library for capturing audio input and the OpenAI API for generating feedback based on the recognized responses. The application is built using Streamlit for a web-based user interface.

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
- OpenAI (for connecting with the LLM)
- Textstat (for analyzing the text)

You can install the required libraries using pip:

```bash
pip install streamlit SpeechRecognition openai textstat
```

### Usage
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
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