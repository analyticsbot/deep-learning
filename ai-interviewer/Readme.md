# AI Interviewer

This Streamlit application creates an AI-powered interview experience. It allows you to:

- Define interview questions
- Reorder questions based on role and company
- Record your responses
- Analyze responses using a Large Language Model (LLM). Currently, I am using Meta-Llama-3.1-8B that I am hosting locally via LM Studio, but this can be easily extended to other models

## Features

- Speech recognition to capture user responses.
- Option to prioritize questions based on user-defined criteria.
- Feedback generation from an LLM (Language Model) based on the user's answers.
- Capability to ask follow-up questions based on the responses.
- Track previously asked questions and responses.

## Requirements

To run this application, you need the following libraries:

- Streamlit
- SpeechRecognition (via Google). TODO: better free and paid alternatives
- OpenAI (for connecting with the LLM, or alternative LLM provider). TODO: experiment with other LLM providers
- Textstat (for analyzing the text)
- gtts
- pydub
- sqlite3

## Setting Up a Virtual Environment and Installing Libraries

1. **Create a Virtual Environment:**

   Open your terminal and run the following command:
   ```bash
   python -m venv myenv
   ```
   Replace myenv with the desired name for your virtual environment.

2. **Activate the Virtual Environment:**

- On Windows:
```bash
myenv\Scripts\activate
```

- On macOS/Linux:
```bash
source myenv/bin/activate
```

3. **Install the Required Libraries:**
Once the virtual environment is activated, use pip to install the libraries:
```bash
pip install streamlit speech_recognition openai gtts pydub
```

4. **Deactivate the Virtual Environment (when done):**
```bash
deactivate
```

## Usage
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

## Functionality Overview
- Speech Recognition: The application listens for audio input using a microphone and converts it to text using Google’s speech recognition service.
- Feedback Analysis: The recognized text is analyzed for brevity, filler words, and overall quality using a language model, and the results are displayed to the user.
- Question Management: Users can define the order of questions and keep track of previously asked questions.

## Contributing
I welcome contributions to make AI Interviewer even better! Here’s how you can help:

1. Report Bugs: If you encounter any issues, please open an issue with details.

2. Suggest Features: Have an idea for improving the application? Create a feature request by creating an issue.

3. Fix TODOs or Bugs:
- Speak the question
- Resolve looping and listening issues
- Prevent all questions from being displayed on the screen
- Improve recording functionality to capture complete answers
- Add a VLM to analyze facial expressions (such as eye contact, looking serious)

4. Submit Pull Requests:

- Fork the repository.
- Create a feature branch (git checkout -b feature-name).
- Commit your changes (git commit -m "Add feature").
- Push to your fork and submit a pull request.

5. Improve Documentation: Help by enhancing the README or adding usage guides.

Feel free to reach out if you have any questions. Thank you for contributing! 🚀

## License
[License](License.md)
