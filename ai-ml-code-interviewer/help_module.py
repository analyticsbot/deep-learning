"""
Help module for the AI-ML Code Interviewer application.
"""

import streamlit as st


class HelpModule:
    """
    Provides help and guidance for using the application.
    """

    def __init__(self):
        """Initialize the help module."""
        self.topics = {
            "coding": "Practice implementing ML/DL algorithms",
            "quiz": "Test your knowledge with multiple-choice questions",
            "settings": "Configure application parameters",
        }

    def get_topic_description(self, topic_key):
        """Get the description for a specific help topic.

        Args:
            topic_key: The key for the topic

        Returns:
            str: Description of the topic
        """
        return self.topics.get(topic_key, "Topic not found")

    def render(self):
        """Render the help UI."""
        st.title("Help")

        st.markdown(
            """
        ## Getting Started

        This application is intended to provide help and guidance for preparing for machine learning and deep learning interviews. You can practise coding and multiple choice questions using this tool.
        This tool uses an LLM to generate code examples and questions. You can use the settings tab to configure the LLM provider and other parameters.
        By default, the application uses LM Studio, which is free and uses open source LLM weights (but these are quantized so the performance might not be the best). You can also use OpenAI, Anthropic, or Google Gemini.

        ### Coding Practice
        - Select a topic and then difficulty level (how much coding you want to do)
        - Implement the requested algorithm or function
        - Submit your code for running and evaluation (by the LLM)

        ### Multiple Choice Questions
        - Choose topic, how many questions and difficulty (easy, medium, hard)
        # TODO: option to select more than one topic
        - Answer ML/DL quiz questions
        - Submit your answers for evaluation (by the LLM)

        ### Settings
        - Configure LLM provider
        # TODO: remove the model when it comes to LM Studio
        - Adjust application parameters
        - Set code execution preferences
        """
        )

        st.subheader("Tips")
        st.markdown(
            """
        - Start with easier difficulty levels and progress gradually
        - Use the feedback provided to improve your solutions
        - Play with prompt engineering to get better results
        - Practice regularly for best results
        """
        )

        # About section
        st.subheader("About")

        st.markdown(
            """
        **AI-ML Code Interviewer** is an interactive tool designed to help you prepare for
        machine learning and deep learning interviews.

        - Practice implementing algorithms from scratch or using standard libraries
        - Test your knowledge with multiple-choice questions
        - Get explanations and feedback to improve your understanding

        This application uses a Large Language Model to generate code examples and questions.
        """
        )
