"""
Quiz module for the AI-ML Code Interviewer application.
"""
import logging

import config
import streamlit as st
import utils
from history_manager import HistoryManager
from llm_service import LLMService

# No unused imports


# Set up logging
logger = logging.getLogger(__name__)


class QuizModule:
    """
    Module for handling quiz functionality.
    """

    def __init__(self):
        """Initialize the quiz module."""
        self.llm_service = LLMService()
        self.history_manager = HistoryManager()

    def render(self):
        """Render the quiz UI."""
        st.header("Multiple Choice Practice")

        # Get user settings for multiple-choice questions
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_topic = st.selectbox("Select a topic:", config.ML_DL_TOPICS, key="quiz_topic")

        with col2:
            num_questions = st.number_input(
                "Number of questions:",
                min_value=1,
                max_value=config.MAX_QUESTIONS,
                value=5,
                key="num_questions",
            )

        with col3:
            difficulty_level = st.selectbox(
                "Difficulty level:", config.DIFFICULTY_LEVELS, key="difficulty_level"
            )

        # Generate questions button
        if st.button("Get Questions", key="get_questions_button"):
            with st.spinner("Generating questions..."):
                # Call the LLM service to generate questions
                questions = self.llm_service.generate_quiz(
                    selected_topic, num_questions, difficulty_level
                )

                if questions:
                    # Store the generated questions in session state
                    utils.save_to_session_state("quiz_questions", questions)
                    utils.save_to_session_state("user_answers", {})
                    utils.save_to_session_state("quiz_submitted", False)
                    st.success(f"Generated {len(questions)} questions!")
                else:
                    st.error("Failed to generate questions. Please try again.")

        # Get the generated questions from session state
        questions = utils.get_from_session_state("quiz_questions", [])
        user_answers = utils.get_from_session_state("user_answers", {})
        quiz_submitted = utils.get_from_session_state("quiz_submitted", False)

        if questions:
            st.subheader(f"Quiz: {selected_topic} ({difficulty_level})")

            # Display each question with options
            for idx, q in enumerate(questions):
                question_key = f"question_{idx}"

                st.markdown(f"**Q{idx + 1}: {q['question']}**")

                # Extract option letters (A, B, C, D) for easier reference

                # Display options with radio buttons
                if not quiz_submitted:
                    selected_option = st.radio(
                        f"Select answer for Q{idx + 1}:", q["options"], key=question_key
                    )

                    # Store the selected option in user_answers
                    selected_letter = selected_option.split(".")[0].strip()
                    user_answers[idx] = [selected_letter]

                    # Update session state
                    utils.save_to_session_state("user_answers", user_answers)
                else:
                    # Display the question with color-coded answers
                    for opt in q["options"]:
                        opt_letter = opt.split(".")[0].strip()
                        is_user_selected = opt_letter in user_answers.get(idx, [])
                        is_correct = opt_letter in q["answer"]

                        if is_user_selected and is_correct:
                            st.markdown(f"✅ **{opt}**")
                        elif is_user_selected and not is_correct:
                            st.markdown(f"❌ **{opt}**")
                        elif not is_user_selected and is_correct:
                            st.markdown(f"✓ *{opt}*")
                        else:
                            st.markdown(f"  {opt}")

            # Submit button
            if not quiz_submitted and st.button("Submit Answers", key="submit_answers_button"):
                utils.save_to_session_state("quiz_submitted", True)
                st.experimental_rerun()

            # Show results if quiz is submitted
            if quiz_submitted:
                st.subheader("Quiz Results")

                # Calculate score
                correct_count = 0
                incorrect_questions = []
                for idx, q in enumerate(questions):
                    user_selection = set(user_answers.get(idx, []))
                    correct_answers = set(q["answer"])
                    if user_selection == correct_answers:
                        correct_count += 1
                    else:
                        incorrect_questions.append(idx)

                # Display score with appropriate feedback
                score_percentage = (correct_count / len(questions)) * 100

                # Create a colored score box based on performance
                score_text = f"Score: {correct_count}/{len(questions)} ({score_percentage:.1f}%)"
                if score_percentage >= 80:
                    st.success(f"### Great job! {score_text}")
                elif score_percentage >= 60:
                    st.warning(f"### Good effort! {score_text}")
                else:
                    st.error(f"### Keep practicing! {score_text}")

                # Provide overall feedback based on performance
                if incorrect_questions:
                    st.markdown(
                        f"You missed {len(incorrect_questions)} question(s). Let's review them:"
                    )
                else:
                    st.markdown("Perfect score! You answered all questions correctly.")

                # Save quiz results to history
                try:
                    session_id = self.history_manager.save_quiz_session(
                        topic=selected_topic,
                        difficulty=difficulty_level,
                        questions=questions,
                        user_answers=user_answers,
                        score=correct_count,
                    )
                    st.success(f"Quiz results saved to history! Session ID: {session_id}")
                except (IOError, ValueError, KeyError) as e:
                    logger.error("Failed to save quiz results: %s", str(e))

                # Generate explanations for all questions automatically
                with st.spinner("Generating explanations for all questions..."):
                    # First, show a detailed review of each question
                    for idx, q in enumerate(questions):
                        user_selection = user_answers.get(idx, [])
                        correct_answers = q["answer"]
                        is_correct = set(user_selection) == set(correct_answers)

                        # Create an expandable section for each question
                        with st.expander(
                            f"Question {idx + 1}: {is_correct and '✅ Correct' or '❌ Incorrect'}"
                        ):
                            # Show the question
                            st.markdown(f"**{q['question']}**")

                            # Show each option with appropriate marking
                            for opt in q["options"]:
                                opt_letter = opt.split(".")[0].strip()
                                is_user_selected = opt_letter in user_selection
                                is_correct_answer = opt_letter in correct_answers

                                if is_user_selected and is_correct_answer:
                                    st.markdown(f"✅ **{opt}** (Your correct answer)")
                                elif is_user_selected and not is_correct_answer:
                                    st.markdown(f"❌ **{opt}** (Your incorrect answer)")
                                elif not is_user_selected and is_correct_answer:
                                    st.markdown(f"✓ *{opt}* (Correct answer you missed)")
                                else:
                                    st.markdown(f"  {opt}")

                            # Generate explanation for this question
                            prompt = (
                                f"Explain the following question and why the answer is correct:\n\n"
                                f"Question: {q['question']}\n\n"
                                f"Options: {', '.join(q['options'])}\n\n"
                                f"Correct Answer(s): {', '.join(q['answer'])}\n\n"
                                "Provide a concise but informative explanation that helps "
                                "the learner understand the concept."
                            )
                            explanation = self.llm_service.generate_response(prompt)

                            if explanation:
                                st.markdown("### Explanation")
                                st.markdown(explanation)
                            else:
                                st.error("Failed to generate explanation for this question.")

                # Reset button
                if st.button("Start New Quiz", key="reset_quiz_button"):
                    utils.save_to_session_state("quiz_questions", [])
                    utils.save_to_session_state("user_answers", {})
                    utils.save_to_session_state("quiz_submitted", False)
                    st.experimental_rerun()
