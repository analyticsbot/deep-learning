import streamlit as st
import requests

# Sample list of ML and DL topics
ml_dl_questions = [
    "Linear Regression",
    "Logistic Regression",
    "K-Means Clustering",
    "Self Attention",
    "Multi-Headed Attention",
    "Decision Trees",
    "Random Forests",
    "Support Vector Machines",
    "Neural Networks",
    "Convolutional Neural Networks",
    "Recurrent Neural Networks",
    "Long Short-Term Memory Networks",
    "Generative Adversarial Networks",
    "Gradient Boosting Machines",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Feature Engineering",
    "Hyperparameter Tuning",
    "Model Evaluation",
    "Cross-Validation",
    "Ensemble Learning",
    "Dimensionality Reduction",
    "Principal Component Analysis",
    "t-Distributed Stochastic Neighbor Embedding",
    "Transfer Learning",
    "Active Learning",
    "Time Series Analysis",
    "Anomaly Detection",
    "Autoencoders",
    "Variational Autoencoders",
]

# Title of the app
st.title("Machine Learning & Deep Learning Interview Preparation")

# Define tabs
tab1, tab2 = st.tabs(["Coding", "Multiple Choice Questions"])

# --- Tab 1: Coding ---
with tab1:
    st.header("Coding Practice")

    # Step 1: Select a topic
    selected_topic = st.selectbox("Select a topic to practice:", ml_dl_questions)

    # Step 2: Set the coding intensity
    coding_intensity = st.slider("How much coding do you want to do?", 0, 100)

    # Step 3: Choose between writing from scratch or using standard packages
    use_standard_package = st.selectbox("Would you like to use a standard package?", ["From Scratch", "Using Standard Package"])

    # Step 4: Generate code based on selected topic, coding intensity, and package choice
    if st.button("Get Code"):
        # Prepare the prompt for the LLM
        if coding_intensity == 0:
            prompt = f"Provide full code for {selected_topic} using {use_standard_package.lower()}."
        elif coding_intensity == 100:
            prompt = f"Provide a clean notebook structure for {selected_topic} using {use_standard_package.lower()}."
        else:
            prompt = f"Provide a mixed-level code for {selected_topic} with some parts to be completed using {use_standard_package.lower()}."

        # Make an API call to the LLM
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "messages": [
                    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }
        )

        if response.status_code == 200:
            code = response.json().get('choices')[0]['message']['content']
            # Allow the user to edit the code
            user_code = st.text_area("Edit your code below:", value=code, height=400)

            # Step 5: Allow the user to run the code (if applicable)
            if st.button("Run Code"):
                try:
                    # WARNING: Using exec() can be dangerous if the code is not trusted
                    exec(user_code)
                    st.success("Code executed successfully!")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        else:
            st.error("Failed to retrieve code from the LLM.")

# --- Tab 2: Multiple Choice Questions ---
with tab2:
    st.header("Multiple Choice Practice")

    # Get user settings for multiple-choice questions
    num_questions = st.number_input("How many questions do you want?", min_value=1, max_value=20, value=5)
    difficulty_level = st.selectbox("Select difficulty level", ["Easy", "Medium", "Hard"])

    # Step 1: Generate multiple-choice questions based on user preferences
    if st.button("Get Questions"):
        prompt = f"Generate {num_questions} {difficulty_level} multiple-choice questions on {selected_topic} with correct answers."

        # Make an API call to get questions and answers
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "messages": [
                    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }
        )

        if response.status_code == 200:
            questions = response.json().get('choices')[0]['message']['content']
            questions_data = eval(questions)  # Expecting list of dicts with "question", "options", and "answer"

            # Dictionary to hold user's answers
            user_answers = {}

            # Display each question with options
            for idx, q in enumerate(questions_data):
                st.write(f"**Q{idx + 1}: {q['question']}**")
                options = q['options']
                user_answers[idx] = st.multiselect("Select answer(s)", options)

            # Step 2: Evaluate answers when done
            if st.button("Submit Answers"):
                correct_count = 0

                for idx, q in enumerate(questions_data):
                    correct_answer = q['answer']
                    user_answer = user_answers.get(idx, [])

                    # Compare user answer with correct answer
                    if set(user_answer) == set(correct_answer):
                        st.success(f"Q{idx + 1} Correct! ✅")
                        correct_count += 1
                    else:
                        st.error(f"Q{idx + 1} Incorrect ❌. Correct Answer(s): {', '.join(correct_answer)}")

                # Summary of results
                st.write(f"### Score: {correct_count}/{len(questions_data)}")
        else:
            st.error("Failed to retrieve questions from the LLM.")
