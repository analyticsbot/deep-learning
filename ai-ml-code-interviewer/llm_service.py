"""
LLM service module for interacting with language models.
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

# Import providers conditionally to avoid dependency errors
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Language Models through various APIs.
    """

    def __init__(
        self,
        provider: str = config.LLM_PROVIDER,
        base_url: str = config.LLM_BASE_URL,
        api_key: str = config.LLM_API_KEY,
        model: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            provider: LLM provider (e.g., 'lmstudio', 'openai', 'anthropic', 'google')
            base_url: Base URL for the LLM API
            api_key: API key for authentication
            model: Model identifier to use
            temperature: Temperature parameter for generation
        """
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        # Initialize client based on provider
        self.client = None

        if provider == "lmstudio" or provider == "openai":
            if OPENAI_AVAILABLE:
                try:
                    self.client = OpenAI(base_url=base_url, api_key=api_key)
                    logger.info(f"Initialized {provider} client with model {model}")
                except Exception as e:
                    logger.warning("Error initializing OpenAI client: %s", e)
                    logger.info("Falling back to direct API calls via requests")
            else:
                logger.warning("OpenAI package not installed. Falling back to direct API calls.")

        elif provider == "anthropic":
            if ANTHROPIC_AVAILABLE and api_key:
                try:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Initialized Anthropic client with model %s", model)
                except Exception as e:
                    logger.warning("Error initializing Anthropic client: %s", e)
                    logger.info("Falling back to direct API calls via requests")
            else:
                logger.warning(
                    "Anthropic package not installed or API key missing. Falling back to direct API calls."
                )

        elif provider == "google":
            if GOOGLE_AVAILABLE and api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.client = genai
                    logger.info("Initialized Google Gemini client with model %s", model)
                except Exception as e:
                    logger.warning("Error initializing Google Gemini client: %s", e)
                    logger.info("Falling back to direct API calls via requests")
            else:
                logger.warning(
                    "Google Generative AI package not installed or API key missing. Falling back to direct API calls."
                )

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt to send to the LLM
            system_prompt: Optional system prompt to guide the LLM behavior
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Generated response text or None if failed
        """
        if not system_prompt:
            system_prompt = (
                "You are an intelligent assistant specialized in machine learning and deep learning. "
                "You provide accurate, educational responses that help users learn and prepare for "
                "technical interviews."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                # Handle different providers
                if self.provider == "lmstudio" or self.provider == "openai":
                    if self.client is not None:
                        # Use the OpenAI client
                        response = self.client.chat.completions.create(
                            model=self.model, messages=messages, temperature=self.temperature
                        )
                        return response.choices[0].message.content
                    else:
                        # Fall back to using requests directly
                        response = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json={
                                "model": self.model,
                                "messages": messages,
                                "temperature": self.temperature,
                            },
                            timeout=180,
                        )

                        if response.status_code == 200:
                            return response.json()["choices"][0]["message"]["content"]
                        else:
                            raise Exception(
                                "API returned status code %s: %s",
                                response.status_code, response.text
                            )

                elif self.provider == "anthropic":
                    if self.client is not None:
                        # Use the Anthropic client
                        message = self.client.messages.create(
                            model=self.model,
                            max_tokens=4096,
                            temperature=self.temperature,
                            system=system_prompt
                            if system_prompt
                            else "You are a helpful AI assistant specializing in machine learning and deep learning.",
                            messages=[
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in messages
                                if msg["role"] != "system"
                            ],
                        )
                        return message.content[0].text
                    else:
                        # Fall back to using requests directly
                        headers = {
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        }

                        # Extract user messages
                        user_messages = []
                        for msg in messages:
                            if msg["role"] != "system":
                                user_messages.append(
                                    {"role": msg["role"], "content": msg["content"]}
                                )

                        system_msg = (
                            system_prompt
                            if system_prompt
                            else "You are a helpful AI assistant specializing in machine learning and deep learning."
                        )
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers=headers,
                            json={
                                "model": self.model,
                                "max_tokens": 4096,
                                "temperature": self.temperature,
                                "system": system_msg,
                                "messages": user_messages,
                            },
                            timeout=180,
                        )

                        if response.status_code == 200:
                            return response.json()["content"][0]["text"]
                        else:
                            raise Exception(
                                "API returned status code %s: %s",
                                response.status_code, response.text
                            )

                elif self.provider == "google":
                    google_messages: List[Dict[str, str]] = []
                    if self.client is not None:
                        # Use the Google Gemini client
                        model = self.client.GenerativeModel(model_name=self.model)

                        # Convert messages to Google format

                        for msg in messages:
                            if msg["role"] == "user":
                                google_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                            elif msg["role"] == "assistant":
                                google_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

                        response = model.generate_content(google_messages)
                        return response.text
                    else:
                        # Fall back to using requests directly
                        # Note: Google API has a different structure
                        api_url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

                        # Convert messages to Google format

                        for msg in messages:
                            if msg["role"] == "user":
                                google_messages.append(
                                    {"role": "user", "parts": [{"text": msg["content"]}]}
                                )
                            elif msg["role"] == "assistant":
                                google_messages.append(
                                    {"role": "model", "parts": [{"text": msg["content"]}]}
                                )

                        response = requests.post(
                            api_url,
                            headers={"Content-Type": "application/json"},
                            json={
                                "contents": google_messages,
                                "generationConfig": {"temperature": self.temperature},
                            },
                            timeout=180,
                        )

                        if response.status_code == 200:
                            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            raise Exception(
                                "API returned status code %s: %s",
                                response.status_code, response.text
                            )

                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                logger.error("Error calling LLM API (attempt %s/%s): %s", 
                             attempt+1, max_retries, str(e))
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to get response after maximum retries")
                    return None
        
        # This should never be reached, but adding for type checking
        return None

    def generate_code(
        self, topic: str, intensity: int, use_standard_package: bool
    ) -> Optional[str]:
        """
        Generate code for a specific ML/DL topic.

        Args:
            topic: ML/DL topic to generate code for
            intensity: Coding intensity (0-100)
            use_standard_package: Whether to use standard packages or implement from scratch

        Returns:
            Generated code or None if failed
        """
        package_choice = (
            "using standard packages" if use_standard_package else "implementing from scratch"
        )

        if intensity == 0:
            prompt = f"Provide complete, working Python code for {topic} {package_choice}. Include imports, data generation/loading, implementation, and visualization. The code should be educational and well-commented."
        elif intensity == 100:
            prompt = f"Provide a skeleton structure for implementing {topic} {package_choice}. Include function/class definitions with docstrings but leave the implementation details as TODO comments for the user to complete."
        else:
            prompt = f"Provide partially implemented Python code for {topic} {package_choice}, with about {intensity}% of the implementation left for the user to complete. Mark incomplete sections with TODO comments. Include imports, data generation/loading, partial implementation, and visualization."

        system_prompt = "You are an expert ML/DL coding assistant. Provide clean, educational Python code that follows best practices. Include helpful comments and explanations."

        return self.generate_response(prompt, system_prompt)

    def generate_quiz(
        self, topic: str, num_questions: int, difficulty: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate a quiz with multiple-choice questions.

        Args:
            topic: ML/DL topic for the quiz
            num_questions: Number of questions to generate
            difficulty: Difficulty level (Easy, Medium, Hard)

        Returns:
            List of question dictionaries or None if failed
        """
        prompt = f"""Generate {num_questions} {difficulty.lower()} multiple-choice questions about {topic} for a technical interview.

For each question:
1. Provide a clear, concise question
2. Provide 4 options labeled A, B, C, D
3. Indicate the correct answer(s)

Format your response as a JSON array of objects with the following structure:
[
  {{
    "question": "Question text here",
    "options": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
    "answer": ["B"]  // The correct answer(s) as a list of options
  }},
  // More questions...
]
"""

        system_prompt = "You are an expert ML/DL quiz creator. Create challenging, educational multiple-choice questions that test understanding of key concepts."

        response = self.generate_response(prompt, system_prompt)
        if not response:
            return None

        try:
            # Extract JSON from the response (in case the LLM adds extra text)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Initial JSON parsing failed: {str(e)}. Attempting to fix the JSON..."
                    )

                    # Try to fix common JSON formatting issues
                    # 1. Fix missing commas between objects
                    fixed_json = json_str.replace("}\n  {", "},\n  {")
                    fixed_json = fixed_json.replace("}\r\n  {", "},\r\n  {")

                    # 2. Fix trailing commas
                    fixed_json = fixed_json.replace(",\n]", "\n]")
                    fixed_json = fixed_json.replace(",\r\n]", "\r\n]")

                    # 3. Fix single quotes to double quotes
                    # This regex tries to replace single quotes with double quotes, but not those within words
                    fixed_json = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', fixed_json)

                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        # If still failing, try a more aggressive approach
                        logger.warning(
                            "Still couldn't parse JSON. Attempting to reconstruct the quiz structure..."
                        )

                        # Log the problematic JSON for debugging
                        logger.debug(f"Problematic JSON: {json_str}")

                        # Fallback: Try to manually parse the quiz structure
                        questions = []
                        # Look for patterns like "question": "text", "options": [...], "answer": [...]
                        pattern = (
                            r'\{\s*"question":\s*"([^"]+)"\s*,\s*"options":\s*\[([^\]]+)\]\s*,'
                            r'\s*"answer":\s*\[([^\]]+)\]\s*\}'
                        )
                        question_blocks = re.findall(pattern, fixed_json)

                        for q_text, options_text, answer_text in question_blocks:
                            # Parse options
                            options = [opt.strip().strip('"') for opt in options_text.split(",")]
                            # Parse answers as strings, not dictionaries
                            answers = [ans.strip().strip('"') for ans in answer_text.split(",")]

                            questions.append(
                                {"question": q_text, "options": options, "answer": answers}
                            )

                        if questions:
                            return questions
                        else:
                            logger.error("Failed to manually parse quiz structure")
                            return None
            else:
                logger.error("Could not find JSON array in LLM response")
                logger.debug(f"Response content: {response}")
                return None
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.debug(f"Response content: {response}")
            return None
