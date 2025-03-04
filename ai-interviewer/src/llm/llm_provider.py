import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider:
    """
    A class to handle different LLM providers (LM Studio, OpenAI, Claude, Grok)
    """
    
    def __init__(self):
        # Initialize API keys and base URLs from environment variables
        self.lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.lmstudio_api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Check which providers are available
        self.has_lmstudio = bool(self.lmstudio_api_key)
        self.has_openai_key = bool(self.openai_api_key)
        self.has_anthropic_key = bool(self.anthropic_api_key)
        self.has_google_key = bool(self.google_api_key)
        
        # Cache for storing responses
        self.cache = {}
        
        # Log available providers
        providers = self.get_available_providers()
        st.info(f"Available LLM providers: {', '.join(providers)}")
        
        # Cache for storing responses
        self.cache = {}
        
    def _get_cache_key(self, provider, model, prompt, temperature):
        """Generate a cache key for storing and retrieving responses"""
        # Create a simplified version of the prompt for caching
        # This helps with minor variations in prompts that would produce the same response
        simplified_prompt = prompt.lower().strip()
        return f"{provider}_{model}_{hash(simplified_prompt)}_{temperature}"
    
    def call_llm(self, prompt, provider="lmstudio", model=None, temperature=0.7, system_prompt=None, stream=True, use_cache=True):
        """
        Call the specified LLM provider with the given prompt
        
        Args:
            prompt (str): The user prompt to send to the LLM
            provider (str): The LLM provider to use (lmstudio, openai, anthropic, google)
            model (str): The specific model to use (provider-dependent)
            temperature (float): The temperature parameter for generation
            system_prompt (str): The system prompt to use (if applicable)
            stream (bool): Whether to stream the response
            use_cache (bool): Whether to use cached responses
            
        Returns:
            str: The generated response
        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = "You are an intelligent assistant."
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(provider, model, prompt, temperature)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Set default models based on provider
        if model is None:
            if provider == "lmstudio":
                model = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
            elif provider == "openai":
                model = "gpt-3.5-turbo"
            elif provider == "anthropic":
                model = "claude-3-sonnet-20240229"
            elif provider == "google":
                model = "gemini-pro"
        
        response_text = ""
        
        try:
            # Call the appropriate provider
            if provider == "lmstudio":
                response_text = self._call_lmstudio(prompt, model, system_prompt, temperature, stream)
            elif provider == "openai":
                response_text = self._call_openai(prompt, model, system_prompt, temperature, stream)
            elif provider == "anthropic":
                response_text = self._call_anthropic(prompt, model, system_prompt, temperature, stream)
            elif provider == "google":
                response_text = self._call_google(prompt, model, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Cache the response if caching is enabled
            if use_cache:
                cache_key = self._get_cache_key(provider, model, prompt, temperature)
                self.cache[cache_key] = response_text
            
            return response_text
        
        except Exception as e:
            st.error(f"Error calling {provider} LLM: {str(e)}")
            return f"Error: {str(e)}"
    
    def _call_lmstudio(self, prompt, model, system_prompt, temperature, stream):
        """Call LM Studio API"""
        try:
            # Import here to avoid initialization issues
            from openai import OpenAI
            
            # Create client on demand
            client = OpenAI(
                api_key=self.lmstudio_api_key,
                base_url=self.lmstudio_base_url
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=stream,
            )
            
            if stream:
                response_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                return response_text.strip()
            else:
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            st.error(f"Error with LM Studio: {str(e)}")
            # Fallback to other providers if available
            if self.has_openai_key:
                st.warning("Falling back to OpenAI...")
                return self._call_openai(prompt, "gpt-3.5-turbo", system_prompt, temperature, stream)
            raise e
    
    def _call_openai(self, prompt, model, system_prompt, temperature, stream):
        """Call OpenAI API"""
        if not self.has_openai_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            # Import here to avoid initialization issues
            from openai import OpenAI
            
            # Create client on demand
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=stream,
            )
            
            if stream:
                response_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                return response_text.strip()
            else:
                return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error with OpenAI: {str(e)}")
            raise e
    
    def _call_anthropic(self, prompt, model, system_prompt, temperature, stream):
        """Call Anthropic (Claude) API"""
        if not self.has_anthropic_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            # Import here to avoid initialization issues
            from anthropic import Anthropic
            
            # Create client on demand with http_client=None to avoid proxies issue
            client = Anthropic(api_key=self.anthropic_api_key, http_client=None)
            
            message = client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=1000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            st.error(f"Error with Anthropic: {str(e)}")
            # Try alternative approach if the first one fails
            try:
                import anthropic
                client = anthropic.Client(api_key=self.anthropic_api_key)
                response = client.completion(
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    model=model,
                    max_tokens_to_sample=1000,
                    temperature=temperature,
                )
                return response.completion
            except Exception as e2:
                st.error(f"Error with alternative Anthropic approach: {str(e2)}")
                # Try a third approach with direct API call
                try:
                    import requests
                    import json
                    
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01"
                    }
                    
                    data = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": temperature
                    }
                    
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        return response.json()["content"][0]["text"]
                    else:
                        st.error(f"Direct API call failed: {response.status_code} - {response.text}")
                        raise e
                except Exception as e3:
                    st.error(f"All Anthropic approaches failed: {e3}")
                    raise e
    
    def _call_google(self, prompt, model, temperature):
        """Call Google (Gemini/Grok) API"""
        if not self.has_google_key:
            raise ValueError("Google API key not configured")
        
        try:
            # Import here to avoid initialization issues
            import google.generativeai as genai
            
            # Configure the API key
            genai.configure(api_key=self.google_api_key)
            
            # Create the model and generate content
            model = genai.GenerativeModel(model_name=model)
            response = model.generate_content(prompt, generation_config={"temperature": temperature})
            
            return response.text
        except Exception as e:
            st.error(f"Error with Google Generative AI: {str(e)}")
            raise e
    
    def get_available_providers(self):
        """Return a list of available LLM providers based on configured API keys"""
        providers = ["lmstudio"]  # LM Studio is always available (local)
        
        if self.has_openai_key:
            providers.append("openai")
        
        if self.has_anthropic_key:
            providers.append("anthropic")
        
        if self.has_google_key:
            providers.append("google")
        
        return providers
