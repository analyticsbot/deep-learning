"""
Settings module for the AI-ML Code Interviewer application.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import config
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SettingsModule:
    """
    Manages application settings and provides a UI for users to configure them.
    """

    def __init__(self):
        """Initialize the settings module."""
        self.settings_file = ".app_settings.json"
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from file or use defaults.

        Returns:
            Dictionary of settings
        """
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                logger.info("Loaded settings from file")
                return settings
            except Exception as e:
                logger.error(f"Error loading settings: {str(e)}")

        # Default settings
        return {
            "llm_provider": config.LLM_PROVIDER,
            "llm_base_url": config.LLM_BASE_URL,
            "llm_api_key": config.LLM_API_KEY,
            "llm_model": config.LLM_MODEL,
            "llm_temperature": config.LLM_TEMPERATURE,
            "enable_code_execution": config.ENABLE_CODE_EXECUTION,
        }

    def _save_settings(self):
        """Save current settings to file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Saved settings to file")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")

    def get_setting(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value
        """
        return self.settings.get(key, default)

    def update_setting(self, key: str, value: Any):
        """
        Update a setting value.

        Args:
            key: Setting key
            value: New value
        """
        self.settings[key] = value
        self._save_settings()

    def render(self):
        """Render the settings UI."""
        st.title("Settings")

        with st.expander("LLM Configuration", expanded=True):
            # LLM Provider selection
            provider_options = list(config.LLM_PROVIDERS.keys())
            provider_display = [config.LLM_PROVIDERS[p]["name"] for p in provider_options]

            current_provider = self.get_setting("llm_provider")
            provider_index = (
                provider_options.index(current_provider)
                if current_provider in provider_options
                else 0
            )

            selected_provider_name = st.selectbox(
                "LLM Provider", provider_display, index=provider_index
            )

            # Get the provider key from the display name
            selected_provider = provider_options[provider_display.index(selected_provider_name)]

            # If provider changed, update settings
            if selected_provider != current_provider:
                self.update_setting("llm_provider", selected_provider)
                provider_config = config.LLM_PROVIDERS[selected_provider]
                self.update_setting("llm_base_url", provider_config["base_url"])
                self.update_setting("llm_api_key", provider_config["api_key"])
                if provider_config["models"]:
                    self.update_setting("llm_model", provider_config["models"][0])

            # Model selection based on provider
            provider_config = config.LLM_PROVIDERS[selected_provider]

            if selected_provider == "lmstudio":
                from llm_service import LLMService

                llm_service = LLMService()
                try:
                    loaded_models = llm_service.check_loaded_models()
                    if loaded_models:
                        model_options = [model.get("id", "") for model in loaded_models["data"]]
                    else:
                        model_options = provider_config["models"]
                        st.warning("No models loaded in LM Studio. Using default models list.")
                except Exception as e:
                    logger.error(f"Error fetching loaded models: {e}")
                    model_options = provider_config["models"]
            else:
                model_options = provider_config["models"]

            current_model = self.get_setting("llm_model")
            model_index = (
                model_options.index(current_model) if current_model in model_options else 0
            )

            selected_model = st.selectbox("Model", model_options, index=model_index)

            if selected_model != current_model:
                self.update_setting("llm_model", selected_model)

            # API Key input (if not LM Studio)
            if selected_provider != "lmstudio":
                current_api_key = self.get_setting("llm_api_key", "")
                api_key = st.text_input(
                    f"{provider_config['name']} API Key", value=current_api_key, type="password"
                )

                if api_key != current_api_key:
                    self.update_setting("llm_api_key", api_key)

            # Base URL (allow customization)
            current_base_url = self.get_setting("llm_base_url")
            base_url = st.text_input("API Base URL", value=current_base_url)

            if base_url != current_base_url:
                self.update_setting("llm_base_url", base_url)

            # Temperature
            current_temp = self.get_setting("llm_temperature")
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.0, value=float(current_temp), step=0.1
            )

            if temperature != current_temp:
                self.update_setting("llm_temperature", temperature)

            # Display current settings
            st.subheader("Current LLM Settings:")
            st.code(
                f"""
Provider: {selected_provider_name}
Base URL: {self.get_setting('llm_base_url')}
Model: {self.get_setting('llm_model')}
Temperature: {self.get_setting('llm_temperature')}
            """
            )

        with st.expander("Code Execution", expanded=True):
            current_execution = self.get_setting("enable_code_execution")
            enable_execution = st.checkbox(
                "Enable Code Execution",
                value=current_execution,
                help="When enabled, user code can be executed in a restricted environment. Disable if you have security concerns.",
            )

            if enable_execution != current_execution:
                self.update_setting("enable_code_execution", enable_execution)

            st.info(
                "Note: Code execution is performed in a restricted environment, but still carries some security risks. "
                "Only enable if you trust the code you're running."
            )

        # Save to environment button
        if st.button("Apply Settings"):
            # Update environment variables
            os.environ["LLM_PROVIDER"] = self.get_setting("llm_provider")
            os.environ["LLM_BASE_URL"] = self.get_setting("llm_base_url")
            os.environ["LLM_API_KEY"] = self.get_setting("llm_api_key")
            os.environ["LLM_MODEL"] = self.get_setting("llm_model")
            os.environ["LLM_TEMPERATURE"] = str(self.get_setting("llm_temperature"))
            os.environ["ENABLE_CODE_EXECUTION"] = str(self.get_setting("enable_code_execution"))

            # Update config variables
            config.LLM_PROVIDER = self.get_setting("llm_provider")
            config.LLM_BASE_URL = self.get_setting("llm_base_url")
            config.LLM_API_KEY = self.get_setting("llm_api_key")
            config.LLM_MODEL = self.get_setting("llm_model")
            config.LLM_TEMPERATURE = self.get_setting("llm_temperature")
            config.ENABLE_CODE_EXECUTION = self.get_setting("enable_code_execution")

            st.success(
                "Settings applied successfully! Some changes may require restarting the application."
            )
