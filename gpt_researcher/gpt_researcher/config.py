"""Configuration management for GPT Researcher.

This module provides the Config class that manages all configuration
settings for GPT Researcher including LLM providers, embeddings,
retrievers, and various operational parameters.

(学习版: 由原版 gpt_researcher/config/config.py + variables/default.py
 精简合并为单个文件, 逻辑与原版一致。)
"""

import json
import json_repair
import os
import warnings
from typing import Any, Dict, List, Type, Union, get_args, get_origin

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts

# ---- 配置键类型声明(对应原版 config/variables/base.py 的 BaseConfig) ----
TYPE_HINTS: Dict[str, Type] = {
    "RETRIEVER": str,
    "EMBEDDING": str,
    "SIMILARITY_THRESHOLD": float,
    "FAST_LLM": str,
    "SMART_LLM": str,
    "STRATEGIC_LLM": str,
    "FAST_TOKEN_LIMIT": int,
    "SMART_TOKEN_LIMIT": int,
    "STRATEGIC_TOKEN_LIMIT": int,
    "BROWSE_CHUNK_MAX_LENGTH": int,
    "SUMMARY_TOKEN_LIMIT": int,
    "TEMPERATURE": float,
    "USER_AGENT": str,
    "MAX_SEARCH_RESULTS_PER_QUERY": int,
    "MEMORY_BACKEND": str,
    "TOTAL_WORDS": int,
    "REPORT_FORMAT": str,
    "CURATE_SOURCES": bool,
    "MAX_ITERATIONS": int,
    "LANGUAGE": str,
    "AGENT_ROLE": Union[str, None],
    "SCRAPER": str,
    "MAX_SCRAPER_WORKERS": int,
    "SCRAPER_RATE_LIMIT_DELAY": float,
    "MAX_SUBTOPICS": int,
    "REPORT_SOURCE": Union[str, None],
    "DOC_PATH": str,
    "PROMPT_FAMILY": str,
    "LLM_KWARGS": dict,
    "EMBEDDING_KWARGS": dict,
    "VERBOSE": bool,
    "DEEP_RESEARCH_CONCURRENCY": int,
    "DEEP_RESEARCH_DEPTH": int,
    "DEEP_RESEARCH_BREADTH": int,
    "MCP_SERVERS": List[Dict[str, Any]],
    "MCP_AUTO_TOOL_SELECTION": bool,
    "MCP_ALLOWED_ROOT_PATHS": List[str],
    "MCP_STRATEGY": str,
    "REASONING_EFFORT": str,
    "IMAGE_GENERATION_MODEL": Union[str, None],
    "IMAGE_GENERATION_MAX_IMAGES": int,
    "IMAGE_GENERATION_ENABLED": bool,
    "IMAGE_GENERATION_STYLE": str,
    "IMAGE_GENERATION_PROVIDER": str,
}

# ---- 默认配置(对应原版 config/variables/default.py) ----
DEFAULT_CONFIG: Dict[str, Any] = {
    "RETRIEVER": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "SIMILARITY_THRESHOLD": 0.42,
    "FAST_LLM": "openai:gpt-5.4-mini",
    "SMART_LLM": "openai:gpt-5.4",  # Has support for long responses (2k+ words).
    "STRATEGIC_LLM": "openai:gpt-5.4",  # Reasoning model used for planning.
    "FAST_TOKEN_LIMIT": 6000,
    "SMART_TOKEN_LIMIT": 12000,
    "STRATEGIC_TOKEN_LIMIT": 8000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.4,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None,
    "SCRAPER": "bs",
    "MAX_SCRAPER_WORKERS": 15,
    "SCRAPER_RATE_LIMIT_DELAY": 0.0,
    "MAX_SUBTOPICS": 3,
    "LANGUAGE": "chinese",
    "REPORT_SOURCE": "web",
    "DOC_PATH": "./my-docs",
    "PROMPT_FAMILY": "default",
    "LLM_KWARGS": {},
    "EMBEDDING_KWARGS": {},
    "VERBOSE": False,
    # Deep research specific settings
    "DEEP_RESEARCH_BREADTH": 3,
    "DEEP_RESEARCH_DEPTH": 2,
    "DEEP_RESEARCH_CONCURRENCY": 4,

    # MCP retriever specific settings
    "MCP_SERVERS": [],
    "MCP_AUTO_TOOL_SELECTION": True,
    "MCP_ALLOWED_ROOT_PATHS": [],
    "MCP_STRATEGY": "fast",
    "REASONING_EFFORT": "medium",

    # Image generation settings (optional)
    "IMAGE_GENERATION_MODEL": "models/gemini-2.5-flash-image",
    "IMAGE_GENERATION_MAX_IMAGES": 3,
    "IMAGE_GENERATION_ENABLED": False,
    "IMAGE_GENERATION_STYLE": "dark",
    "IMAGE_GENERATION_PROVIDER": "google",
}


class Config:
    """Configuration manager for GPT Researcher.

    Handles loading, parsing, and managing all configuration settings
    from files, environment variables, and defaults.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the config class.

        Args:
            config_path: Optional path to a JSON configuration file.
        """
        self.config_path = config_path
        self.llm_kwargs: Dict[str, Any] = {}
        self.embedding_kwargs: Dict[str, Any] = {}

        config_to_use = self.load_config(config_path)
        self._set_attributes(config_to_use)
        self._set_embedding_attributes()
        self._set_llm_attributes()
        self._handle_deprecated_attributes()
        if config_to_use['REPORT_SOURCE'] != 'web':
            self._set_doc_path(config_to_use)

        # MCP support configuration
        self.mcp_servers = self.mcp_servers or []
        self.mcp_allowed_root_paths = self.mcp_allowed_root_paths or []

    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """Set configuration attributes from config dictionary.

        Merges environment variables with config file values, with
        environment variables taking precedence.
        """
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                value = self.convert_env_value(key, env_value, TYPE_HINTS[key])
            setattr(self, key.lower(), value)

        # Handle RETRIEVER with default value
        retriever_env = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {str(e)}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _set_embedding_attributes(self) -> None:
        """Parse and set embedding provider and model attributes."""
        self.embedding_provider, self.embedding_model = self.parse_embedding(
            self.embedding
        )

    def _set_llm_attributes(self) -> None:
        """Parse and set LLM provider and model attributes for all LLM types."""
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(self.fast_llm)
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(self.strategic_llm)
        self.reasoning_effort = self.parse_reasoning_effort(os.getenv("REASONING_EFFORT"))

    def _handle_deprecated_attributes(self) -> None:
        """Handle deprecated configuration attributes with warnings."""
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.embedding_provider = (
                os.environ["EMBEDDING_PROVIDER"] or self.embedding_provider
            )

            embedding_provider = os.environ["EMBEDDING_PROVIDER"]
            if embedding_provider == "ollama":
                self.embedding_model = os.environ["OLLAMA_EMBEDDING_MODEL"]
            elif embedding_provider == "custom":
                self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "custom")
            elif embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "azure_openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "huggingface":
                self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            elif embedding_provider == "gigachat":
                self.embedding_model = "Embeddings"
            elif embedding_provider == "google_genai":
                self.embedding_model = "text-embedding-004"
            else:
                raise Exception("Embedding provider not found.")

        _deprecation_warning = (
            "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and "
            "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        )
        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.fast_llm_provider
            )
            self.smart_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.smart_llm_provider
            )
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_model = os.environ["FAST_LLM_MODEL"] or self.fast_llm_model
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.smart_llm_model = os.environ["SMART_LLM_MODEL"] or self.smart_llm_model

    def _set_doc_path(self, config: Dict[str, Any]) -> None:
        self.doc_path = config['DOC_PATH']
        if self.doc_path:
            try:
                self.validate_doc_path()
            except Exception as e:
                print(f"Warning: Error validating doc_path: {str(e)}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG['DOC_PATH']

    @classmethod
    def load_config(cls, config_path: str | None) -> Dict[str, Any]:
        """Load a configuration by name."""
        config_path = config_path or os.environ.get("CONFIG_PATH")
        if not config_path:
            return DEFAULT_CONFIG

        if not os.path.exists(config_path):
            if config_path and config_path != "default":
                print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")
                if not config_path.endswith(".json"):
                    print(f"Do you mean '{config_path}.json'?")
            return DEFAULT_CONFIG

        with open(config_path, "r") as f:
            custom_config = json.load(f)

        # Merge with default config to ensure all keys are present
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(custom_config)
        return merged_config

    def parse_retrievers(self, retriever_str: str) -> List[str]:
        """Parse the retriever string into a list of retrievers and validate them."""
        from .retrievers.utils import get_all_retriever_names

        retrievers = [retriever.strip()
                      for retriever in retriever_str.split(",")]
        valid_retrievers = get_all_retriever_names() or []
        invalid_retrievers = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(
                f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. "
                f"Valid options are: {', '.join(valid_retrievers)}."
            )
        return retrievers

    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """Parse llm string into (llm_provider, llm_model)."""
        from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {llm_provider}.\nSupported llm providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return llm_provider, llm_model
        except ValueError:
            raise ValueError(
                "Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' "
                "Eg 'openai:gpt-4o-mini'"
            )

    @staticmethod
    def parse_reasoning_effort(reasoning_effort_str: str | None) -> str | None:
        """Parse reasoning effort string."""
        if reasoning_effort_str is None:
            return ReasoningEfforts.Medium.value
        if reasoning_effort_str not in [effort.value for effort in ReasoningEfforts]:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort_str}. Valid options are: {', '.join([effort.value for effort in ReasoningEfforts])}")
        return reasoning_effort_str

    @staticmethod
    def parse_embedding(embedding_str: str | None) -> tuple[str | None, str | None]:
        """Parse embedding string into (embedding_provider, embedding_model)."""
        from gpt_researcher.embeddings import _SUPPORTED_PROVIDERS

        if embedding_str is None:
            return None, None
        try:
            embedding_provider, embedding_model = embedding_str.split(":", 1)
            assert embedding_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {embedding_provider}.\nSupported embedding providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return embedding_provider, embedding_model
        except ValueError:
            raise ValueError(
                "Set EMBEDDING = '<embedding_provider>:<embedding_model>' "
                "Eg 'openai:text-embedding-3-large'"
            )

    def validate_doc_path(self):
        """Ensure that the folder exists at the doc path"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
        """Convert environment variable to the appropriate type based on the type hint."""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # Handle Union types (e.g., Union[str, None] / Optional[str]).
            if type(None) in args and env_value.lower() in ("none", "null", ""):
                return None
            for arg in args:
                if arg is type(None):
                    continue
                try:
                    return Config.convert_env_value(key, env_value, arg)
                except ValueError:
                    continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif type_hint is list or origin is list or origin is List:
            try:
                value = json_repair.loads(env_value)
            except Exception as exc:
                raise ValueError(f"Cannot convert {env_value} to list") from exc
            if not isinstance(value, list):
                raise ValueError(f"Cannot convert {env_value} to list")
            return value
        elif type_hint is dict:
            try:
                value = json_repair.loads(env_value)
            except Exception as exc:
                raise ValueError(f"Cannot convert {env_value} to dict") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Cannot convert {env_value} to dict")
            return value
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbosity level."""
        self.llm_kwargs["verbose"] = verbose
