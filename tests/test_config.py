"""Tests for core.config — configuration loading, saving, and validation."""

import json
import os

import pytest

from core.config import load_config, save_config, validate_api_keys


# ═══════════════════════════════════════════════════════════════════════════════
# load_config
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadConfig:
    """Configuration loading from JSON files."""

    def test_loads_valid_config(self, tmp_path) -> None:
        config_data = {"model": "gpt-4o-mini", "rate_limit": {"max_requests": 10}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config(str(config_file))
        assert result["model"] == "gpt-4o-mini"
        assert result["rate_limit"]["max_requests"] == 10

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_config(str(bad_file))


# ═══════════════════════════════════════════════════════════════════════════════
# save_config
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveConfig:
    """Configuration saving to JSON files."""

    def test_saves_and_roundtrips(self, tmp_path) -> None:
        config_data = {"model": "gpt-4o", "chunk_size": 1000}
        config_file = tmp_path / "config.json"

        save_config(config_data, str(config_file))
        loaded = load_config(str(config_file))

        assert loaded == config_data

    def test_overwrites_existing_file(self, tmp_path) -> None:
        config_file = tmp_path / "config.json"

        save_config({"version": 1}, str(config_file))
        save_config({"version": 2}, str(config_file))

        loaded = load_config(str(config_file))
        assert loaded["version"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# validate_api_keys
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateApiKeys:
    """API key validation using environment variables."""

    def test_valid_keys_pass(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        monkeypatch.setenv("SERPAPI_API_KEY", "some-serpapi-key")
        # Should not raise.
        validate_api_keys()

    def test_missing_openai_key_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set"):
            validate_api_keys()

    def test_invalid_openai_key_prefix_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "bad-prefix-key")
        with pytest.raises(EnvironmentError, match="appears invalid"):
            validate_api_keys()

    def test_missing_serpapi_key_warns_but_passes(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        # Should not raise — SERPAPI is optional.
        validate_api_keys()

    def test_langsmith_tracing_enabled_with_key(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        # Should not raise — both tracing vars are set.
        validate_api_keys()

    def test_langsmith_tracing_enabled_without_key_warns(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        # Should not raise, but should log a warning.
        import logging
        with caplog.at_level(logging.WARNING, logger="core.config"):
            validate_api_keys()
        assert "LANGCHAIN_API_KEY is not set" in caplog.text

    def test_langsmith_tracing_disabled_no_warning(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        # No warning about LangSmith when tracing is not enabled.
        import logging
        with caplog.at_level(logging.WARNING, logger="core.config"):
            validate_api_keys()
        assert "LANGCHAIN_API_KEY" not in caplog.text
