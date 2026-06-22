import sys
import types
from unittest.mock import MagicMock

import pytest

from src.llm_client import LLMClient, LLMError, _openai_clients
from src.prompt_builder import TRAINING_SYSTEM_PROMPT


@pytest.fixture(autouse=True)
def clear_openai_client_cache():
    _openai_clients.clear()
    yield
    _openai_clients.clear()


@pytest.fixture
def local_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_MODEL", "realpage-message-agent-v1")
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_local_provider_uses_model_name(local_env):
    client = LLMClient(mock=False)
    assert client.provider == "local"
    assert client.model_name == "realpage-message-agent-v1"


def test_provider_override(local_env):
    client = LLMClient(mock=False, provider="openai")
    assert client.provider == "openai"
    assert client.model_name == "gpt-4o-mini"


def test_local_provider_requires_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = LLMClient(mock=False)
    with pytest.raises(LLMError, match="LOCAL_BASE_URL"):
        client.generate('{"task":"test"}')


def test_openai_ignores_hf_base_url(monkeypatch):
    hf_url = "https://abc.eu-west-1.aws.endpoints.huggingface.cloud/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", hf_url)
    monkeypatch.setenv("LOCAL_BASE_URL", hf_url)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"should_send": false}'))]
    mock_client.chat.completions.create.return_value = mock_response

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = MagicMock(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    client = LLMClient(mock=False, provider="openai")
    result = client.generate('{"task":"test"}')

    assert result == {"should_send": False}
    openai_module.OpenAI.assert_called_once_with(api_key="sk-test")


def test_local_provider_calls_openai_compatible_endpoint(local_env, monkeypatch):
    monkeypatch.delenv("LOCAL_MAX_TOKENS", raising=False)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"should_send": false}'))]
    mock_client.chat.completions.create.return_value = mock_response

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = MagicMock(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    client = LLMClient(mock=False)
    result = client.generate('{"task":"test"}')

    assert result == {"should_send": False}
    openai_module.OpenAI.assert_called_once_with(
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "realpage-message-agent-v1"
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["messages"][0]["content"] == TRAINING_SYSTEM_PROMPT
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_openai_defaults_max_tokens_to_512(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"should_send": false}'))]
    mock_client.chat.completions.create.return_value = mock_response

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = MagicMock(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    client = LLMClient(mock=False, provider="openai")
    client.generate('{"task":"test"}')

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 512
