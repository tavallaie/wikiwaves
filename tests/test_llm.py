"""Tests for the shared LLMClient."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import responses

from wikiwaves.llm import LLMClient, LLMError


class TestLLMClientEnvVars(unittest.TestCase):
    """LLMClient must read configuration from environment variables."""

    def test_reads_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "sk-secret"}, clear=True):
            client = LLMClient(env_path="/nonexistent")
            self.assertEqual(client.api_key, "sk-secret")

    def test_reads_base_url_from_env(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:1234/v1"}, clear=True):
            client = LLMClient(env_path="/nonexistent")
            self.assertEqual(client.base_url, "http://localhost:1234/v1")

    def test_reads_model_from_env(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL": "llama3.2"}, clear=True):
            client = LLMClient(env_path="/nonexistent")
            self.assertEqual(client.model, "llama3.2")

    def test_explicit_args_override_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "env-key",
                "LLM_BASE_URL": "http://env-host/v1",
                "LLM_MODEL": "env-model",
            },
            clear=True,
        ):
            client = LLMClient(
                api_key="arg-key",
                base_url="http://arg-host/v1",
                model="arg-model",
                env_path="/nonexistent",
            )
            self.assertEqual(client.api_key, "arg-key")
            self.assertEqual(client.base_url, "http://arg-host/v1")
            self.assertEqual(client.model, "arg-model")

    def test_defaults_when_nothing_provided(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = LLMClient(env_path="/nonexistent")
            self.assertIsNone(client.api_key)
            self.assertEqual(client.base_url, "https://api.openai.com/v1")
            self.assertEqual(client.model, "gpt-4o-mini")

    def test_base_url_strips_trailing_slash(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://host/v1/"}, clear=True):
            client = LLMClient(env_path="/nonexistent")
            self.assertEqual(client.base_url, "http://host/v1")


class TestLLMClientChat(unittest.TestCase):
    """LLMClient.chat() behaviour."""

    @responses.activate
    def test_chat_sends_correct_payload(self):
        responses.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello!"}}]
            },
            status=200,
        )

        client = LLMClient(api_key="test-key")
        result = client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "Hello!")
        req = responses.calls[0].request
        payload = __import__("json").loads(req.body)
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(req.headers["Authorization"], "Bearer test-key")

    @responses.activate
    def test_chat_no_auth_when_no_api_key(self):
        responses.post(
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "ok"}}]},
            status=200,
        )

        client = LLMClient()
        client.chat([{"role": "user", "content": "Hi"}])

        req = responses.calls[0].request
        self.assertNotIn("Authorization", req.headers)

    @responses.activate
    def test_chat_raises_on_network_error(self):
        responses.post(
            "https://api.openai.com/v1/chat/completions",
            body=__import__("requests").ConnectionError("Refused"),
        )

        client = LLMClient()
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("Refused", str(ctx.exception))

    @responses.activate
    def test_chat_raises_on_malformed_response(self):
        responses.post(
            "https://api.openai.com/v1/chat/completions",
            json={"unexpected": "shape"},
            status=200,
        )

        client = LLMClient()
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("Could not parse LLM response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
