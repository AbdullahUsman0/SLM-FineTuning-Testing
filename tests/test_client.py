import io
import json
import unittest
from unittest.mock import patch

from local_slm_lab.client import LocalModelConfig, LocalModelError, chat_completion


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ClientTests(unittest.TestCase):
    @patch("local_slm_lab.client.request.urlopen")
    def test_returns_assistant_content(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "What forecast horizon?"}}]}
        )
        answer = chat_completion(
            [{"role": "user", "content": "Forecast sales"}], LocalModelConfig()
        )
        self.assertEqual(answer, "What forecast horizon?")

    @patch("local_slm_lab.client.request.urlopen")
    def test_rejects_malformed_server_response(self, urlopen):
        urlopen.return_value = FakeResponse({"unexpected": True})
        with self.assertRaises(LocalModelError):
            chat_completion(
                [{"role": "user", "content": "Forecast sales"}], LocalModelConfig()
            )

    @patch("local_slm_lab.client.request.urlopen")
    def test_rejects_empty_content(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]}
        )
        with self.assertRaisesRegex(LocalModelError, "reasoning off"):
            chat_completion(
                [{"role": "user", "content": "Forecast sales"}], LocalModelConfig()
            )

    @patch("local_slm_lab.client.request.urlopen")
    def test_posts_to_local_chat_completions_endpoint(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "OK"}}]}
        )
        chat_completion(
            [{"role": "user", "content": "Hello"}],
            LocalModelConfig(base_url="http://127.0.0.1:9999/v1"),
        )
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.full_url,
            "http://127.0.0.1:9999/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
