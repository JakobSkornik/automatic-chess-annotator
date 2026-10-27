import unittest
from unittest.mock import patch

from app.core.commentary.llm_policy import resolve_model


class TestLlmPolicy(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_frontier_override_applies_to_every_pass(self):
        with patch.dict("os.environ", {"COMMENTARY_OPENAI_MODEL": "gpt-6-astra", "COMMENTARY_GROK_MODEL": "grok-4.7"}):
            for label in (None, "key_moment", "teaching"):
                self.assertEqual(resolve_model("openai", "composer", label), "gpt-6-astra")
                self.assertEqual(resolve_model("grok", "composer", label), "grok-4.7")

    def test_wrong_provider_override_is_not_silently_used(self):
        with patch.dict("os.environ", {"COMMENTARY_OPENAI_MODEL": "grok-4.6"}):
            with self.assertRaises(ValueError):
                resolve_model("openai", "composer")

    def test_resolve_openai(self) -> None:
        self.assertEqual(resolve_model("openai", "composer"), "gpt-6-luna")

    def test_resolve_openai_composer_by_pass_label(self) -> None:
        self.assertEqual(
            resolve_model("openai", "composer", pass_label="key_moment"),
            "gpt-6-luna",
        )
        self.assertEqual(
            resolve_model("openai", "composer", pass_label="teaching"),
            "gpt-6-luna",
        )

    def test_anthropic_is_not_a_policy_and_falls_back_to_openai(self) -> None:
        self.assertEqual(resolve_model("anthropic", "composer"), "gpt-6-luna")

    def test_resolve_grok(self) -> None:
        self.assertEqual(resolve_model("grok", "composer"), "grok-4.6")

    def test_resolve_cursor(self) -> None:
        self.assertEqual(resolve_model("cursor", "composer"), "composer-2.5")

    def test_resolve_cursor_composer_pass_label_uses_composer_25(self) -> None:
        self.assertEqual(
            resolve_model("cursor", "composer", pass_label="key_moment"),
            "composer-2.5",
        )

    def test_unknown_provider_defaults_openai(self) -> None:
        self.assertEqual(resolve_model("unknown", "composer"), "gpt-6-luna")


if __name__ == "__main__":
    unittest.main()
