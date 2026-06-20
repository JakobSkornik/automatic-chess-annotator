import unittest

from app.core.commentary.llm_policy import resolve_model


class TestLlmPolicy(unittest.TestCase):
    def test_resolve_openai(self) -> None:
        self.assertEqual(resolve_model("openai", "composer"), "gpt-4.1-mini")

    def test_resolve_openai_composer_by_pass_label(self) -> None:
        self.assertEqual(
            resolve_model("openai", "composer", pass_label="key_moment"),
            "gpt-4.1",
        )
        self.assertEqual(
            resolve_model("openai", "composer", pass_label="teaching"),
            "gpt-4.1-mini",
        )

    def test_resolve_anthropic(self) -> None:
        self.assertEqual(resolve_model("anthropic", "composer"), "claude-haiku-4-5")

    def test_resolve_cursor(self) -> None:
        self.assertEqual(resolve_model("cursor", "composer"), "composer-2.5")

    def test_resolve_cursor_composer_pass_label_uses_composer_25(self) -> None:
        self.assertEqual(
            resolve_model("cursor", "composer", pass_label="key_moment"),
            "composer-2.5",
        )

    def test_unknown_provider_defaults_openai(self) -> None:
        self.assertEqual(resolve_model("unknown", "composer"), "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
