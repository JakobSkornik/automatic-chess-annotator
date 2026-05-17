import unittest

from app.core.commentary.llm_policy import resolve_model


class TestLlmPolicy(unittest.TestCase):
    def test_resolve_openai(self) -> None:
        self.assertEqual(resolve_model("openai", "composer"), "gpt-4.1-mini")
        self.assertEqual(resolve_model("openai", "digest"), "gpt-4.1")
        self.assertEqual(resolve_model("openai", "narrative"), "gpt-4.1")

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
        self.assertEqual(resolve_model("anthropic", "digest"), "claude-sonnet-4-5")
        self.assertEqual(resolve_model("anthropic", "narrative"), "claude-sonnet-4-5")

    def test_unknown_provider_defaults_openai(self) -> None:
        self.assertEqual(resolve_model("unknown", "episode"), "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
