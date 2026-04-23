"""Per-category composer prompt map covers all MoveCategory values."""

import unittest

from app.core.commentary.advanced_comment_service import CATEGORY_COMPOSER_PROMPTS
from app.models.chess_events import MoveCategory


class TestPerCategoryPrompts(unittest.TestCase):
    def test_all_categories_mapped(self) -> None:
        for c in MoveCategory:
            self.assertIn(c.value, CATEGORY_COMPOSER_PROMPTS)
            prompt = CATEGORY_COMPOSER_PROMPTS[c.value]
            self.assertIn("named_motifs", prompt.lower())
            self.assertIn("plain prose", prompt.lower())


if __name__ == "__main__":
    unittest.main()
