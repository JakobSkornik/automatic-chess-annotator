import unittest

from app.core.commentary.pipeline.game_pipeline import GameAnnotationPipeline


class TestGameAnnotationPipeline(unittest.TestCase):
    def test_instantiates(self) -> None:
        p = GameAnnotationPipeline()
        self.assertTrue(hasattr(p, "run_llm_phases"))


if __name__ == "__main__":
    unittest.main()
