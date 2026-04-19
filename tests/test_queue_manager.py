import asyncio
import unittest

from app.core.queue_manager import QueueManager, _extract_pgn_snapshot
from app.models.job import JobStatus


MIN_PGN = """[Event "T"]
[White "A"]
[Black "B"]
[Result "*"]

1. e4 e5"""


class TestQueueManager(unittest.TestCase):
    def test_extract_pgn_snapshot(self):
        meta, n = _extract_pgn_snapshot(MIN_PGN)
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta.whiteName, "A")
        self.assertEqual(meta.blackName, "B")
        self.assertEqual(n, 2)

    def test_list_jobs_ordering_and_retry(self):
        async def run():
            qm = QueueManager()
            id1 = await qm.add_job(MIN_PGN, llm_model="gpt-5.4", llm_effort="medium")
            id2 = await qm.add_job(MIN_PGN)
            lst = qm.list_jobs(10)
            self.assertEqual(len(lst), 2)
            self.assertEqual(lst[0].job_id, id2)
            self.assertEqual(lst[1].job_id, id1)
            qm.mark_failed(id2, "boom")
            st = qm.get_job_status(id2)
            self.assertIsNotNone(st)
            assert st is not None
            self.assertEqual(st.status, JobStatus.FAILED)
            self.assertEqual(st.error, "boom")
            id3 = await qm.add_job(qm.get_job_data(id2)["pgn"], llm_model="gpt-5.4", llm_effort="medium")
            self.assertNotEqual(id3, id2)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
