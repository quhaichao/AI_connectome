import io
import time
import unittest
from contextlib import redirect_stdout

from fc_pruning.progress import report_progress
from scripts.run_ffn_ratio_matrix import _tee_child_output


class ProgressTests(unittest.TestCase):
    def test_progress_bar_reports_percentage_and_eta(self):
        output = io.StringIO()
        with redirect_stdout(output):
            report_progress(
                "Test stage",
                5,
                10,
                time.monotonic() - 5,
            )
        rendered = output.getvalue()
        self.assertIn("Test stage", rendered)
        self.assertIn("5/10", rendered)
        self.assertIn("50.00%", rendered)
        self.assertIn("eta=", rendered)

    def test_scheduler_tees_child_output_to_log_and_terminal(self):
        stream = io.StringIO("first\nsecond\n")
        log = io.StringIO()
        terminal = io.StringIO()
        with redirect_stdout(terminal):
            _tee_child_output(stream, log, "seed8")
        self.assertEqual(log.getvalue(), "first\nsecond\n")
        self.assertEqual(
            terminal.getvalue(),
            "[seed8] first\n[seed8] second\n",
        )


if __name__ == "__main__":
    unittest.main()
