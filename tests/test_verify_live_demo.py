import unittest

from scripts.verify_live_demo import _join


class VerifyLiveDemoTests(unittest.TestCase):
    def test_join_handles_missing_and_extra_slashes(self):
        self.assertEqual(_join("https://example.com", "/health"), "https://example.com/health")
        self.assertEqual(_join("https://example.com/", "health"), "https://example.com/health")


if __name__ == "__main__":
    unittest.main()
