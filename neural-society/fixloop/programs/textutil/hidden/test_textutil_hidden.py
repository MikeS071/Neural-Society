import unittest

from textutil import is_palindrome, title_case, truncate, word_count


class TestTextHidden(unittest.TestCase):
    def test_word_count_empty(self):
        self.assertEqual(word_count("  ... "), {})

    def test_title_case_first_small(self):
        self.assertEqual(title_case("a tale OF two cities"), "A Tale of Two Cities")

    def test_palindrome_empty(self):
        self.assertTrue(is_palindrome(""))

    def test_truncate_exact(self):
        self.assertEqual(truncate("abcdef", 6), "abcdef")
        self.assertEqual(truncate("abcdefg", 6), "abc...")

    def test_truncate_tiny_limit(self):
        self.assertEqual(truncate("abcdef", 2), "..")


if __name__ == "__main__":
    unittest.main()
