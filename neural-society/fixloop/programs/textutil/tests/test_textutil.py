import unittest

from textutil import is_palindrome, title_case, truncate, word_count


class TestText(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(word_count("The cat. the CAT!"), {"the": 2, "cat": 2})

    def test_title_case(self):
        self.assertEqual(title_case("the lord of the rings"), "The Lord of the Rings")

    def test_palindrome(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertFalse(is_palindrome("hello"))

    def test_truncate(self):
        self.assertEqual(truncate("hello world", 8), "hello...")
        self.assertEqual(truncate("short", 10), "short")


if __name__ == "__main__":
    unittest.main()
