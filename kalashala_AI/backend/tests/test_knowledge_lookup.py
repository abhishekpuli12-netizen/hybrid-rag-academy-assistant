import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_lookup import lookup_course_fee, read_markdown_table


class KnowledgeLookupTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.fees_file = Path(self.tempdir.name) / "fees.md"
        self.fees_file.write_text(
            "# Course Fees\n\n"
            "| Course | Fee |\n|---|---|\n"
            "| Professional Makeup Course | ₹35,000 |\n"
            "| Hair Styling Course | ₹18,000 |\n"
            "| Bridal Styling Master Class | Contact Academy |\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_returns_exact_course_fee_from_markdown(self):
        result = lookup_course_fee(
            "What are the fees for the Professional Makeup Course?", self.fees_file
        )
        self.assertIsNotNone(result)
        self.assertIn("₹35,000", result["answer"])
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["source"]["file"], "fees.md")

    def test_lists_fees_when_user_asks_for_all_courses(self):
        result = lookup_course_fee("List all course fees", self.fees_file)
        self.assertIsNotNone(result)
        self.assertEqual(result["matched_rows"], 3)
        self.assertIn("Contact Academy", result["answer"])

    def test_unknown_course_does_not_return_an_unrelated_fee(self):
        result = lookup_course_fee("What is the pottery course fee?", self.fees_file)
        self.assertIsNone(result)

    def test_non_fee_question_uses_normal_retrieval(self):
        self.assertIsNone(lookup_course_fee("What courses do you offer?", self.fees_file))

    def test_markdown_table_parser_skips_header_and_separator(self):
        self.assertEqual(
            read_markdown_table(self.fees_file),
            [
                ("Professional Makeup Course", "₹35,000"),
                ("Hair Styling Course", "₹18,000"),
                ("Bridal Styling Master Class", "Contact Academy"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
