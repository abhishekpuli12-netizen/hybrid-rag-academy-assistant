import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_lookup import (
    lookup_admissions,
    lookup_academy_fact,
    lookup_course_fee,
    lookup_courses,
    lookup_course_details,
    lookup_trainers,
    read_markdown_table,
)


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

    def test_lists_courses_from_the_knowledge_file(self):
        courses_file = Path(self.tempdir.name) / "courses.md"
        courses_file.write_text(
            "# Courses Offered\n\n"
            "- Professional Makeup Course\n"
            "- Hair Styling Course\n\n---\n\n"
            "# Course Details\n- Not an offered course\n",
            encoding="utf-8",
        )
        result = lookup_courses("What courses do you offer?", courses_file)
        self.assertIsNotNone(result)
        self.assertIn("Professional Makeup Course", result["answer"])
        self.assertNotIn("Not an offered course", result["answer"])

    def test_returns_trainer_information_from_knowledge_file(self):
        trainers_file = Path(self.tempdir.name) / "trainers.md"
        trainers_file.write_text("# Trainers\nCIDESCO Certified Trainers.", encoding="utf-8")
        result = lookup_trainers("Are your trainers CIDESCO certified?", trainers_file)
        self.assertIsNotNone(result)
        self.assertIn("CIDESCO Certified Trainers", result["answer"])

    def test_returns_admissions_contact_from_knowledge_file(self):
        contacts_file = Path(self.tempdir.name) / "contacts.md"
        contacts_file.write_text("Contact admissions at example.test", encoding="utf-8")
        result = lookup_admissions("How do I enroll?", contacts_file)
        self.assertIsNotNone(result)
        self.assertIn("example.test", result["answer"])

    def test_duration_question_does_not_trigger_course_catalog(self):
        self.assertIsNone(
            lookup_courses(
                "What is the duration of the Weekend Basic to Advance Makeup Course?",
                self.fees_file,
            )
        )

    def test_returns_duration_for_the_named_course(self):
        courses_file = Path(self.tempdir.name) / "courses.md"
        courses_file.write_text(
            "# Courses Offered\n- Weekend Basic to Advance Makeup Course\n\n---\n\n"
            "# Weekend Basic to Advance Makeup Course\n## Course Details\n- **Duration:** 8 Weeks\n",
            encoding="utf-8",
        )
        result = lookup_course_details(
            "What is the duration of the Weekend Basic to Advance Makeup Course?",
            courses_file,
        )
        self.assertIsNotNone(result)
        self.assertIn("8 Weeks", result["answer"])

    def test_returns_hostel_status_from_academy_file(self):
        academy_file = Path(self.tempdir.name) / "academy.md"
        academy_file.write_text("# Hostel\nNot Available\n", encoding="utf-8")
        result = lookup_academy_fact("Is hostel available?", Path(self.tempdir.name))
        self.assertIsNotNone(result)
        self.assertIn("Not Available", result["answer"])


if __name__ == "__main__":
    unittest.main()
