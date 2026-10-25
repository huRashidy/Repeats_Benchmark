"""Unit tests for repeat_benchmark pipeline components."""

import unittest
import pandas as pd
from src.prompts import get_class_prompt, get_family_prompt
from src.normalize import normalize_label, clean_raw_text
from src.evaluate import compute_metrics


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.classes = ["DNA", "LTR", "LINE", "SINE", "Satellite", "rRNA"]
        self.families = ["Alu", "MIR", "L1", "Gypsy", "TcMar-Tigger"]

    def test_prompts(self):
        seq = "ATGCGATCGATCGATCG"
        p_class = get_class_prompt(seq, self.classes)
        self.assertIn("<dna>ATGCGATCGATCGATCG</dna>", p_class)
        self.assertIn("Which repeat class", p_class)
        self.assertIn("DNA, LINE, LTR", p_class)

        p_fam = get_family_prompt(seq, self.families)
        self.assertIn("<dna>ATGCGATCGATCGATCG</dna>", p_fam)
        self.assertIn("Which repeat family", p_fam)

    def test_clean_raw_text(self):
        self.assertEqual(clean_raw_text("<think>reasoning</think>DNA"), "DNA")
        self.assertEqual(clean_raw_text("**LINE**."), "LINE")
        self.assertEqual(clean_raw_text("The answer is: SINE"), "SINE")
        self.assertEqual(clean_raw_text("<dna>Alu</dna>"), "Alu")

    def test_normalize_label(self):
        # Exact match
        label, status = normalize_label("DNA", self.classes)
        self.assertEqual(label, "DNA")
        self.assertEqual(status, "exact")

        # Case-insensitive
        label, status = normalize_label("line", self.classes)
        self.assertEqual(label, "LINE")
        self.assertEqual(status, "case_insensitive")

        # Punctuation / bold
        label, status = normalize_label("**LTR**.", self.classes)
        self.assertEqual(label, "LTR")

        # Fuzzy match
        label, status = normalize_label("Sattelite", self.classes)
        self.assertEqual(label, "Satellite")
        self.assertEqual(status, "fuzzy")

        # Invalid
        label, status = normalize_label("Unicorn", self.classes)
        self.assertEqual(label, "INVALID")
        self.assertEqual(status, "invalid")

    def test_compute_metrics(self):
        df = pd.DataFrame({
            "true_label": ["DNA", "DNA", "LTR", "LINE", "SINE"],
            "pred_label": ["DNA", "LTR", "LTR", "LINE", "INVALID"],
        })
        m = compute_metrics(df)
        self.assertEqual(m["total_samples"], 5)
        self.assertEqual(m["invalid_predictions"], 1)
        self.assertAlmostEqual(m["accuracy"], 3 / 5)
        self.assertIn("f1_macro", m)
        self.assertIn("mcc", m)


if __name__ == "__main__":
    unittest.main()
