import unittest

from src.data import label_from_training_filename


class TrainingLabelTests(unittest.TestCase):
    def test_pathological_myopia_is_positive(self):
        self.assertEqual(label_from_training_filename("P0001.jpg"), 1)

    def test_normal_and_high_myopia_are_non_pm(self):
        self.assertEqual(label_from_training_filename("N0001.jpg"), 0)
        self.assertEqual(label_from_training_filename("H0001.jpg"), 0)

    def test_unknown_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            label_from_training_filename("X0001.jpg")


if __name__ == "__main__":
    unittest.main()
