"""A test for some attack classes."""

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

import sys

sys.path.append("../..")
from prive.datasets import TabularDataset, TabularRecord
from prive.datasets.data_description import DataDescription
from prive.threat_models import (
    TargetedMIA,
    TargetedAIA,
    AuxiliaryDataKnowledge,
    BlackBoxKnowledge,
)
from prive.generators import Raw

# The classes being tested.
from prive.attacks import (
    ClosestDistanceAttack,
    GroundhogAttack,
    NaiveSetFeature,
    HistSetFeature,
    FeatureBasedSetClassifier,
)

from sklearn.linear_model import LogisticRegression

## Test for closest-distance.

dummy_data_description = DataDescription(
    [
        {"name": "a", "type": "countable", "representation": "integer"},
        {"name": "b", "type": "countable", "representation": "integer"},
    ]
)

dummy_data = pd.DataFrame([(0, 1), (0, 2), (3, 4), (3, 5)], columns=["a", "b"])


class TestClosestDistance(TestCase):
    """Test the closest-distance attack."""

    def setUp(self):
        self.dataset = TabularDataset(dummy_data, dummy_data_description)

    def _make_mia(self, a, b):
        """Helper function to generate a MIA threat model."""
        return TargetedMIA(
            AuxiliaryDataKnowledge(self.dataset, sample_real_frac=0.5),
            self._make_target(a, b),
            None,
        )

    def _make_target(self, a, b):
        """Helper function to generate a target record."""
        return TabularDataset(
            pd.DataFrame([(a, b)], columns=["a", "b"]), dummy_data_description
        )

    def test_dummy(self):
        # Check whether the attack works on a dummy dataset,
        #  with a specified threshold.

        # Take a record that is not in the dataset (distance 1/2).
        mia = self._make_mia(0, 0)
        attack = ClosestDistanceAttack(threshold=0.3)
        attack.train(mia)
        # Check that the training worked as intended.
        self.assertEqual(attack.trained, True)
        # Check that the score is working as intended.
        scores = attack.attack_score([rec for rec in self.dataset])
        self.assertEqual(len(scores), len(self.dataset))
        for score, distance in zip(scores, [0.5, 0.5, 1, 1]):
            self.assertEqual(score, distance)
        # Assert that the total score and decisions are ok.
        self.assertEqual(attack.attack_score([self.dataset])[0], 0.5)
        self.assertEqual(attack.attack([self.dataset])[0], False)

        # Perform the attack for a user *in* the dataset.
        attack = ClosestDistanceAttack(threshold=0.3)
        attack.train(self._make_mia(0, 1))
        self.assertEqual(attack.attack([self.dataset])[0], True)

    def test_training(self):
        # Check that the threshold selection works.
        # This merely checks that the code runs, not that it is correct.
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                self.dataset, sample_real_frac=0.5, num_training_records=2
            ),
            self._make_target(0, 4),
            BlackBoxKnowledge(generator=Raw(), num_synthetic_records=2),
            replace_target=True,
        )
        attack_tpr = ClosestDistanceAttack(tpr=0.1)
        attack_tpr.train(mia)
        attack_fpr = ClosestDistanceAttack(fpr=0.1)
        attack_fpr.train(mia)


## Test for features.


class TestSetFeatures(TestCase):
    """Test whether the set features defined for Groundhog are implemented correctly."""

    def test_naive(self):
        """Test that the naive features work properly."""
        num_records = 20
        num_datasets = 10
        num_finite = 2
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "real", "representation": "number"},
                {"name": "c", "type": "finite", "representation": num_finite},
            ]
        )
        real_data = [
            np.concatenate(
                (
                    np.random.random(size=(num_records, 2)),
                    np.random.randint(num_finite, size=(num_records, 1)),
                ),
                axis=1,
            )
            for _ in range(num_datasets)
        ]
        datasets = [
            TabularDataset(
                pd.DataFrame(data, columns=["a", "b", "c"]), data_description
            )
            for data in real_data
        ]
        feature = NaiveSetFeature()
        values = feature(datasets)
        # Check that it has the proper shape.
        self.assertEqual(values.shape, (num_datasets, 3 * (2 + num_finite)))
        # Check that it is correct (for continuous variables only).
        # This feature set starts with means for all variables (finite vars are
        # one-hot encoded), then medians and finally variances.
        for data, val in zip(real_data, values):
            print(val)
            self.assertAlmostEqual(data[:, 0].mean(axis=0), val[0])
            self.assertAlmostEqual(data[:, 1].mean(axis=0), val[1])
            self.assertAlmostEqual(np.median(data[:, 0], axis=0), val[2 + num_finite])
            self.assertAlmostEqual(
                np.median(data[:, 1], axis=0), val[2 + num_finite + 1]
            )
            self.assertAlmostEqual(data[:, 0].var(axis=0), val[2 * (2 + num_finite)])
            self.assertAlmostEqual(
                data[:, 1].var(axis=0), val[2 * (2 + num_finite) + 1]
            )

    def test_histogram(self):
        """Test that the histogram features work properly."""
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "finite", "representation": ["x", "y", "z"]},
            ]
        )
        data1 = pd.DataFrame(
            [(0.1, "x"), (0.9, "y"), (0.7, "x"), (0.9, "z")], columns=["a", "b"]
        )
        data2 = pd.DataFrame([(0.5, "z")], columns=["a", "b"])
        feature = HistSetFeature(num_bins=5, bounds=(0, 1))
        histograms = feature(
            [
                TabularDataset(data1, data_description),
                TabularDataset(data2, data_description),
            ]
        )
        self.assertEqual(histograms.shape, (2, 8))
        # Bins (0,.2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)
        expected_answers = np.array(
            [
                [1 / 4, 0, 0, 1 / 4, 2 / 4, 2 / 4, 1 / 4, 1 / 4],
                [0, 0, 1, 0, 0, 0, 0, 1],
            ]
        )
        # Check that the features are the proper answer.
        for computed, expected in zip(histograms.flatten(), expected_answers.flatten()):
            self.assertEqual(computed, expected)

    def test_combination(self):
        """Test whether combining feature maps works."""
        data_description = DataDescription(
            [
                {"name": "a", "type": "real", "representation": "number"},
                {"name": "b", "type": "finite", "representation": ["x", "y", "z"]},
            ]
        )
        dataset = TabularDataset(
            pd.DataFrame(
                [(0.1, "x"), (0.9, "y"), (0.7, "x"), (0.9, "z")], columns=["a", "b"]
            ),
            data_description,
        )
        num_bins = 10
        feature = NaiveSetFeature() + HistSetFeature(num_bins=num_bins, bounds=(0, 1))
        result = feature([dataset])
        # We only test whether the size of the output is correct.
        # We assume the content is correct, from other tests.
        num_continuous = 1
        discrete_1hot = 3
        self.assertEqual(
            result.shape,
            (
                1,
                3 * (num_continuous + discrete_1hot)
                + num_bins * num_continuous
                + discrete_1hot,
            ),
        )


## Test for the Groundhog attack.


class TestGroundHog:
    """Test whether the groundhog attack (Stadler et al.) works."""

    def test_groundhog_runs(self):
        """Test whether the Groundhog attack runs."""
        values = ["x", "y", "z"]
        total_dataset = TabularDataset(
            pd.DataFrame(
                [
                    (np.random.random(), values[np.random.randint(3)])
                    for _ in range(100)
                ],
                columns=["a", "b"],
            ),
            DataDescription(
                [
                    {"name": "a", "type": "real", "representation": "number"},
                    {"name": "b", "type": "finite", "representation": values},
                ]
            ),
        )
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                total_dataset, sample_real_frac=0.5, num_training_records=20
            ),
            total_dataset.sample(1),  # Random target.
            BlackBoxKnowledge(Raw(), num_synthetic_records=10),
        )
        attack = GroundhogAttack(
            FeatureBasedSetClassifier(
                NaiveSetFeature() + HistSetFeature(num_bins=10, bounds=(0, 1)),
                LogisticRegression(),
            )
        )
        attack.train(mia)


if __name__ == "__main__":
    unittest.main()
