
import unittest

from ddp_rarity_sampling.scripts.rarity_sampler import RarityAwareSampler


class TestSamplerValidation(unittest.TestCase):

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            RarityAwareSampler(None, batch_size=0)

    def test_too_many_rare_images(self):
        with self.assertRaises(ValueError):
            RarityAwareSampler(
                None, batch_size=16, rare_per_batch=20
            )

    def test_negative_rare_count(self):
        with self.assertRaises(ValueError):
            RarityAwareSampler(
                None, batch_size=16, rare_per_batch=-1
            )

    def test_invalid_gpu_count(self):
        with self.assertRaises(ValueError):
            RarityAwareSampler(
                None, batch_size=16, num_replicas=0
            )

    def test_invalid_gpu_rank(self):
        with self.assertRaises(ValueError):
            RarityAwareSampler(
                None, batch_size=16, num_replicas=2, rank=2
            )


if __name__ == "__main__":
    unittest.main()