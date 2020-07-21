import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Iterable, Tuple, Dict
from unittest import TestCase
from easytune.TuneKeras import TuneKeras
import numpy as np
import tensorflow as tf


def simple_numpy_generator() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple generator, yields numpy ndarrays
    :return:
    """
    for i in range(0, 100):
        features = np.zeros((8, 32), dtype=np.float32) + i
        labels = np.zeros((8, 4), dtype=np.int32) + i
        yield features, labels

def simple_dict_generator() -> Iterable[Tuple[Dict[str,np.ndarray], Dict[str, np.ndarray]]]:
    """
    Simple generator, yields numpy ndarrays
    :return:
    """
    for i in range(0, 100):
        features = np.zeros((8, 32), dtype=np.float32) + i
        labels = np.zeros((8, 4), dtype=np.int32) + i
        yield {'features_A': features, 'features_B': features}, {'labels': labels}


class Test(TestCase):
    def test_dataset_builder_1(self):
        """
        This test checks NumPy -> TF Dataset conversionv (np version)
        """
        kgs = TuneKeras({'lr': [0.001, 0.002, 0.003]}, None, 0)
        dts = kgs._TuneKeras__build_tf_dataset(simple_numpy_generator)
        self.assertTrue(isinstance(dts, tf.data.Dataset))

        cnt = 0
        for ds in dts.as_numpy_iterator():
            self.assertEqual(2, len(ds))
            self.assertEqual(np.float32, ds[0].dtype)
            self.assertEqual(np.int32, ds[1].dtype)

            self.assertEqual((8, 32), ds[0].shape)
            self.assertEqual((8, 4), ds[1].shape)
            cnt += 1

        self.assertEqual(100, cnt)

    def test_dataset_builder_2(self):
        """
        This test checks NumPy -> TF Dataset conversionv (dict version)
        """
        kgs = TuneKeras({'lr': [0.001, 0.002, 0.003]}, None, 0)
        dts = kgs._TuneKeras__build_tf_dataset(simple_dict_generator)
        self.assertTrue(isinstance(dts, tf.data.Dataset))

        cnt = 0
        for ds in dts.as_numpy_iterator():
            self.assertEqual(2, len(ds))
            self.assertEqual(np.float32, ds[0]['features_A'].dtype)
            self.assertEqual(np.float32, ds[0]['features_B'].dtype)
            self.assertEqual(np.int32, ds[1]['labels'].dtype)

            self.assertEqual((8, 32), ds[0]['features_A'].shape)
            self.assertEqual((8, 32), ds[0]['features_B'].shape)
            self.assertEqual((8, 4), ds[1]['labels'].shape)
            cnt += 1

        self.assertEqual(100, cnt)
