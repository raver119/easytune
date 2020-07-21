"""
Tests suit for GridBuilder
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from unittest import TestCase
from easytune.GridBuilder import GridBuilder


class TestGridBuilder(TestCase):
    def test_sequenial_1(self):
        b = GridBuilder({'alpha': [1, 2, 3, 4]})

        self.assertEqual(4, b.combinations())

        cnt = 0
        for v in b.sequenial():
            cnt += 1

        self.assertEqual(b.combinations(), cnt)

    def test_sequenial_2(self):
        b = GridBuilder({'alpha': [1, 2], 'beta': [1, 2]})

        self.assertEqual(4, b.combinations())

        cnt = 0
        for v in b.sequenial():
            cnt += 1

        self.assertEqual(b.combinations(), cnt)

        cnt = 0
        for v in b.random():
            cnt += 1

        self.assertEqual(b.combinations(), cnt)
