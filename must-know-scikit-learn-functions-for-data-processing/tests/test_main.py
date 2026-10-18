import contextlib
import io
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
from pyarrow import csv


class ProcessingTests(unittest.TestCase):
    def test_arrow_csv_processing_and_independent_scaling(self):
        cases = (
            ('Alice,25,Dog,30000\nBob,,Cat,45000\nClaire,30,Dog,\nDave,22,Bird,28000\n',
             [25, 77 / 3, 30, 22], [30000, 45000, 30000, 28000]),
            ('Alice,25,Dog,30000\nBob,35,Cat,45000\nClaire,30,Dog,55000\nDave,22,Bird,28000\n',
             [25, 35, 30, 22], [30000, 45000, 55000, 28000]),
        )
        for rows, ages, incomes in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.csv'
                path.write_text('name,Age,Pet_Preference,income\n' + rows, encoding='utf-8')
                read_csv = csv.read_csv
                with (
                    patch(
                        'pyarrow.csv.read_csv',
                        side_effect=lambda *args, read_csv=read_csv, path=path, **kwargs: read_csv(path, **kwargs),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = runpy.run_module('must_know_scikit_learn_functions_for_data_processing.main')
                frame = result['df']
                self.assertIsInstance(frame, pl.DataFrame)
                self.assertEqual(frame.columns, ['name', 'Age', 'income', 'Bird', 'Cat', 'Dog'])
                self.assertEqual(frame.height, 4)
                self.assertEqual(sum(frame.null_count().row(0)), 0)
                self.assertEqual(frame['name'].to_list(), [0, 1, 2, 3])
                np.testing.assert_array_equal(
                    frame.select('Bird', 'Cat', 'Dog').to_numpy(),
                    [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
                )
                for column, values in [('Age', ages), ('income', incomes)]:
                    expected = np.array(values, dtype=float)
                    expected = (expected - expected.mean()) / expected.std()
                    np.testing.assert_allclose(frame[column].to_numpy(), expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
