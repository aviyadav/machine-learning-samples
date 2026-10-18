import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from must_know_scikit_learn_functions_for_data_processing.generate_input import main


class GenerateInputTests(unittest.TestCase):
    def test_generates_requested_rows_and_compatible_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'data' / 'input.csv'
            for count in (1, 30, 125):
                with self.subTest(count=count):
                    with (
                        patch('sys.argv', ['generate-input', str(count), '--output', str(output)]),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        main()
                    with output.open(newline='', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        self.assertEqual(reader.fieldnames, ['name', 'Age', 'Pet_Preference', 'income'])
                        rows = list(reader)
                    self.assertEqual(len(rows), count)
                    for row in rows:
                        self.assertTrue(row['name'].strip())
                        self.assertTrue(18 <= int(row['Age']) <= 75)
                        self.assertIn(row['Pet_Preference'], ('Dog', 'Cat', 'Bird', 'Rabbit', 'Hamster', 'Fish', 'Turtle', 'Guinea Pig'))
                        self.assertTrue(20000 <= int(row['income']) <= 150000)
                    frame = pl.read_csv(output, use_pyarrow=True)
                    self.assertEqual(len(frame), count)
                    self.assertEqual(sum(frame.null_count().row(0)), 0)

    def test_rejects_invalid_counts_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'input.csv'
            output.write_text('existing data', encoding='utf-8')
            for count in ('0', '-1', 'abc', '1.5'):
                with self.subTest(count=count):
                    with (
                        patch('sys.argv', ['generate-input', count, '--output', str(output)]),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as error,
                    ):
                        main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertEqual(output.read_text(encoding='utf-8'), 'existing data')


if __name__ == '__main__':
    unittest.main()
