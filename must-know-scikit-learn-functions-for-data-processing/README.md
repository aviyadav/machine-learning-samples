# Scikit-learn data processing examples

## Generate input data

Generate a CSV with any positive number of entries using Faker:

```sh
uv run generate-input 100
```

By default, this replaces `data/input.csv` in the project, regardless of the
working directory. To write somewhere else:

```sh
uv run generate-input 100 --output data/sample.csv
```

The CSV has the columns expected by the processing example: `name`, `Age`,
`Pet_Preference`, and `income`. Rows contain synthetic US names, ages from 18 to
75, a preference for Dog, Cat, Bird, Rabbit, Hamster, Fish, Turtle, or Guinea Pig,
and annual incomes from 20,000 to 150,000.
Generated rows contain no missing values; the existing sample CSV retains its
missing values until you replace it.

## Run the processing example

The program reads `data/sample.csv` with PyArrow, uses Polars for dataframe
operations, and passes NumPy arrays to scikit-learn for transformations.
Generate this file with `uv run generate-input 100 --output data/sample.csv`.

```sh
uv run python -m must_know_scikit_learn_functions_for_data_processing.main
```

## Run tests

```sh
uv run python -m unittest discover -s tests
```
