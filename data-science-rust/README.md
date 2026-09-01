# Data Science Rust

A Rust project for experimenting with data science, happiness data, decision
trees, and LaTeX/PDF output.

## Prerequisites

Install the following tools:

- [Rust](https://www.rust-lang.org/tools/install), including `cargo` and
  `rustc`
- A LaTeX distribution with `pdflatex` available on the system `PATH` if you
  want to generate PDFs. Common choices include:
  - [MiKTeX](https://miktex.org/) on Windows
  - [TeX Live](https://www.tug.org/texlive/) on Linux and macOS

Verify the installations:

```bash
cargo --version
rustc --version
pdflatex --version
```

### Ubuntu LaTeX installation

Install TeX Live packages containing `pdflatex`, TikZ, and Forest:

```bash
sudo apt update
sudo apt install texlive-latex-extra texlive-pictures
```

Verify that Ubuntu can find the compiler:

```bash
which pdflatex
pdflatex --version
```

`pdflatex` is only required by the PDF conversion program. The Rust programs
can be built without it.

## Project structure

```text
.
├── Cargo.toml
├── Cargo.lock
├── README.md
├── happiness-data.csv
├── dt.tex
└── src
    ├── main.rs
    └── bin
        ├── basic_main.rs
        ├── convert_latex_to_pdf.rs
        └── generate_happiness_data.rs
```

## Cargo project details

- Package name: `data-science-rust`
- Version: `0.1.0`
- Rust edition: `2024`
- Dependencies:
  - `linfa = 0.8.1`
  - `linfa-trees = 0.8.1`
  - `ndarray = 0.16.1`

The `ndarray` version is kept on the `0.16.x` line for compatibility with
`linfa 0.8.1`.

## Programs

### Analysis program

The default binary is `src/main.rs`. It currently:

1. Reads and validates `happiness-data.csv`.
2. Converts `Yes` to `1.0` and `No` to `0.0` for the categorical columns.
3. Parses `code_written` and `happiness_value` as numeric values.
4. Splits the data into four features and a happiness target.
5. Maps target values into `Sad`, `Ok`, and `Happy` categories.
6. Trains a decision tree using the Gini split quality with a maximum depth
   of 5. The depth limit keeps the tree small enough to render as a PDF;
   without it, the fully grown tree can exceed TeX's maximum page dimension.
7. Exports the decision tree as a TikZ/LaTeX document named `dt.tex`.
8. Evaluates the model on a held-out test set and prints the results to the
   terminal (see "Model evaluation" below).

The export uses `linfa-trees`' `export_to_tikz().with_legend()`. The crate
writes feature names unescaped, so the program replaces `_` with `\_` before
writing the file (underscores are special characters in LaTeX text mode).

### Model evaluation

The analysis program evaluates how well the decision tree predicts the
happiness category (`Sad`, `Ok`, or `Happy`) from the four feature columns.

- The data is shuffled with a dependency-free LCG random number generator and
  split into a training set (80%) and a test set (20%). The split ratio is
  controlled by the `TRAIN_RATIO` constant, and the maximum tree depth by
  `MAX_DEPTH`, both at the top of `src/main.rs`.
- A fresh tree is trained on the training set only and evaluated on the test
  set, so the reported metrics measure generalization rather than memorization.
- The program prints:
  - Overall accuracy and the majority-class baseline for comparison.
  - A confusion matrix (rows = actual class, columns = predicted class).
  - Per-class precision, recall, and support (one-vs-all).

Because the split is re-shuffled on every run, the reported numbers vary
slightly between runs. The evaluation is only as good as the dataset: with a
small CSV the test set is tiny and the metrics are noisy, so consider
generating a larger dataset first (e.g. `cargo run --bin generate_happiness_data --
5000`).

### Basic test program

`src/bin/basic_main.rs` is a self-contained test version of the analysis that
works entirely with hard-coded data. It does not read or write any CSV file.
Instead, it embeds a 15-row sample dataset directly in the source, applies the
same feature mapping (`watched_tv`, `play_with_pets`, `code_written`,
`had_coffee` → `Sad`/`Ok`/`Happy`), trains a Gini decision tree with no depth
limit, and writes the tree to `dt.tex` (with the same underscore escaping as
the main program).

It is useful for a quick sanity check of the training and LaTeX export
pipeline without depending on the CSV data or the data generator. Like the
other programs, it overwrites `dt.tex` when run, so re-run the main analysis
program afterwards if you want the CSV-based tree back.

Run it with:

```bash
cargo run --bin basic_main
```

## End-to-end example (real run)

A complete run — generate data, train and evaluate the tree, and render it as
a PDF:

```bash
# 1. Generate the dataset (default 1,500 rows, or pass a custom count)
cargo run --bin generate_happiness_data

# 2. Train the decision tree on happiness-data.csv.
#    Writes dt.tex and prints the model evaluation to the terminal.
cargo run --bin data-science-rust

# 3. Convert dt.tex to dt.pdf and open it to view the decision tree.
cargo run --bin convert_latex_to_pdf -- dt.tex
```

After step 3, open `dt.pdf` with any PDF viewer to see the rendered decision
tree, including the Y/N edge labels and the feature-name legend. A larger
dataset produces a better-evaluated but deeper tree, so `MAX_DEPTH` in
`src/main.rs` (currently 5) bounds the size of the rendered PDF.

The previous hard-coded sample dataset is retained in a comment in `src/main.rs`
for reference. This file is also the main place to add future analysis code.

### Happiness data generator

`src/bin/generate_happiness_data.rs` creates `happiness-data.csv` using a
standard-library pseudo-random number generator.

The CSV contains one row per day. Dates are not captured. It has the following
columns:

| Column | Allowed values |
| --- | --- |
| `watched_tv` | `Yes` or `No` |
| `play_with_pets` | `Yes` or `No` |
| `code_written` | Integer from `0` through `1500` |
| `had_coffee` | `Yes` or `No` |
| `happiness_value` | Integer from `0` through `10` |

The generator creates 1,500 data rows by default. Existing files with the same
name are overwritten.

### LaTeX-to-PDF converter

`src/bin/convert_latex_to_pdf.rs` converts a LaTeX file to PDF by invoking
`pdflatex`. It accepts exactly one input file and creates a PDF with the same
name and directory:

```text
reports/tree.tex  ->  reports/tree.pdf
```

The converter uses `-interaction=nonstopmode` and `-halt-on-error`, checks that
the input file exists, and reports a useful error if `pdflatex` is missing or
compilation fails. It passes `-output-directory` pointing at the input file's
directory (falling back to the current directory for bare filenames such as
`dt.tex`), so all auxiliary files are written next to the input.

## Build commands

Run these commands from the project root, the directory containing
`Cargo.toml`.

Build the analysis program:

```bash
cargo build --bin data-science-rust
```

Build the happiness data generator:

```bash
cargo build --bin generate_happiness_data
```

Build the LaTeX-to-PDF converter:

```bash
cargo build --bin convert_latex_to_pdf
```

Build all binaries:

```bash
cargo build --bins
```

Build optimized release binaries by adding `--release`:

```bash
cargo build --release --bins
```

Compiled binaries are placed under `target/debug` or `target/release`.

## Run commands

Run the analysis program. This creates or overwrites `dt.tex`:

```bash
cargo run --bin data-science-rust
```

Generate the default 1,500-row happiness dataset:

```bash
cargo run --bin generate_happiness_data
```

Generate a custom number of rows, such as 500:

```bash
cargo run --bin generate_happiness_data -- 500
```

The row count must be a non-negative integer. The generator writes
`happiness-data.csv` to the current working directory.

Convert the generated `dt.tex` file to `dt.pdf`:

```bash
cargo run --bin convert_latex_to_pdf -- dt.tex
```

Convert another LaTeX file:

```bash
cargo run --bin convert_latex_to_pdf -- reports/tree.tex
```

The input file must exist, and only one input path may be supplied.

## Recommended workflow

See the "End-to-end example (real run)" above for the full generate → analyze →
convert-to-PDF sequence. To experiment with dataset size, change the row count
passed to the generator — a larger dataset gives more reliable evaluation
metrics but produces a deeper tree:

```bash
cargo run --bin generate_happiness_data -- 5000
cargo run --bin data-science-rust
cargo run --bin convert_latex_to_pdf -- dt.tex
```

The analysis program expects `happiness-data.csv` in the current working
directory. Run the data generator first if the file does not exist. Note that
`basic_main` also writes `dt.tex` (from its hard-coded sample), so re-run the
main analysis program afterwards if you want the CSV-based tree back.

## Validation, formatting, and tests

Format all Rust source files:

```bash
cargo fmt
```

Check formatting without changing files:

```bash
cargo fmt -- --check
```

Run compiler checks:

```bash
cargo check
```

Check all binaries explicitly:

```bash
cargo check --bins
```

Run the test suite:

```bash
cargo test
```

## Generated files

The following files are generated or updated by the programs:

- `happiness-data.csv` — generated by `generate_happiness_data`
- `dt.tex` — generated by the analysis program
- `dt.pdf` — generated by `convert_latex_to_pdf`
- LaTeX auxiliary files such as `.aux`, `.log`, and `.out` — generated by
  `pdflatex`

The Rust build directory, `target/`, is ignored by Git.

## Troubleshooting

### `main` function not found

Files under `src/bin` are compiled as independent binaries. Each file must
contain its own `fn main()`. Both standalone programs in this project provide
one.

### `pdflatex` is not recognized

The converter depends on an external LaTeX installation; `pdflatex` is not
provided by Rust or Cargo. On Windows, install [MiKTeX](https://miktex.org/)
or [TeX Live](https://www.tug.org/texlive/). During MiKTeX setup, allowing it
to install missing packages automatically is recommended.

After installation, make sure the distribution's `bin` directory is on the
system `PATH`. For a typical MiKTeX installation, the directory is similar to:

```text
C:\\Program Files\\MiKTeX\\miktex\\bin\\x64
```

The exact path depends on whether MiKTeX was installed for all users or only
the current user. Open a new PowerShell or Command Prompt after changing
`PATH`, then verify that Windows can find the executable:

```powershell
where.exe pdflatex
pdflatex --version
```

If `where.exe pdflatex` returns no path, `pdflatex` is still not available to
the terminal running Cargo. Once the version command succeeds, retry:

```bash
cargo run --bin convert_latex_to_pdf -- dt.tex
```

### `dt.tex` is not found

Run the converter from the project root, or provide the correct relative or
absolute path to the input file:

```bash
cargo run --bin convert_latex_to_pdf -- path/to/input.tex
```

### `I can't write on file 'dt.log'`

This means `pdflatex` could not write its auxiliary files, usually because it
received an invalid output directory. Versions of the converter before the
`-output-directory` fix passed an empty path when given a bare filename such as
`dt.tex`. Rebuild the converter so you are running the fixed version:

```bash
cargo build --bin convert_latex_to_pdf
```

### `Dimension too large` when compiling `dt.tex`

The decision tree is wider than TeX's maximum dimension of about 5.75 meters.
This happens when the tree is fully grown and has many leaves, because the
`linfa-trees` export sets a large sibling separation (`s sep'+=2cm`). Fix it by
limiting the tree depth in `src/main.rs`:

```rust
DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .max_depth(Some(5))
```

A smaller dataset (fewer rows via `generate_happiness_data`) also produces a
smaller tree.

### `Missing $ inserted` when compiling `dt.tex`

Feature names containing underscores, such as `code_written`, were written to
the legend unescaped. Current versions of the analysis program escape
underscores before writing `dt.tex`. If you generate `dt.tex` another way,
escape each `_` as `\_` manually.

### Dependency version mismatch involving `ndarray`

Keep the direct `ndarray` dependency aligned with the version expected by
`linfa 0.8.1`, currently `0.16.1` in `Cargo.toml`. If dependencies have changed,
refresh the lockfile with:

```bash
cargo update
```
