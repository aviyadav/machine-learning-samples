use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::prelude::*;
use ndarray::{Array2, s};
use std::{
    error::Error,
    fs::File,
    io::{self, Write},
    time::{SystemTime, UNIX_EPOCH},
};

const INPUT_FILE: &str = "happiness-data.csv";
const COLUMN_COUNT: usize = 5;
const TRAIN_RATIO: f32 = 0.8;
const MAX_DEPTH: Option<usize> = Some(5);

fn main() -> Result<(), Box<dyn Error>> {
    let original_data = read_happiness_data(INPUT_FILE)?;

    /*
    let original_data: Array2<f32> = array!(
        [1., 1., 1000., 1., 10.],
        [1., 0., 0., 1., 6.],
        [1., 0., 0., 1., 6.],
        [1., 0., 0., 1., 6.],
        [1., 0., 0., 1., 6.],
        [1., 0., 800., 1., 8.],
        [1., 0., 0., 0., 0.],
        [1., 1., 0., 1., 9.],
        [1., 1., 0., 1., 8.],
        [1., 0., 800., 0., 8.],
        [1., 1., 0., 1., 8.],
        [1., 1., 500., 0., 8.],
        [1., 0., 50., 0., 3.],
        [1., 1., 50., 0., 4.],
        [1., 0., 50., 0., 3.],
    );
    */

    let feature_names = vec!["watched_tv", "play_with_pets", "code_written", "had_coffee"];

    let num_features = original_data.len_of(Axis(1)) - 1;
    let features = original_data.slice(s![.., 0..num_features]).to_owned();
    let labels = original_data.column(num_features).to_owned();

    let linfra_dataset = Dataset::new(features, labels)
        .map_targets(happiness_label)
        .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(MAX_DEPTH)
        .fit(&linfra_dataset)
        .unwrap();

    // linfa-trees' tikz export writes feature names unescaped; underscores
    // are special characters in LaTeX text mode, so escape them.
    let tikz = model
        .export_to_tikz()
        .with_legend()
        .to_string()
        .replace('_', "\\_");

    File::create("dt.tex")?.write_all(tikz.as_bytes())?;

    evaluate(&original_data);

    Ok(())
}

fn happiness_label(value: &f32) -> String {
    match *value as i32 {
        i32::MIN..=4 => "Sad".to_string(),
        5..=7 => "Ok".to_string(),
        8..=i32::MAX => "Happy".to_string(),
    }
}

/// Shuffles row indices with a dependency-free LCG and splits the data into
/// training and test sets.
fn train_test_split(data: &Array2<f32>, train_ratio: f32) -> (Array2<f32>, Array2<f32>) {
    let row_count = data.nrows();
    let mut indices: Vec<usize> = (0..row_count).collect();

    let mut state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
        | 1;
    for i in (1..row_count).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_count = (row_count as f32 * train_ratio).round() as usize;
    let (train_indices, test_indices) = indices.split_at(train_count);

    (
        data.select(Axis(0), train_indices),
        data.select(Axis(0), test_indices),
    )
}

fn evaluate(original_data: &Array2<f32>) {
    let num_features = original_data.len_of(Axis(1)) - 1;

    let (train_data, test_data) = train_test_split(original_data, TRAIN_RATIO);
    let train_features = train_data.slice(s![.., 0..num_features]).to_owned();
    let train_labels = train_data.column(num_features).to_owned();
    let test_features = test_data.slice(s![.., 0..num_features]).to_owned();
    let test_labels = test_data.column(num_features).to_owned();

    let train_dataset = build_dataset(train_features, train_labels);
    let test_dataset = build_dataset(test_features, test_labels);

    let model = match DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(MAX_DEPTH)
        .fit(&train_dataset)
    {
        Ok(model) => model,
        Err(error) => {
            eprintln!("evaluation training failed: {error}");
            return;
        }
    };

    let predictions = model.predict(&test_dataset);
    let predicted: Vec<String> = predictions.as_targets().to_vec();
    let actual: Vec<String> = test_dataset.targets().to_vec();

    print_evaluation(train_dataset.nsamples(), &actual, &predicted);
}

fn build_dataset(
    features: Array2<f32>,
    labels: Array1<f32>,
) -> DatasetBase<Array2<f32>, Array1<String>> {
    Dataset::new(features, labels).map_targets(happiness_label)
}

fn read_happiness_data(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let contents = std::fs::read_to_string(path)?;
    let mut lines = contents.lines();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "CSV file is empty"))?;
    let expected_header = "watched_tv,play_with_pets,code_written,had_coffee,happiness_value";

    if header != expected_header {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected CSV header in {path}"),
        )
        .into());
    }

    let mut values = Vec::new();
    let mut row_count = 0;

    for (line_number, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let columns: Vec<&str> = line.split(',').collect();
        if columns.len() != COLUMN_COUNT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected {COLUMN_COUNT} columns on CSV line {}",
                    line_number + 2
                ),
            )
            .into());
        }

        values.push(parse_yes_no(columns[0], line_number + 2)?);
        values.push(parse_yes_no(columns[1], line_number + 2)?);
        values.push(parse_number(columns[2], "code_written", line_number + 2)?);
        values.push(parse_yes_no(columns[3], line_number + 2)?);
        values.push(parse_number(
            columns[4],
            "happiness_value",
            line_number + 2,
        )?);
        row_count += 1;
    }

    if row_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} contains no data rows"),
        )
        .into());
    }

    Ok(Array2::from_shape_vec((row_count, COLUMN_COUNT), values)?)
}

fn parse_yes_no(value: &str, line_number: usize) -> Result<f32, io::Error> {
    match value {
        "Yes" => Ok(1.0),
        "No" => Ok(0.0),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected Yes or No on CSV line {line_number}"),
        )),
    }
}

fn parse_number(value: &str, column: &str, line_number: usize) -> Result<f32, io::Error> {
    value.parse::<f32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid {column} value on CSV line {line_number}"),
        )
    })
}

fn print_evaluation(train_count: usize, actual: &[String], predicted: &[String]) {
    let mut classes: Vec<String> = actual
        .iter()
        .chain(predicted)
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    classes.sort();

    let class_count = classes.len();
    let mut matrix = vec![vec![0usize; class_count]; class_count];
    for (a, p) in actual.iter().zip(predicted) {
        let ai = classes.iter().position(|c| c == a).unwrap();
        let pi = classes.iter().position(|c| c == p).unwrap();
        matrix[ai][pi] += 1;
    }

    let total: usize = matrix.iter().map(|row| row.iter().sum::<usize>()).sum();
    let correct: usize = (0..class_count).map(|i| matrix[i][i]).sum();

    println!(
        "\n=== Evaluation (train/test split {:.0}/{:.0}) ===",
        100.0 * TRAIN_RATIO,
        100.0 * (1.0 - TRAIN_RATIO)
    );
    println!("Training samples: {train_count}, test samples: {total}");
    println!(
        "Accuracy: {correct}/{total} = {:.1}%",
        100.0 * correct as f32 / total as f32
    );

    let majority = matrix
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .max()
        .unwrap_or(0);
    println!(
        "Majority-class baseline: {:.1}%",
        100.0 * majority as f32 / total as f32
    );

    println!("\nConfusion matrix (rows = actual, columns = predicted):");
    let header: String = classes.iter().map(|c| format!("{c:>8}")).collect();
    println!("{:>12}{header}", "actual\\pred");
    for (i, row) in matrix.iter().enumerate() {
        let cells: String = row.iter().map(|v| format!("{v:>8}")).collect();
        println!("{:>12}{cells}", classes[i]);
    }

    println!("\nPer-class metrics (one-vs-all):");
    println!(
        "{:>10} {:>10} {:>10} {:>10}",
        "class", "precision", "recall", "support"
    );
    for (i, class) in classes.iter().enumerate() {
        let tp = matrix[i][i] as f32;
        let predicted_as_class: usize = (0..class_count).map(|r| matrix[r][i]).sum();
        let actual_class: usize = matrix[i].iter().sum();
        let precision = if predicted_as_class > 0 {
            tp / predicted_as_class as f32
        } else {
            0.0
        };
        let recall = if actual_class > 0 {
            tp / actual_class as f32
        } else {
            0.0
        };
        println!(
            "{:>10} {:>10.3} {:>10.3} {:>10}",
            class, precision, recall, actual_class
        );
    }
}
