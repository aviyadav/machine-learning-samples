use std::{
    env, io,
    path::{Path, PathBuf},
    process::{self, Command},
};

fn main() {
    let input_file = match env::args().nth(1) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("Usage: cargo run --bin convert_latex_to_pdf -- <input.tex>");
            process::exit(2);
        }
    };

    if env::args().nth(2).is_some() {
        eprintln!("Only one input LaTeX file may be provided.");
        process::exit(2);
    }

    if let Err(error) = convert_latex_to_pdf(&input_file) {
        eprintln!("Failed to convert {}: {error}", input_file.display());
        process::exit(1);
    }
}

fn convert_latex_to_pdf(input_file: &Path) -> io::Result<()> {
    if !input_file.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "input LaTeX file was not found",
        ));
    }

    let output_directory = input_file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let output_file = output_directory.join(input_file.file_stem().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "input file has no filename")
    })?);
    let output_file = output_file.with_extension("pdf");

    let status = Command::new("pdflatex")
        .args([
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
        ])
        .arg(output_directory)
        .arg(input_file)
        .status()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("could not run pdflatex; ensure it is installed and on PATH: {error}"),
            )
        })?;

    if !status.success() {
        let code = status
            .code()
            .map_or_else(|| "unknown".to_string(), |code| code.to_string());
        return Err(io::Error::other(format!(
            "pdflatex exited with status {code}"
        )));
    }

    println!(
        "Created {} from {}.",
        output_file.display(),
        input_file.display()
    );
    Ok(())
}
