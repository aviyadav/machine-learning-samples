use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    process,
    time::{SystemTime, UNIX_EPOCH},
};

const OUTPUT_FILE: &str = "happiness-data.csv";
const DEFAULT_ROWS: usize = 1_500;

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG; sufficient for generating sample data.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn range_inclusive(&mut self, min: u64, max: u64) -> u64 {
        min + self.next_u64() % (max - min + 1)
    }

    fn yes_or_no(&mut self) -> &'static str {
        if self.range_inclusive(0, 1) == 0 {
            "Yes"
        } else {
            "No"
        }
    }
}

fn main() {
    let rows = match env::args().nth(1) {
        Some(value) => match value.parse::<usize>() {
            Ok(rows) => rows,
            Err(_) => {
                eprintln!("Row count must be a non-negative integer.");
                process::exit(2);
            }
        },
        None => DEFAULT_ROWS,
    };

    if let Err(error) = generate_csv(rows) {
        eprintln!("Failed to create {OUTPUT_FILE}: {error}");
        process::exit(1);
    }

    println!("Created {OUTPUT_FILE} with {rows} rows.");
}

fn generate_csv(rows: usize) -> io::Result<()> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut rng = SimpleRng::new(seed);
    let file = File::create(OUTPUT_FILE)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "watched_tv,play_with_pets,code_written,had_coffee,happiness_value"
    )?;

    for _ in 0..rows {
        writeln!(
            writer,
            "{},{},{},{},{}",
            rng.yes_or_no(),
            rng.yes_or_no(),
            rng.range_inclusive(0, 1_500),
            rng.yes_or_no(),
            rng.range_inclusive(0, 10),
        )?;
    }

    writer.flush()
}
