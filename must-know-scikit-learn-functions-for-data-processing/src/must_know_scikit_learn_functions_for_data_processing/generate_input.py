import argparse
import csv
from pathlib import Path

from faker import Faker


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate realistic input data as CSV.')
    parser.add_argument('entries', type=int, help='Number of rows to generate (must be positive)')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'data' / 'input.csv',
        help='Destination CSV file (default: project data/input.csv; overwrites existing files)',
    )
    args = parser.parse_args()
    if args.entries <= 0:
        parser.error('entries must be a positive integer')

    fake = Faker('en_US')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'Age', 'Pet_Preference', 'income'])
        for _ in range(args.entries):
            writer.writerow([
                fake.name(),
                fake.random_int(min=18, max=75),
                fake.random_element(elements=('Dog', 'Cat', 'Bird', 'Rabbit', 'Hamster', 'Fish', 'Turtle', 'Guinea Pig')),
                fake.random_int(min=20000, max=150000, step=1000),
            ])

    print(f'Generated {args.entries} entries in {args.output}')


if __name__ == '__main__':
    main()
