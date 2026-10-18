from pathlib import Path

import polars as pl
import pyarrow as pa
from pyarrow import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def simple_imputer(df: pl.Dataframe) -> pl.Dataframe:
    # SimpleImputer : fill in the blanks
    df = df.with_columns(
        pl.Series('Age', SimpleImputer(strategy='mean').fit_transform(df.select('Age').to_numpy()).ravel()),
        pl.Series('income', SimpleImputer(strategy='constant', fill_value=30000).fit_transform(df.select('income').to_numpy()).ravel()),
    )

    print(f"Updates DF - {df}")
    return df

def label_encoder(df) -> pl.Dataframe:
    # LabelEncoder: Turning Words into Numbers
    df = df.with_columns(
        pl.Series('name', LabelEncoder().fit_transform(df['name'].to_numpy()))
    )
    print(f"LabelEncoded DF - \n{df}")

    return df

def onehot_encoder(df) -> pl.Dataframe:

    # OneHotEncoder: Spreading Out Categories
    encoder = OneHotEncoder(sparse_output=False)

    encoded_data = encoder.fit_transform(df.select('Pet_Preference').to_numpy())
    df = df.with_columns([
        pl.Series(col, encoded_data[:, i]) for i, col in enumerate(encoder.categories_[0])
    ])
    print(f"Encoded DF - \n{df}")

    df = df.drop('Pet_Preference')
    print(f"Encoded cleaned DF - \n{df}")
    return df

def std_scalar(df) -> pl.Dataframe:
    # StandardScaler: Balancing the Numbers
    scaled_data = StandardScaler().fit_transform(df.select('Age', 'income').to_numpy())
    df = df.with_columns([
        pl.Series(col, scaled_data[:, i]) for i, col in enumerate(['Age', 'income'])
    ])

    print(f"Scaled DF - \n{df}")

    return df

def load_data() -> pl.Dataframe:
    # load data
    data_path = Path(__file__).resolve().parents[2] / 'data' / 'sample.csv'
    df = pl.from_arrow(csv.read_csv(
        data_path,
        convert_options=csv.ConvertOptions(column_types={'Age': pa.float64(), 'income': pa.float64()}),
    ))
    print(df)
    return df

def main():
    df = load_data()
    df = simple_imputer(df)
    df = label_encoder(df)
    df = onehot_encoder(df)
    df = std_scalar(df)

if __name__ == "__main__":
    main()
