import duckdb

# Adjust path if needed
PARQUET_PATH = "data/public_match_info_dict.parquet"

def main():
    con = duckdb.connect()

    print("\nReading parquet file...\n")

    # Show column names + types
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{PARQUET_PATH}')"
    ).fetchdf()

    print("=== Columns ===")
    print(schema.to_string(index=False))

    # Show first few rows (just to confirm structure)
    preview = con.execute(
        f"SELECT * FROM read_parquet('{PARQUET_PATH}') LIMIT 5"
    ).fetchdf()

    print("\n=== First 5 Rows ===")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()