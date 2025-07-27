import pandas as pd

# Load .pkl
df = pd.read_pickle("data/raw/datasteA.pkl")

# Inspect the descriptor dataset
print("=== DATASET INFO ===")
print(f"Shape: {df.shape}")  # Rows/cols
print(f"Columns: {list(df.columns)}")  # Feature names
print("\n=== FIRST FEW ROWS ===")
print(df.head())  # First rows

print("\n=== MISSING VALUES ===")
missing_percent = df.isnull().mean() * 100
print(missing_percent[missing_percent > 0])  # Only show columns with missing values

print("\n=== DATA TYPES ===")
print(df.dtypes.value_counts())

print("\n=== BASIC STATS ===")
print(df.describe())

# Check for duplicates and basic preprocessing
print(f"\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove any rows with missing values
df_clean = df.dropna()
print(f"\nAfter removing missing values: {df_clean.shape}")

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates: {df_clean.shape}")

print(f"\n=== FINAL CLEAN DATASET ===")
print(f"Shape: {df_clean.shape}")
print("Ready for modeling!")

# Save the cleaned dataset as CSV
output_path = "data/processed/datasteA_clean.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n=== SAVED TO CSV ===")
print(f"Dataset saved to: {output_path}")
print("Conversion complete!")