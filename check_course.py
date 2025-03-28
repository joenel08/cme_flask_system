import pandas as pd

# Load the CSV file
file_path = "excel_files/CME_DATA.csv"
df = pd.read_csv(file_path)

# Group by a column and display unique values per group
grouped = df.groupby('COURSE CODE')

for key, group in grouped:
    print(f"Group: {key}")
    print(group)
    print("-" * 30)

# Ensure all unique values are shown without truncation
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width

# Display unique values as a Python list (no truncation)
unique_values = df['COURSE CODE'].unique().tolist()  # Convert to list
print("\nUnique COURSE:")
print(unique_values)
