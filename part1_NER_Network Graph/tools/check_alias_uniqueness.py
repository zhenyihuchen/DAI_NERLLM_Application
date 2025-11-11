import pandas as pd

# Load dataset
df = pd.read_csv('data/teachers_db_practice.csv')

# Check alias uniqueness
total_rows = len(df)
unique_aliases = df['alias'].nunique()
duplicate_aliases = df[df.duplicated('alias', keep=False)]

print(f"Total rows: {total_rows}")
print(f"Unique aliases: {unique_aliases}")
print(f"Are all aliases unique? {total_rows == unique_aliases}")

if not duplicate_aliases.empty:
    print(f"\nFound {len(duplicate_aliases)} rows with duplicate aliases:")
    print(duplicate_aliases[['alias']].value_counts())
else:
    print("\nAll aliases are unique!")