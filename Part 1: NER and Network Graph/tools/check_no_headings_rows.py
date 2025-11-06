import pandas as pd
import re

# Load dataset
df = pd.read_csv('data/teachers_db_practice.csv')

# Check for rows without h4 headings
no_headings = []

for idx, row in df.iterrows():
    full_info = row.get('full_info', '')
    if pd.isna(full_info) or not re.search(r'<h4>', str(full_info), re.I):
        no_headings.append(idx)

print(f"Total rows without h4 headings: {len(no_headings)}")
print(f"Total rows in dataset: {len(df)}")
print(f"Percentage without headings: {len(no_headings)/len(df)*100:.1f}%")

if no_headings:
    print(f"\nRow numbers without headings: {no_headings[:20]}...")  # Show first 20
    if len(no_headings) > 20:
        print(f"... and {len(no_headings) - 20} more")