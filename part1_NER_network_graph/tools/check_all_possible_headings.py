import pandas as pd
import re
from collections import Counter

# Load dataset
df = pd.read_csv('../data/teachers_db_practice.csv')

# Extract all h4 headings
all_headings = []

for idx, row in df.iterrows():
    full_info = row.get('full_info', '')
    if pd.notna(full_info):
        # Find all h4 tags and extract text between them
        headings = re.findall(r'<h4[^>]*>(.*?)</h4>', str(full_info), re.IGNORECASE)
        all_headings.extend(headings)

# Count occurrences of each heading
heading_counts = Counter(all_headings)

print(f"Total unique headings found: {len(heading_counts)}")
print(f"Total heading occurrences: {sum(heading_counts.values())}")
print("\nAll headings and their frequencies:")
print("-" * 50)

for heading, count in heading_counts.most_common():
    print(f"{count:4d} | {heading}")