import pandas as pd
import json

# Load original dataset
df = pd.read_csv('data/teachers_db_practice.csv')
original_row_ids = set(df.index)

# Load JSON results
with open('gliner_entities_results.json', 'r') as f:
    results = json.load(f)

# Extract row IDs from JSON
json_row_ids = set(result['row_id'] for result in results)

# Find missing row IDs
missing_row_ids = original_row_ids - json_row_ids

print(f"Original dataset rows: {len(original_row_ids)}")
print(f"JSON results rows: {len(json_row_ids)}")
print(f"Coverage: {len(json_row_ids)}/{len(original_row_ids)} ({len(json_row_ids)/len(original_row_ids)*100:.1f}%)")

if missing_row_ids:
    print(f"\nMissing {len(missing_row_ids)} row IDs:")
    print(sorted(list(missing_row_ids))[:20])  # Show first 20
    if len(missing_row_ids) > 20:
        print(f"... and {len(missing_row_ids) - 20} more")
else:
    print("\nAll rows are covered!")