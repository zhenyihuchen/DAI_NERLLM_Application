import json
from data_preprocessor import preprocess_dataset
from gliner_extractor import extract_entities_gliner

def main():
    """Main function to orchestrate the NER pipeline"""
    
    # Step 1: Preprocess the dataset
    print("Step 1: Preprocessing dataset...")
    csv_path = 'data/teachers_db_practice.csv'
    df_processed = preprocess_dataset(csv_path)
    
    # Step 2: Extract entities using GLiNER
    print("\nStep 2: Extracting entities with GLiNER...")
    gliner_results = extract_entities_gliner(df_processed)
    
    # Step 3: Save GLiNER results
    with open('gliner_entities_results.json', 'w') as f:
        json.dump(gliner_results, f, indent=2)
    
    print(f"\nCompleted NER extraction for {len(gliner_results)} rows")
    print("Results saved to gliner_entities_results.json")
    
    # Show sample results
    print("\nSample results:")
    for i in range(min(3, len(gliner_results))):
        print(f"\nRow {i} - {gliner_results[i]['alias']}:")
        print(f"Academic Experience: {gliner_results[i]['academic_experience']}")
        print(f"Academic Background: {gliner_results[i]['academic_background']}")
        print(f"Corporate Experience: {gliner_results[i]['corporate_experience']}")

if __name__ == "__main__":
    main()