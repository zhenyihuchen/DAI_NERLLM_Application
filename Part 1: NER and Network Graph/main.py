import json
from data_preprocessor import preprocess_dataset
# from gliner_extractor import extract_entities_gliner
from langextract_extractor import process_full_dataset_in_batches

def main():
    """Main function to orchestrate the NER pipeline"""
    
    # Step 1: Load preprocessed dataset
    print("Step 1: Loading preprocessed dataset...")
    csv_path = 'data/teachers_db_practice_processed.csv'
    import pandas as pd
    df_processed = pd.read_csv(csv_path)
    
    # Step 2: Extract entities using GLiNER
    # print("\nStep 2: Extracting entities with GLiNER...")
    # gliner_results = extract_entities_gliner(df_processed)
    
    # Step 2: Extract entities using LangExtract in batches
    print("\nStep 2: Extracting entities with LangExtract in batches...")
    langextract_results = process_full_dataset_in_batches(df_processed, batch_size=50)
    
    # Step 4: Save results
    # gliner_output_path = 'results/gliner_entities_results.json'
    # with open(gliner_output_path, 'w') as f:
    #     json.dump(gliner_results, f, indent=2)
    
    langextract_output_path = 'results/langextract_entities_results.json'
    with open(langextract_output_path, 'w') as f:
        json.dump(langextract_results, f, indent=2)
    
    print(f"\nCompleted NER extraction for {len(langextract_results)} rows")
    # print(f"GLiNER results saved to {gliner_output_path}")
    print(f"LangExtract results saved to {langextract_output_path}")
    
    # Show sample results comparison
    # print("\nSample GLiNER results:")
    # for i in range(min(2, len(gliner_results))):
    #     print(f"\nRow {i} - {gliner_results[i]['alias']}:")
    #     print(f"Academic Experience: {gliner_results[i]['academic_experience']}")
    #     print(f"Academic Background: {gliner_results[i]['academic_background']}")
    #     print(f"Corporate Experience: {gliner_results[i]['corporate_experience']}")
    
    print("\nSample LangExtract results:")
    for i in range(min(2, len(langextract_results))):
        print(f"\nRow {i} - {langextract_results[i]['alias']}:")
        print(f"Academic Experience: {langextract_results[i]['academic_experience']}")
        print(f"Academic Background: {langextract_results[i]['academic_background']}")
        print(f"Corporate Experience: {langextract_results[i]['corporate_experience']}")

if __name__ == "__main__":
    main()