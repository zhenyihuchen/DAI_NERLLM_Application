import json
from data_preprocessor import preprocess_dataset
from gliner_extractor import extract_entities_gliner
from bert_extractor import HybridNERProcessor
from entity_merger import load_results_from_files, merge_entity_results, save_merged_results
from graphx import build_knowledge_graph

def main():
    """Main function to orchestrate the NER pipeline"""
    
    # ========================================================================
    # SECTION 1: DATA PREPROCESSING
    # ========================================================================
    # Step 1: Preprocess raw dataset
    print("Step 1: Preprocessing raw dataset...")
    raw_csv_path = 'data/teachers_db_practice.csv'
    processed_csv_path = 'data/teachers_db_practice_processed.csv'
    
    import pandas as pd
    df_processed = preprocess_dataset(raw_csv_path)
    
    # Save preprocessed dataset
    df_processed.to_csv(processed_csv_path, index=False)
    print(f"Preprocessed dataset saved to {processed_csv_path}")
    print(f"Dataset shape: {df_processed.shape}")
    
    # ========================================================================
    # SECTION 2: GLINER EXTRACTION
    # ========================================================================
    # Step 2: Extract entities using GLiNER
    print("\nStep 2: Extracting entities with GLiNER...")
    gliner_results = extract_entities_gliner(df_processed)
    
    # Save GLiNER results
    gliner_output_path = 'results/gliner_entities_results.json'
    with open(gliner_output_path, 'w') as f:
        json.dump(gliner_results, f, indent=2)
    print(f"GLiNER results saved to {gliner_output_path}")
    
    # ========================================================================
    # SECTION 3: BERT+REGEX EXTRACTION
    # ========================================================================
    # Step 3: Extract entities using BERT + Regex approach
    print("\nStep 3: Extracting entities with BERT + Regex...")
    processor = HybridNERProcessor()
    bert_regex_results = []
    
    for idx, row in df_processed.iterrows():
        print(f"Processing Professor {idx+1}: {row.get('alias', 'Unknown')}")
        
        html_content = row.get('full_info', '')
        prof_id = idx  # Use dataset index starting from 0
        alias = row.get('alias', f'Professor_{idx}')
        
        if html_content:
            result = processor.process_professor(html_content, prof_id, alias)
            bert_regex_results.append(result)
    
    # Save BERT+Regex results
    bert_regex_output_path = 'results/bert_regex_entities_results.json'
    with open(bert_regex_output_path, 'w') as f:
        json.dump(bert_regex_results, f, indent=2)
    print(f"BERT+Regex results saved to {bert_regex_output_path}")
    print(f"Completed NER extraction for {len(bert_regex_results)} rows")
    
    # ========================================================================
    # SECTION 4: ENTITY MERGING (ACTIVE)
    # ========================================================================
    print("\nStep 4: Merging GLiNER and BERT+Regex results...")
    
    # File paths
    gliner_file = 'results/gliner_entities_results.json'
    bert_file = 'results/bert_regex_entities_results.json'
    output_file = 'results/merged_entities_results.json'
    
    try:
        # Load results
        print("Loading GLiNER and BERT+Regex results...")
        gliner_results, bert_results = load_results_from_files(gliner_file, bert_file)
        
        print(f"GLiNER results: {len(gliner_results)} entries")
        print(f"BERT+Regex results: {len(bert_results)} entries")
        
        # Merge results
        print("Merging results...")
        merged_results = merge_entity_results(gliner_results, bert_results)
        
        # Save merged results
        print(f"Saving merged results to {output_file}...")
        save_merged_results(merged_results, output_file)
        
    #    print(f"Successfully merged {len(merged_results)} entries")
        
        # Print summary statistics
        total_entities = 0
        for result in merged_results:
            for section in ['academic_experience', 'academic_background', 'corporate_experience']:
                for category in result[section]:
                    total_entities += len(result[section][category])
        
        print(f"Total entities in merged results: {total_entities}")
        
        # Show sample merged results
        print("\nSample merged results:")
        for i in range(min(2, len(merged_results))):
            print(f"\nRow {i} - {merged_results[i]['alias']}:")
            print(f"Academic Experience: {merged_results[i]['academic_experience']}")
            print(f"Academic Background: {merged_results[i]['academic_background']}")
            print(f"Corporate Experience: {merged_results[i]['corporate_experience']}")
        
    except Exception as e:
        print(f"Error during merging: {str(e)}")
        raise
    
    # ========================================================================
    # SECTION 5: NETWORK GRAPH ANALYSIS
    # ========================================================================
    # Step 5: Generate social network graph from extracted entities
    print("\nStep 5: Generating network graph...")

    merged_json_path = 'results/merged_entities_results.json'
    output_gexf = 'results/professor_network.gexf'

    try:
        build_knowledge_graph(merged_json_path, output_gexf)
        print("Knowledge graph successfully generated and saved.")
    except Exception as e:
        print(f"Error generating knowledge graph: {e}")

if __name__ == "__main__":
    main()