"""
GLINER ENTITY EXTRACTOR FOR ACADEMIC BIOGRAPHIES
===============================================
Uses GLiNER model for zero-shot named entity recognition.
Extracts structured entities from preprocessed biography sections.
"""

from gliner import GLiNER

# ============================================================================
# SECTION 1: MODEL INITIALIZATION
# ============================================================================

def extract_entities_gliner(df_processed):
    """
    Extract entities using GLiNER approach
    
    Args:
        df_processed: DataFrame with preprocessed sections
        
    Returns:
        List of dictionaries with extracted entities
    """
    
    # Load GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
    model.eval()
    print("GLiNER model loaded!")

# ============================================================================
# SECTION 2: ENTITY LABEL DEFINITIONS
# ============================================================================

    # Define labels for each column
    academic_experience_labels = [
        "subject or course taught by professor", 
        "academic program or degree level taught", 
        "teaching institution or university where professor works"
    ]

    academic_background_labels = [
        "educational institution where studied", 
        "location of educational institution", 
        "academic degree or qualification earned", 
        "years of study or graduation year"
    ]

    corporate_experience_labels = [
        "employer company or organization", 
        "workplace location or company headquarters"
    ]

# ============================================================================
# SECTION 3: MAIN PROCESSING PIPELINE
# ============================================================================

    # Process each row and extract entities
    results = []

    for idx, row in df_processed.iterrows():
        row_result = {
            "id": idx,
            "alias": row.get('alias', ''),
            "academic_experience": {
                "Course": [],
                "Program": [],
                "Organization": []
            },
            "academic_background": {
                "Organization": [],
                "Location": [],
                "Education": [],
                "Period": []
            },
            "corporate_experience": {
                "Organization": [],
                "Location": []
            }
        }

# ============================================================================
# SECTION 4: ENTITY EXTRACTION BY SECTION
# ============================================================================
        
        # Academic Experience NER
        if row['academic_experience'].strip():
            entities = model.predict_entities(row['academic_experience'], academic_experience_labels, threshold=0.4)
            for entity in entities:
                if entity["label"] == "subject or course taught by professor":
                    row_result["academic_experience"]["Course"].append(entity["text"])
                elif entity["label"] == "academic program or degree level taught":
                    row_result["academic_experience"]["Program"].append(entity["text"])
                elif entity["label"] == "teaching institution or university where professor works":
                    row_result["academic_experience"]["Organization"].append(entity["text"])
        
        # Academic Background NER
        if row['academic_background'].strip():
            entities = model.predict_entities(row['academic_background'], academic_background_labels, threshold=0.4)
            for entity in entities:
                if entity["label"] == "educational institution where studied":
                    row_result["academic_background"]["Organization"].append(entity["text"])
                elif entity["label"] == "location of educational institution":
                    row_result["academic_background"]["Location"].append(entity["text"])
                elif entity["label"] == "academic degree or qualification earned":
                    row_result["academic_background"]["Education"].append(entity["text"])
                elif entity["label"] == "years of study or graduation year":
                    row_result["academic_background"]["Period"].append(entity["text"])
        
        # Corporate Experience NER
        if row['corporate_experience'].strip():
            entities = model.predict_entities(row['corporate_experience'], corporate_experience_labels, threshold=0.4)
            for entity in entities:
                if entity["label"] == "employer company or organization":
                    row_result["corporate_experience"]["Organization"].append(entity["text"])
                elif entity["label"] == "workplace location or company headquarters":
                    row_result["corporate_experience"]["Location"].append(entity["text"])

# ============================================================================
# SECTION 5: POST-PROCESSING AND OUTPUT
# ============================================================================
        
        # Remove duplicates from each list
        for section in row_result.values():
            if isinstance(section, dict):
                for key, value_list in section.items():
                    section[key] = list(set(value_list))  # Remove duplicates
        
        results.append(row_result)
        
        if idx % 50 == 0:
            print(f"Processed {idx} rows...")

    return results