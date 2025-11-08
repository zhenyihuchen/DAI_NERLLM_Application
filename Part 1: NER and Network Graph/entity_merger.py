import json
from typing import List, Dict, Any, Set

# ============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================================

def normalize_entity(entity: str) -> str:
    """Normalize entity text for comparison"""
    return entity.strip().lower().replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '-')

def entities_are_similar(entity1: str, entity2: str, threshold: float = 0.8) -> bool:
    """Check if two entities are similar enough to be considered duplicates
    
    Uses Jaccard similarity (word overlap) instead of Levenshtein distance because:
    - Jaccard is word-order independent: "Stanford University" = "University Stanford"
    - Better handles different entity lengths: "MIT" vs "Massachusetts Institute of Technology"
    - Focuses on semantic word overlap rather than character-level edits
    - Natural threshold interpretation: 0.8 = 80% shared words
    """
    norm1, norm2 = normalize_entity(entity1), normalize_entity(entity2)
    
    # Exact match
    if norm1 == norm2:
        return True
    
    # Check if one is contained in the other (for partial matches)
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Jaccard similarity: |intersection| / |union| of word sets
    words1, words2 = set(norm1.split()), set(norm2.split())
    if len(words1) > 0 and len(words2) > 0:
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union
        return jaccard_similarity >= threshold
    
    return False

# ============================================================================
# SECTION 2: ENTITY DEDUPLICATION
# ============================================================================

def deduplicate_entities(entities: List[str]) -> List[str]:
    """Remove duplicate entities from a list"""
    if not entities:
        return []
    
    unique_entities = []
    for entity in entities:
        is_duplicate = False
        for existing in unique_entities:
            if entities_are_similar(entity, existing):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_entities.append(entity)
    
    return unique_entities

# ============================================================================
# SECTION 3: CATEGORY MERGING
# ============================================================================

def merge_category_entities(gliner_entities: List[str], bert_entities: List[str]) -> List[str]:
    """Merge entities from two lists, removing duplicates"""
    # Combine all entities
    all_entities = gliner_entities + bert_entities
    
    # Remove duplicates
    return deduplicate_entities(all_entities)

def merge_experience_section(gliner_section: Dict[str, List[str]], bert_section: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge experience sections (academic_experience, academic_background, corporate_experience)"""
    merged_section = {}
    
    # Get all unique keys from both sections
    all_keys = set(gliner_section.keys()).union(set(bert_section.keys()))
    
    for key in all_keys:
        gliner_entities = gliner_section.get(key, [])
        bert_entities = bert_section.get(key, [])
        merged_section[key] = merge_category_entities(gliner_entities, bert_entities)
    
    return merged_section

# ============================================================================
# SECTION 4: MAIN MERGING LOGIC
# ============================================================================

def merge_single_result(gliner_result: Dict[str, Any], bert_result: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results for a single person"""
    # Verify IDs match
    if gliner_result['id'] != bert_result['id']:
        raise ValueError(f"ID mismatch: GLiNER {gliner_result['id']} vs BERT {bert_result['id']}")
    
    merged_result = {
        'id': gliner_result['id'],
        'alias': gliner_result['alias'],  # Use GLiNER alias as primary
        'academic_experience': merge_experience_section(
            gliner_result['academic_experience'],
            bert_result['academic_experience']
        ),
        'academic_background': merge_experience_section(
            gliner_result['academic_background'],
            bert_result['academic_background']
        ),
        'corporate_experience': merge_experience_section(
            gliner_result['corporate_experience'],
            bert_result['corporate_experience']
        )
    }
    
    return merged_result

def merge_entity_results(gliner_results: List[Dict[str, Any]], bert_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge entity extraction results from GLiNER and BERT+Regex approaches"""
    # Create dictionaries for quick lookup by ID
    gliner_dict = {result['id']: result for result in gliner_results}
    bert_dict = {result['id']: result for result in bert_results}
    
    # Get all unique IDs
    all_ids = set(gliner_dict.keys()).union(set(bert_dict.keys()))
    
    merged_results = []
    for person_id in sorted(all_ids):
        gliner_result = gliner_dict.get(person_id, {
            'id': person_id,
            'alias': f'Person_{person_id}',
            'academic_experience': {'Course': [], 'Program': [], 'Organization': []},
            'academic_background': {'Organization': [], 'Location': [], 'Education': [], 'Period': []},
            'corporate_experience': {'Organization': [], 'Location': []}
        })
        
        bert_result = bert_dict.get(person_id, {
            'id': person_id,
            'alias': f'Person_{person_id}',
            'academic_experience': {'Course': [], 'Program': [], 'Organization': []},
            'academic_background': {'Organization': [], 'Location': [], 'Education': [], 'Period': []},
            'corporate_experience': {'Organization': [], 'Location': []}
        })
        
        merged_result = merge_single_result(gliner_result, bert_result)
        merged_results.append(merged_result)
    
    return merged_results

# ============================================================================
# SECTION 5: FILE PROCESSING FUNCTIONS
# ============================================================================

def load_results_from_files(gliner_file: str, bert_file: str) -> tuple:
    """Load results from JSON files"""
    with open(gliner_file, 'r', encoding='utf-8') as f:
        gliner_results = json.load(f)
    
    with open(bert_file, 'r', encoding='utf-8') as f:
        bert_results = json.load(f)
    
    return gliner_results, bert_results

def save_merged_results(merged_results: List[Dict[str, Any]], output_file: str) -> None:
    """Save merged results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

# Main execution logic moved to main.py for testing