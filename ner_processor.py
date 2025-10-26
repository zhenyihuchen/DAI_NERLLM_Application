import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

class ProfessorNER:
    def __init__(self):
        # Load pre-trained NER model
        self.ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    
    def load_data(self, file_path):
        """Load professor data from CSV/parquet"""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)
    
    def clean_text(self, text):
        """Clean HTML and format text for NER"""
        if pd.isna(text):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', str(text))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text):
        """Extract entities using NER model"""
        if not text or len(text) < 10:
            return []
        
        # Split long text into chunks (BERT has token limits)
        chunks = [text[i:i+500] for i in range(0, len(text), 450)]
        all_entities = []
        
        for chunk in chunks:
            try:
                entities = self.ner_pipeline(chunk)
                all_entities.extend(entities)
            except:
                continue
        
        return all_entities
    
    def categorize_entities(self, entities, text):
        """Categorize entities into Universities, Companies, Studies, Courses"""
        categories = {
            'universities': [],
            'companies': [],
            'studies': [],
            'courses': []
        }
        
        # Keywords for categorization
        uni_keywords = ['university', 'college', 'school', 'institute']
        company_keywords = ['group', 'consulting', 'systems', 'technologies', 'corp']
        study_keywords = ['mba', 'phd', 'bachelor', 'master', 'degree', 'engineering']
        course_keywords = ['management', 'finance', 'marketing', 'analytics', 'relations']
        
        for entity in entities:
            entity_text = entity['word'].lower()
            
            # Check context around entity
            start_pos = max(0, text.lower().find(entity_text) - 50)
            end_pos = min(len(text), text.lower().find(entity_text) + len(entity_text) + 50)
            context = text[start_pos:end_pos].lower()
            
            if entity['entity_group'] == 'ORG':
                if any(keyword in context for keyword in uni_keywords):
                    categories['universities'].append(entity['word'])
                elif any(keyword in context for keyword in company_keywords):
                    categories['companies'].append(entity['word'])
                elif any(keyword in context for keyword in study_keywords):
                    categories['studies'].append(entity['word'])
                elif any(keyword in context for keyword in course_keywords):
                    categories['courses'].append(entity['word'])
                else:
                    categories['companies'].append(entity['word'])  # Default for ORG
            
            elif entity['entity_group'] == 'LOC':
                categories['universities'].append(entity['word'])  # Locations often relate to unis
        
        return categories
    
    def process_professor_data(self, df):
        """Process all professor data and extract entities"""
        results = []
        
        for idx, row in df.iterrows():
            # Combine relevant text fields
            text_content = ""
            if 'description' in row and pd.notna(row['description']):
                text_content += self.clean_text(row['description'])
            
            # Extract entities
            entities = self.extract_entities(text_content)
            categorized = self.categorize_entities(entities, text_content)
            
            professor_data = {
                'professor_id': idx,
                'name': row.get('name', f'Professor_{idx}'),
                'department': row.get('department', ''),
                'universities': list(set(categorized['universities'])),
                'companies': list(set(categorized['companies'])),
                'studies': list(set(categorized['studies'])),
                'courses': list(set(categorized['courses'])),
                'raw_entities': entities
            }
            
            results.append(professor_data)
            
            if idx % 10 == 0:
                print(f"Processed {idx} professors...")
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize NER processor
    ner_processor = ProfessorNER()
    
    # Load data
    df = ner_processor.load_data('data/teachers_db_practice.csv')
    print(f"Loaded {len(df)} professors")
    
    # Process first 5 professors for testing
    sample_df = df.head(5)
    results = ner_processor.process_professor_data(sample_df)
    
    # Display results
    for result in results:
        print(f"\nProfessor: {result['name']}")
        print(f"Universities: {result['universities']}")
        print(f"Companies: {result['companies']}")
        print(f"Studies: {result['studies']}")
        print(f"Courses: {result['courses']}")