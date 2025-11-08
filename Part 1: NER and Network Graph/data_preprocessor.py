"""
DATA PREPROCESSOR FOR ACADEMIC BIOGRAPHIES
==========================================
Extracts and preprocesses HTML sections from professor biographies.
Normalizes headings and extracts content by section type.
"""

import pandas as pd
import re
from bs4 import BeautifulSoup

# ============================================================================
# SECTION 1: TEXT NORMALIZATION UTILITIES
# ============================================================================

def normalize_heading(heading):
    """Normalize heading by removing spaces, tags, punctuation and making lowercase"""
    # Remove HTML tags
    heading = re.sub(r'<[^>]+>', '', heading)
    # Remove spaces and punctuation
    heading = re.sub(r'[^\w]', '', heading)
    # Make lowercase
    return heading.lower()

# ============================================================================
# SECTION 2: HTML CONTENT EXTRACTION
# ============================================================================

def extract_section_content(html_text, section_headings):
    """Extract content following specific section headings"""
    if pd.isna(html_text):
        return ""
    
    # Normalize target headings
    normalized_targets = [normalize_heading(h) for h in section_headings]
    
    # Find all h4 sections
    soup = BeautifulSoup(html_text, 'html.parser')
    h4_tags = soup.find_all('h4')
    
    content = []
    for i, h4 in enumerate(h4_tags):
        heading_text = h4.get_text()
        normalized_heading = normalize_heading(heading_text)
        
        # Check if this heading matches our targets
        if normalized_heading in normalized_targets:
            # Get content until next h4 or end
            current = h4.next_sibling
            section_content = []
            
            while current:
                if current.name == 'h4':
                    break
                if hasattr(current, 'get_text'):
                    section_content.append(current.get_text())
                elif isinstance(current, str):
                    section_content.append(current)
                current = current.next_sibling
            
            content.append(' '.join(section_content))
    
    return ' '.join(content).strip()

# ============================================================================
# SECTION 3: SECTION HEADING DEFINITIONS
# ============================================================================

def preprocess_dataset(csv_path):
    """Preprocess the dataset by extracting sections from HTML content"""
    
    # Define normalized heading categories
    academic_experience_headings = [
        "academicexperience", "acamicexperience", "teachingandresearchexperience",
        "coursesacademicexperience", "executiveeducation", "subject", 
        "acdemicexperience", "academicandresearchexperience",
        "visitingscholaratthefollowinguniversities", "assistantprofessoreconomics",
        "teachingandresearchexperience", "academicandprofessionalexperience",
        "assistantprofessoroperationstechnology", "position",
        "assistantprofessoreconomía", "professionalteachingexperience",
        "experience", "course", "professionalexamination"
    ]

    academic_background_headings = [
        "academicbackground", "education", "acamicbackground",
        "researchareas", "academicandresearchexperience", "awards", 
        "awardsandrecognitions", "selectedpublications", "honorsawards",
        "awardsgrants", "achievements", "educationprofessionalqualifications",
        "academicbackgound", "latestpublications", "publications", 
        "mainpublications", "formaciónacadémica", "academicbackground"
    ]

    corporate_experience_headings = [
        "corporateexperience", "professionalexperience", "corporativeexperience",
        "professionalbackground", "corporateandotherprofessionalexperience",
        "mainprojects", "academicandprofessionalexperience", 
        "professionalteachingexperience", "experience", "industryawards"
    ]

# ============================================================================
# SECTION 4: MAIN PREPROCESSING PIPELINE
# ============================================================================
    
    # Load and process dataset
    df = pd.read_csv(csv_path)
    df_processed = df.copy()
    
    # Extract sections
    df_processed['academic_experience'] = df_processed['full_info'].apply(
        lambda x: extract_section_content(x, academic_experience_headings)
    )
    
    df_processed['academic_background'] = df_processed['full_info'].apply(
        lambda x: extract_section_content(x, academic_background_headings)
    )
    
    df_processed['corporate_experience'] = df_processed['full_info'].apply(
        lambda x: extract_section_content(x, corporate_experience_headings)
    )

# ============================================================================
# SECTION 5: OUTPUT AND STATISTICS
# ============================================================================
    
    # Save processed dataset
    import os
    filename = os.path.basename(csv_path).replace('.csv', '_processed.csv')
    output_path = os.path.join(os.path.dirname(csv_path), filename)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Processed {len(df_processed)} rows")
    print(f"Academic Experience sections found: {(df_processed['academic_experience'] != '').sum()}")
    print(f"Academic Background sections found: {(df_processed['academic_background'] != '').sum()}")
    print(f"Corporate Experience sections found: {(df_processed['corporate_experience'] != '').sum()}")
    print(f"Saved to: {output_path}")
    
    return df_processed