import pandas as pd
import re
from bs4 import BeautifulSoup
from gliner import GLiNER
import json

#ajustar output gliner con entidades que ya tengo 
# ponerlo para que lo haga desde un main
#implenetar resultados en langextract 
# juntar con GLiNER (todos de todos)

#filtrar los valores de cada categorias mas freecuentes para que el graph sea representativo. 
#unique count de todos los valores de cada categoria y cambiar los valores menos frecuentes por "other", threshold = 10% 
#probar el graph 
#usar distancias levenstans para ver si se repiten entidades que son iguales pero que estan expresadas de forma diferente. ej: NY or New York



# 1. Preprocessing of data

def normalize_heading(heading):
    """Normalize heading by removing spaces, tags, punctuation and making lowercase"""
    # Remove HTML tags
    heading = re.sub(r'<[^>]+>', '', heading)
    # Remove spaces and punctuation
    heading = re.sub(r'[^\w]', '', heading)
    # Make lowercase
    return heading.lower()

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

# Load dataset
df = pd.read_csv('../data/teachers_db_practice.csv')

# Create a copy with new columns
df_processed = df.copy()

# Define normalized heading categories (duplicates removed)
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

# Extract sections for full dataset
df_processed['academic_experience'] = df_processed['full_info'].apply(
    lambda x: extract_section_content(x, academic_experience_headings)
)

df_processed['academic_background'] = df_processed['full_info'].apply(
    lambda x: extract_section_content(x, academic_background_headings)
)

df_processed['corporate_experience'] = df_processed['full_info'].apply(
    lambda x: extract_section_content(x, corporate_experience_headings)
)

# Save full processed dataset
df_processed.to_csv('../data/teachers_db_processed.csv', index=False)

print(f"Processed {len(df_processed)} rows")
print(f"Academic Experience sections found: {(df_processed['academic_experience'] != '').sum()}")
print(f"Academic Background sections found: {(df_processed['academic_background'] != '').sum()}")
print(f"Corporate Experience sections found: {(df_processed['corporate_experience'] != '').sum()}")



# # Show sample results
# print("\nSample results:")
# for i in range(3):
#     if i < len(df_processed):
#         print(f"\nRow {i} - {df_processed.iloc[i]['alias']}:")
#         print(f"Academic Experience: {df_processed.iloc[i]['academic_experience'][:100]}...")
#         print(f"Academic Background: {df_processed.iloc[i]['academic_background'][:100]}...")
#         print(f"Corporate Experience: {df_processed.iloc[i]['corporate_experience'][:100]}...")



# 2. NER per field

model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
model.eval()
print("GLiNER model loaded!")

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
    
    # Remove duplicates from each list
    for section in row_result.values():
        if isinstance(section, dict):
            for key, value_list in section.items():
                section[key] = list(set(value_list))  # Remove duplicates
    
    results.append(row_result)
    
    if idx % 50 == 0:
        print(f"Processed {idx} rows...")

# Save results to JSON
with open('gliner_entities_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nCompleted NER extraction for {len(results)} rows")
print("Results saved to gliner_entities_results.json")

# Show sample results
print("\nSample results:")
for i in range(min(3, len(results))):
    print(f"\nRow {i} - {results[i]['alias']}:")
    print(f"Academic Experience: {results[i]['academic_experience']}")
    print(f"Academic Background: {results[i]['academic_background']}")
    print(f"Corporate Experience: {results[i]['corporate_experience']}")




# All headings and their frequencies:
# --------------------------------------------------
# ACADEMIC EXPERIENCE, Academic Experience, <strong>Academic Experience</strong>, Acamic Experience, <strong>Acamic Experience</strong>,   Academic Experience,  <strong>ACADEMIC EXPERIENCE</strong>, Academic experience, <strong>Academic Experience </strong>, Teaching and Research Experience, Acamic experience, COURSES. ACADEMIC EXPERIENCE<strong><br/></strong>, Executive Education, Subject, <strong>Academic Experienc</strong><strong>e</strong>, ACDEMIC EXPERIENCE, Academic and research experience, Visiting Scholar at the following universities, ASSISTANT PROFESSOR, ECONOMICS, TEACHING AND RESEARCH EXPERIENCE, ACADEMIC AND PROFESSIONAL EXPERIENCE, ASSISTANT PROFESSOR, OPERATIONS &amp; TECHNOLOGY, POSITION, <strong>ASSISTANT PROFESSOR, ECONOMÍA</strong>, Professional &amp; Teaching Experience, Academic and Professional Experience, EXPERIENCE, COURSE, Professional Examination

# ACADEMIC BACKGROUND, Academic Background, <strong>Academic Background</strong>, Education, Acamic Background, <strong>Acamic Background</strong>, Academic background, Academic Background, EDUCATION, <strong>Academic Background </strong>, <strong>ACADEMIC BACKGROUND</strong>,   Academic Background, ACADEMIC BACKGROUND<strong><br/></strong>, ACADEMIC BACKGROUND<strong> </strong>,    1 | <strong>Academic Background:</strong>,  1 | Academic Background / Awards,  Research areas, Academic and Research Experience, AWARDS, AWARDS AND RECOGNITIONS, SELECTED PUBLICATIONS, HONORS &amp; AWARDS, <strong>Awards &amp; grants:</strong>, Achievements, Education &amp; Professional Qualifications, ACADEMIC BACKGOUND, LATEST PUBLICATIONS, PUBLICATIONS, MAIN PUBLICATIONS, Latest Publications, Formación Académica, Publications:, Academic background:, 

# Corporate Experience, CORPORATE EXPERIENCE, <strong>Corporate Experience</strong>, Professional Experience, PROFESSIONAL EXPERIENCE, <strong>CORPORATE EXPERIENCE</strong>, Professional experience, Corporate experience, Corporative Experience, Professional Background, CORPORATE EXPERIENCE<strong><br/></strong>,  <strong>PROFESSIONAL EXPERIENCE</strong>, Corporate (and other professional) Experience, Main Projects, ACADEMIC AND PROFESSIONAL EXPERIENCE, <strong><strong>Corporate Experience</strong></strong>, Professional &amp; Teaching Experience, Academic and Professional Experience, EXPERIENCE, Industry Awards


#    7 | Other information of interest
#    4 | Specialties
#    1 | ADDITIONAL INFORMATION
#    1 | Expertise
#    1 | 





