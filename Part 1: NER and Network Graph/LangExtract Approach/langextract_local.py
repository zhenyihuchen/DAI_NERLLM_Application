import langextract as lx
import pandas as pd
import json
import os
import time
import textwrap

def extract_entities_local(df_processed, start_idx=0, batch_size=50):
    """Extract entities using LangExtract with local Ollama model"""
    
    # Initialize components - exact copy from original
    prompt = textwrap.dedent("""\
        Extract academic and professional entities from professor biographies. Use exact text as it appears - do not paraphrase, summarize, or modify.
        
        ENTITY TYPES TO EXTRACT:
        
        ACADEMIC EXPERIENCE:
        - course: Specific subjects/courses taught (e.g., "Computer Vision", "Economics", "Microeconomics")
        - program: Degree levels/programs taught (e.g., "undergraduate", "masters", "MBA")
        - teaching_organization: Universities/schools where they teach (e.g., "IE University", "Kent State University")
        
        ACADEMIC BACKGROUND:
        - education_organization: Universities/institutions where they studied (e.g., "Stanford", "MIT")
        - location: Geographic locations of institutions (e.g., "Spain", "USA", "Boston")
        - education: Degrees and qualifications earned (e.g., "Ph.D. in Economics", "MBA", "B.E. Electrical Engineering")
        - year: Graduation years or study periods (e.g., "2020", "2018", "2000-2004")
        
        CORPORATE EXPERIENCE:
        - company: Employer organizations and companies (e.g., "Google", "McKinsey", "Millwood Inc.")
        - company_location: Workplace locations (e.g., "USA", "Silicon Valley", "New York")
        
        EXTRACTION RULES:
        - Extract from entire biography, regardless of section headings
        - Use exact text as written in the source
        - Include full organization names, not abbreviations
        - Extract complete degree titles with field of study
        - Capture all years mentioned for education or work periods
        """)

    examples = [
        lx.data.ExampleData(
        text="<p> Carrio is a seasoned technology leader, researcher, and academic with extensive experience in Machine Learning, Computer Vision, Robotics and Automation. He currently serves as the CEO of Dronomy, a company specializing in autonomous drone technology, and as an Adjunct Professor at IE University in Spain.</p><p>Previously, he held roles as Lead Data Scientist at Shapelets, Chief Technology Officer at Accurate Quant, and Co-founder &amp; Head of Technology at ThermoHuman, where he played a pivotal role in advancing AI-driven solutions for various industries. His research career includes positions at the Massachusetts Institute of Technology (MIT), the Autonomous System Technologies Research &amp; Integration Laboratory at Arizona State University, and the Centro de Automática y Robótica (UPM-CSIC), where he focused on computer vision applications for aerial robots.</p><p> holds a Ph.D. in Automation and Robotics from the Universidad Politécnica de Madrid and an Industrial Engineering degree from the Universidad de Oviedo. He has received several prestigious awards, including recognition from the Norman Foster Foundation and multiple honors for his work in robotics.</p><p>Beyond his professional work,  is passionate about advancing innovation through research and education. His expertise spans AI-driven automation, computer vision, and data science, and he actively contributes to technological developments in these fields. In addition to his career in technology, he is also a professional jazz pianist and has performed at international jazz festivals, showcasing his artistic talent alongside his scientific and entrepreneurial pursuits.</p><h4><strong>Corporate Experience</strong></h4><p>• CEO, Dronomy, Spain, 2020 – Present</p><p>• Lead Data Scientist, Shapelets, Spain, 2021 – 2024</p><p>• Chief Technology Officer, Accurate Quant, Spain, 2020 – 2021</p><p>• Chief Technology Officer, ThermoHuman, Spain, 2015 – 2021</p><p>• Research Scientist, Centre for Automation and Robotics (UPM-CSIC), Spain, 2013 – 2020</p><p>• Research Scholar, Massachusetts Institute of Technology, USA, 2017 – 2018</p><p>• Research Scholar, Autonomous System Technologies Research &amp; Integration Laboratory, Arizona State University, USA, 2015</p><p>• Researcher, Universidad de Oviedo, Spain, 2012 – 2013</p><h4><strong>Academic Experience</strong></h4><p>• Adjunct Professor of Computer Vision, IE University, Spain, 2022 – Present</p><h4><strong>Academic Background</strong></h4><p>• Ph.D. in Automation and Robotics, Universidad Politécnica de Madrid, Spain, 2020</p><p>• Industrial Engineer, Universidad de Oviedo, Spain, 2012</p>",
        extractions=[
            # COURSE
            lx.data.Extraction(
                extraction_class="course",
                extraction_text="Computer Vision",
                attributes={"type": "subject"}
            ),
            # TEACHING ORGANIZATION
            lx.data.Extraction(
                extraction_class="teaching_organization",
                extraction_text="IE University",
                attributes={"type": "university"}
            ),
            # EDUCATION ORGANIZATIONS
            lx.data.Extraction(
                extraction_class="education_organization",
                extraction_text="Universidad Politécnica de Madrid",
                attributes={"type": "university"}
            ),
            lx.data.Extraction(
                extraction_class="education_organization",
                extraction_text="Universidad de Oviedo",
                attributes={"type": "university"}
            ),
            # LOCATIONS
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Spain",
                attributes={"type": "country"}
            ),
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="USA",
                attributes={"type": "country"}
            ),
            # EDUCATION (DEGREES)
            lx.data.Extraction(
                extraction_class="education",
                extraction_text="Ph.D. in Automation and Robotics",
                attributes={"level": "phd"}
            ),
            lx.data.Extraction(
                extraction_class="education",
                extraction_text="Industrial Engineer",
                attributes={"level": "bachelor"}
            ),
            # PERIODS (GRADUATION YEARS)
            lx.data.Extraction(
                extraction_class="period",
                extraction_text="2020",
                attributes={"type": "graduation"}
            ),
            lx.data.Extraction(
                extraction_class="period",
                extraction_text="2012",
                attributes={"type": "graduation"}
            ),
            # COMPANIES
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Dronomy",
                attributes={"type": "corporation"}
            ),
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Shapelets",
                attributes={"type": "corporation"}
            ),
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Accurate Quant",
                attributes={"type": "corporation"}
            ),
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="ThermoHuman",
                attributes={"type": "corporation"}
            ),
        ]
        )
    ]
    
    # Process batch
    end_idx = min(start_idx + batch_size, len(df_processed))
    df_batch = df_processed.iloc[start_idx:end_idx]
    print(f"Processing batch: rows {start_idx}-{end_idx-1} ({len(df_batch)} rows)")
    print("Using local Ollama model - no rate limits!")
    
    results = []
    
    for batch_idx, (idx, row) in enumerate(df_batch.iterrows()):
        print(f"Processing row {idx} (batch row {batch_idx + 1}/{len(df_batch)})")
        
        # Initialize result structure
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
        
        # Extract entities from full_info column
        if pd.notna(row['full_info']) and row['full_info'].strip():
            try:
                result = lx.extract(
                    text_or_documents=row['full_info'],
                    prompt_description=prompt,
                    examples=examples,
                    model_id="gemma2:2b",  # Local Ollama model
                    model_url="http://localhost:11434",
                    fence_output=False,
                    use_schema_constraints=False
                )
                
                # Convert to structured format
                for extraction in result.extractions:
                    entity_class = extraction.extraction_class
                    entity_text = extraction.extraction_text
                    
                    if entity_class == "course":
                        row_result["academic_experience"]["Course"].append(entity_text)
                    elif entity_class == "program":
                        row_result["academic_experience"]["Program"].append(entity_text)
                    elif entity_class == "teaching_organization":
                        row_result["academic_experience"]["Organization"].append(entity_text)
                    elif entity_class == "education_organization":
                        row_result["academic_background"]["Organization"].append(entity_text)
                    elif entity_class == "location":
                        row_result["academic_background"]["Location"].append(entity_text)
                    elif entity_class == "education":
                        row_result["academic_background"]["Education"].append(entity_text)
                    elif entity_class == "period":
                        row_result["academic_background"]["Period"].append(entity_text)
                    elif entity_class == "company":
                        row_result["corporate_experience"]["Organization"].append(entity_text)
                    elif entity_class == "company_location":
                        row_result["corporate_experience"]["Location"].append(entity_text)
                
                # Remove duplicates
                for section in [row_result["academic_experience"], row_result["academic_background"], row_result["corporate_experience"]]:
                    for key, value_list in section.items():
                        section[key] = list(set(value_list))
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        results.append(row_result)
    
    return results

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/teachers_db_practice_processed.csv')
    
    # Test with first 5 rows
    results = extract_entities_local(df, start_idx=0, batch_size=5)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/langextract_local_test.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Local extraction complete!")