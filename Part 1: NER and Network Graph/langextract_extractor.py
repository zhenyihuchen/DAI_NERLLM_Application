import pandas as pd
import langextract as lx
import textwrap
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('LANGEXTRACT_API_KEY')

def create_extraction_prompt():
    """Create the prompt description for LangExtract"""
    
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
    
    return prompt

def create_few_shot_examples():
    """Create few-shot examples for LangExtract"""
    
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
        # ),
        # lx.data.ExampleData(
        # text=textwrap.dedent("""\
        #     <p> has over 20 years of global experience in sustainability strategy design and implementation; social impact; impact measurement; ESG and innovation management. A highly creative individual with excellent interpersonal and communication skills and a bias towards action,  is an entrepreneur at heart. After starting his professional life at United Nations in New York,  joined a philanthropic foundation in India; started an organization addressing the needs of vulnerable children in South India; developed an incubation and investment  initiative focused on environmental issues in Cambodia; co-founded Impact Hub Phnom Penh in 2013 and served as a Board of Directors of Impact Hub Global Network, one of the world’s largest networks for entrepreneurial communities striving to achieve impact at scale.</p><p> has been leading numerous activities around sustainable development; public policy; entrepreneurship, ESG practice and access to education with international institutions such as UNDP, UNESCO, ILO, and the European Commission; as well as international foundations and organizations (WWF, ACRA, ICS Driver for Change).</p><p>He holds a Global Executive MBA degree from IE Business School and a MSc degree in Economics and International Relations from the Postgraduate Schools of Economics and International Affairs. Since 2018 he teaches on topics of sustainability; leadership; social impact; social entrepreneurship, system thinking and innovation at IE University as well as serving as Academic Director for the IE Sustainability Bootcamp.</p><h4>Corporate Experience</h4><p>• Global University Systems, 2024 - Present</p><p>• Executive Director, Higher Education Partners , US, 2020 - 2024</p><p>• Board of Director, Impact Hub Network, Global, 2018 - 2020</p><p>• Founder, Impact Hub Phnom Penh, Cambodia, 2013 - 2019</p><p>• Consultant, Access to Education, UNESCO, 2018 - 2019</p><p>• Consultant, Education, UNDP, 2017 - 2018</p><p>• Consultant, Upskilling, ILO, 2016 - 2017</p><p>• Director, Asia Debt Management Foundation, India, 2009 - 2012</p><h4>Academic Experience</h4><p>• Academic Director, IE University, Spain, 2024 - Present</p><p>• Adjunct Professor of PM and Impact Assessment, IE University, Spain, 2024 - Present</p><p>• Adjunct Professor of Critical Thinking, IE University, Spain, 2023 - Present</p><p>• Adjunct Professor of Systems Thinking, IE University, Spain, 2023 - Present</p><p>• Adjunct Professor of Sustainability, IE University, Spain, 2021 - Present</p><h4>Academic Background</h4><p>• Global Executive MBA in Business Administration IE Business School, Spain, 2018</p><p>• Master in Economics and International Relations, Università Cattolica, Italy, 2008</p><p>• Master of Arts in English Language and Literature, San Francisco State U, USA, 2006</p><p>• Bachelor in Letters, Trinity College Dublin, Ireland, 2003</p>
        #     """

        # ),
        # extractions=[
        #     # COURSES / SUBJECTS HE TEACHES
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="sustainability",
        #         attributes={"type": "subject"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="leadership",
        #         attributes={"type": "subject"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="social impact",
        #         attributes={"type": "subject"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="social entrepreneurship",
        #         attributes={"type": "subject"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="systems thinking",
        #         attributes={"type": "subject"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="course",
        #         extraction_text="innovation",
        #         attributes={"type": "subject"}
        #     ),
        #     # TEACHING ORGANIZATION
        #     lx.data.Extraction(
        #         extraction_class="teaching_organization",
        #         extraction_text="IE University",
        #         attributes={"type": "university"}
        #     ),
        #     # EDUCATION ORGANIZATIONS
        #     lx.data.Extraction(
        #         extraction_class="education_organization",
        #         extraction_text="IE Business School",
        #         attributes={"type": "university"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education_organization",
        #         extraction_text="Università Cattolica",
        #         attributes={"type": "university"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education_organization",
        #         extraction_text="San Francisco State U",
        #         attributes={"type": "university"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education_organization",
        #         extraction_text="Trinity College Dublin",
        #         attributes={"type": "university"}
        #     ),
        #     # LOCATIONS
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="Spain",
        #         attributes={"type": "country"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="Italy",
        #         attributes={"type": "country"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="USA",
        #         attributes={"type": "country"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="Ireland",
        #         attributes={"type": "country"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="Cambodia",
        #         attributes={"type": "country"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="location",
        #         extraction_text="India",
        #         attributes={"type": "country"}
        #     ),
        #     # EDUCATION (DEGREES)
        #     lx.data.Extraction(
        #         extraction_class="education",
        #         extraction_text="Global Executive MBA in Business Administration",
        #         attributes={"level": "mba"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education",
        #         extraction_text="Master in Economics and International Relations",
        #         attributes={"level": "master"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education",
        #         extraction_text="Master of Arts in English Language and Literature",
        #         attributes={"level": "master"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="education",
        #         extraction_text="Bachelor in Letters",
        #         attributes={"level": "bachelor"}
        #     ),
        #     # PERIODS (GRADUATION YEARS)
        #     lx.data.Extraction(
        #         extraction_class="period",
        #         extraction_text="2018",
        #         attributes={"type": "graduation"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="period",
        #         extraction_text="2008",
        #         attributes={"type": "graduation"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="period",
        #         extraction_text="2006",
        #         attributes={"type": "graduation"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="period",
        #         extraction_text="2003",
        #         attributes={"type": "graduation"}
        #     ),
        #     # COMPANIES / ORGANIZATIONS
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="Global University Systems",
        #         attributes={"type": "corporation"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="Higher Education Partners",
        #         attributes={"type": "corporation"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="Impact Hub Network",
        #         attributes={"type": "organization"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="Impact Hub Phnom Penh",
        #         attributes={"type": "organization"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="UNESCO",
        #         attributes={"type": "international_organization"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="UNDP",
        #         attributes={"type": "international_organization"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="ILO",
        #         attributes={"type": "international_organization"}
        #     ),
        #     lx.data.Extraction(
        #         extraction_class="company",
        #         extraction_text="Asia Debt Management Foundation",
        #         attributes={"type": "foundation"}
        #     ),
        # ]
        # )
    ]
    return examples

def convert_to_structured_format(langextract_result):
    """Convert LangExtract result to GLiNER-compatible format"""
    
    structured_result = {
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
    
    # Map LangExtract entities to structured format
    for extraction in langextract_result.extractions:
        entity_class = extraction.extraction_class
        entity_text = extraction.extraction_text
        
        if entity_class == "course":
            structured_result["academic_experience"]["Course"].append(entity_text)
        elif entity_class == "program":
            structured_result["academic_experience"]["Program"].append(entity_text)
        elif entity_class == "teaching_organization":
            structured_result["academic_experience"]["Organization"].append(entity_text)
        elif entity_class == "education_organization":
            structured_result["academic_background"]["Organization"].append(entity_text)
        elif entity_class == "location":
            structured_result["academic_background"]["Location"].append(entity_text)
        elif entity_class == "education":
            structured_result["academic_background"]["Education"].append(entity_text)
        elif entity_class == "period":
            structured_result["academic_background"]["Period"].append(entity_text)
        elif entity_class == "company":
            structured_result["corporate_experience"]["Organization"].append(entity_text)
        elif entity_class == "company_location":
            structured_result["corporate_experience"]["Location"].append(entity_text)
    
    # Remove duplicates
    for section in structured_result.values():
        for key, value_list in section.items():
            section[key] = list(set(value_list))
    
    return structured_result

def extract_entities_langextract(df_processed, start_idx=0, batch_size=50):
    """Extract entities using LangExtract approach
    
    Args:
        df_processed: DataFrame with full_info column
        start_idx: Starting row index for batch processing
        batch_size: Number of rows to process in this batch
        
    Returns:
        List of dictionaries with extracted entities
    """
    
    # Initialize components
    prompt = create_extraction_prompt()
    examples = create_few_shot_examples()
    
    # Process batch
    end_idx = min(start_idx + batch_size, len(df_processed))
    df_batch = df_processed.iloc[start_idx:end_idx]
    print(f"Processing batch: rows {start_idx}-{end_idx-1} ({len(df_batch)} rows)")
    print("Note: Processing with 7-second delays between requests to respect API limits")
    
    results = []
    
    for batch_idx, (idx, row) in enumerate(df_batch.iterrows()):
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
                    api_key=api_key,
                    extraction_passes=1,
                    max_workers=1
                )
                
                # Rate limiting: wait 7 seconds between requests (10 requests/minute = 6s + buffer)
                time.sleep(7)
                
                structured_entities = convert_to_structured_format(result)
                row_result.update(structured_entities)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    # Retry once
                    try:
                        result = lx.extract(
                            text_or_documents=row['full_info'],
                            prompt_description=prompt,
                            examples=examples,
                            api_key=api_key,
                            extraction_passes=1,
                            max_workers=1
                        )
                        structured_entities = convert_to_structured_format(result)
                        row_result.update(structured_entities)
                        time.sleep(7)
                    except Exception as retry_e:
                        print(f"Retry failed for row {idx}: {retry_e}")
        
        results.append(row_result)
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx} rows in current batch...")
    
    return results

def save_results_to_json(results, filename="langextract_results.json"):
    """Save extraction results to JSON file in results folder"""
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to JSON file
    filepath = os.path.join('results', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filepath}")
    return filepath

def process_full_dataset_in_batches(df_processed, batch_size=50):
    """Process full dataset in batches to respect API limits"""
    
    total_rows = len(df_processed)
    all_results = []
    
    for start_idx in range(0, total_rows, batch_size):
        print(f"\n=== Processing Batch {start_idx//batch_size + 1} ===")
        
        batch_results = extract_entities_langextract(df_processed, start_idx, batch_size)
        all_results.extend(batch_results)
        
        # Save batch results
        batch_filename = f"langextract_batch_{start_idx//batch_size + 1}.json"
        save_results_to_json(batch_results, batch_filename)
        
        # Wait between batches if not the last batch
        if start_idx + batch_size < total_rows:
            print(f"Batch complete. Waiting 60 seconds before next batch...")
            time.sleep(60)
    
    # Save final combined results
    save_results_to_json(all_results, "langextract_full_results.json")
    return all_results


