import langextract as lx
import textwrap
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('LANGEXTRACT_API_KEY')

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
    - period: Graduation years or study periods (e.g., "2020", "2018", "2000-2004")
    
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
        text="Professor teaches Computer Vision at IE University. He worked at Dronomy and holds a Ph.D. from Universidad Politécnica de Madrid.",
        extractions=[
            lx.data.Extraction(
                extraction_class="course",
                extraction_text="Computer Vision"
            ),
            lx.data.Extraction(
                extraction_class="teaching_organization",
                extraction_text="IE University"
            ),
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Dronomy"
            ),
            lx.data.Extraction(
                extraction_class="education_organization",
                extraction_text="Universidad Politécnica de Madrid"
            )
        ]
    )
]

test_text = """<p>Mr. Madgar has been teaching economics part time while working in various professional roles across a wide array of industries. He taught undergraduate microeconomics and macroeconomics at Kent State University for 5 years before advancing to teach economic environment & country economic analysis at the masters' level at IE, where he has been teaching at now going into his third year. He has 21 years of professional work experience ranging from engineering design (where he holds one patent in the US for a 'nail press') to director level positions throughout supply chain and operations management. Having worked for large Fortune 100 companies such as GE and Halliburton to small startups such Millwood Inc., where he currently works, has an extensive background in applied theory in the workplace.</p><h4>Corporate Experience</h4><p>• Strategic Business Unit Manager, Millwood Inc., USA, 2018-Present</p><p>• Global Manufacturing Director, Permasteelisa, USA, 2015-2018</p><p>• Jet Research Center Manager, Halliburton, USA, 2013-2015</p><p>• Corporate Finance Master Six Sigma Black Belt, Cameron 'a Schlumberger Co', USA, 2010-2013</p><p>• Director of Process Technology, Millwood Inc., USA, 2005-2010</p><p>• Corporate Six Sigma Black Belt, MACtac 'a Bemis Co', USA, 2003-2005</p><p>• Six Sigma / Lean Manufacturing Black Belt, G.E., USA, 1999-2003</p><h4>Academic Experience</h4><p>• Adjunct Professor of Economics, IE Business School, Spain, 2018-Present</p><p>• Adjunct Level II Professor of Economics, Kent State University, USA, 2005-2010</p><h4>Academic Background</h4><p>• E.M.B.A., IE Business School / Brown University, Spain / USA, 2018</p><p>• Strategic Decision & Risk Management Certificate, Stanford, USA, 2014</p><p>• Executive Certificate in HR Leadership, Cornell, USA, 2013</p><p>• Lean Six Sigma Master Black Belt Certificate, Villanova, USA, 2011</p><p>• M.A. Financial Economics, Kent State University, USA, 2004</p><p>• M.B.A., Youngstown State University, USA, 2001</p><p>• B.E. Electrical Engineering, Youngstown State University, USA 2000</p>"""

try:
    result = lx.extract(
        prompt_description=prompt,
        text_or_documents=test_text,
        examples=examples,
        api_key=api_key,
        extraction_passes=1,
        max_workers=5
    )
    
    print("Extraction Results:")
    for extraction in result.extractions:
        print(f"- {extraction.extraction_class}: {extraction.extraction_text}")
        
except Exception as e:
    print(f"Error: {e}")