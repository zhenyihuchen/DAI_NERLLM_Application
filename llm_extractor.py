import pandas as pd, json, re
from transformers import pipeline
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

def clean_html_keep_bullets(html):
    if not html: return ""
    soup = BeautifulSoup(html, "html.parser")
    # preserve bullets and commas by grabbing text with separators
    txt = soup.get_text(" ", strip=True)
    # compress whitespace
    txt = re.sub(r"\s{2,}", " ", txt)
    # lightly cap length to keep under T5 token limit (prompt+input <= ~512 for base)
    return txt[:1800]

PROMPT_TEMPLATE = """You extract structured entities about professors from short bios (HTML or text).
Return ONLY JSON BETWEEN <json> and </json> with this exact minified schema:

{{
 "universities":[{{"name":"", "location":""}}],
 "studies":[{{"degree":"", "field":null, "institution":"", "years":""}}],
 "companies":[{{"name":"", "location":""}}],
 "courses":[{{"title":"", "institution":""}}]
}}

Rules:
- Read semantically. For companies, output only company names (no roles).
- A study can be Bachelor/Master/PhD/LL.M./MBA/EMBA/Certificate.
- field = specialization after "in"/"of" in the degree.
- If an item is absent, output [] for that array. Do not guess.
- Split "A / B" universities into separate university entries.
- Locations only if explicitly present.

Example INPUT:
"• Studio Director, A&M Studio, Spain, 2023 – Present. • Studio Director, Vidivixi, Mexico, 2017 – 2023. • Bachelor in Graphic Design, Camberwell College of Arts UAL, U.K., 2013."

Example OUTPUT:
<json>{{"universities":[{{"name":"Camberwell College of Arts UAL","location":"U.K."}}],"studies":[{{"degree":"Bachelor in Graphic Design","field":"Graphic Design","institution":"Camberwell College of Arts UAL","years":"2013"}}],"companies":[{{"name":"A&M Studio","location":"Spain"}},{{"name":"Vidivixi","location":"Mexico"}}],"courses":[]}}</json>

Now extract for this INPUT:
{input_block}
Output only the JSON between <json> and </json>.
"""

class LLMExtractor:
    def __init__(self):
        print("Loading LLM model...")
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            # do NOT set max_length here; we'll control per-call
            do_sample=False
        )
        print("LLM model loaded!")
    
    def _extract_json_between_tags(self, text):
        m = re.search(r"<json>\s*(\{.*\})\s*</json>", text, flags=re.S|re.I)
        if not m:
            return None
        js = m.group(1)
        # light repairs
        js = re.sub(r",\s*}", "}", js)
        js = re.sub(r",\s*]", "]", js)
        try:
            return json.loads(js)
        except Exception:
            return None

    def extract_entities(self, html_text, prof_id, alias):
        if pd.isna(html_text) or not html_text:
            return self._empty_result(prof_id, alias)

        text = clean_html_keep_bullets(html_text)
        prompt = PROMPT_TEMPLATE.format(input_block=text)

        try:
            out = self.llm(
                prompt,
                max_new_tokens=256,   # <-- room to generate
                num_beams=4,
                temperature=0.0,
                top_p=1.0
            )[0]["generated_text"].strip()

            data = self._extract_json_between_tags(out)
            if not data:
                # fallback: try to grab the last {...} block if tags failed
                m = re.search(r"(\{.*\})\s*$", out, flags=re.S)
                if m:
                    try:
                        data = json.loads(m.group(1))
                    except Exception:
                        data = None

            if not data:
                return self._empty_result(prof_id, alias)

            return {
                "id": prof_id,
                "alias": alias,
                "universities": data.get("universities", []),
                "studies": data.get("studies", []),
                "companies": data.get("companies", []),
                "courses": data.get("courses", [])
            }

        except Exception as e:
            print(f"Error processing {alias}: {e}")
            return self._empty_result(prof_id, alias)
    
    def _empty_result(self, prof_id, alias):
        return {"id": prof_id, "alias": alias,
                "universities": [], "studies": [],
                "companies": [], "courses": []}

# Test with first 5 professors
if __name__ == "__main__":
    extractor = LLMExtractor()
    
    # Load data
    df = pd.read_csv('data/teachers_db_practice.csv')
    data = df.head(5)
    
    results = []
    
    for idx, row in data.iterrows():
        print(f"\nProcessing Professor {idx+1}: {row.get('alias', 'Unknown')}")
        
        html_content = row.get('full_info', '')
        prof_id = row.get('id', idx)
        alias = row.get('alias', f'Professor_{idx}')
        
        result = extractor.extract_entities(html_content, prof_id, alias)
        if result:
            results.append(result)
            
            # Display summary
            print(f"Universities: {len(result['universities'])}, Studies: {len(result['studies'])}, Companies: {len(result['companies'])}, Courses: {len(result['courses'])}")
        else:
            print("No result returned")
            
            # Show full result for first professor
            if idx == 0:
                print(json.dumps(result, indent=2))
    
    print(f"\nProcessed {len(results)} professors with LLM extraction")
    
    # Save results
    with open('llm_entities.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to llm_entities.json")