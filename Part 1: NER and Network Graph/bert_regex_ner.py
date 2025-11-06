import pandas as pd
import re
import json
import unicodedata
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class HybridNERProcessor:
    def __init__(self):
        print("Loading NER model...")
        self.ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        self.MIN_SCORE = 0.85
        
        # Improved patterns for classification
        self.ROLE_WORDS = r"(Director|Manager|Partner|Analyst|CEO|CTO|Professor|Adjunct|Assistant|Fellow|Counsel|Engineer|Officer)"
        
        # Tightened degree regex
        self.DEGREE = re.compile(r"""(?ix)
            \b(
                ph\.?\s*d\.?           |  # PhD
                m\.?\s*b\.?\s*a\.?     |  # M.B.A. / MBA
                m\.?\s*a\.?            |  # M.A.
                m\.?\s*sc\.?           |  # M.Sc. / MSc
                mphil\.?               |
                ll\.?\s*m\.?           |  # LL.M.
                b\.?\s*a\.?            |  # B.A.
                b\.?\s*sc\.?           |  # B.Sc.
                b\.?\s*e\.?            |  # B.E.
                (?:bachelor(?:'s|s)?\s+(?:of|in)\s+[^,;]+) |
                (?:master(?:'s|s)?\s+(?:of|in)\s+[^,;]+)   |
                industrial\s+engineer
            )\b
        """)
        
        self.YEAR = re.compile(r"\b(19|20)\d{2}\b(?:\s*[–-]\s*(?:Present|\b(19|20)\d{2}\b))?", re.I)
        
        # University detection patterns
        self.UNI_HINT = re.compile(r"\b(University|College|School|Institute|Polytechnic|Pontifical|Tecnun|IESE)\b", re.I)
        self.LAB_BLACKLIST = re.compile(r"\b(Laboratory|Centre|Center|Panel|Committee|Board|Initiative)\b", re.I)
        self.FIELD_FROM_DEG = re.compile(r"\b(?:in|of)\s+([^,;]+)", re.I)
        
        # Corporate line pattern
        self.CORP_LINE = re.compile(
            r"^\s*[^,]+,\s*[^,]+,\s*[^,]+(?:,\s*(?:\b(19|20)\d{2}\b(?:\s*[–-]\s*(?:Present|\b(19|20)\d{2}\b))?\s*))?$",
            re.I
        )
        self.COURSE_HINTS = r"(Professor of|teaches|taught|Course|Courses|Program|curriculum)"
        
        # ORG connection patterns
        self.CONNECT = re.compile(r"\s*(?:/|&|and|of|–|-|,)\s*", re.I)
        
        # Canonicalization patterns
        self.LEGAL_SUFFIX = re.compile(r"\b(inc\.?|llc|ltd\.?|s\.?a\.?|gmbh|co\.|corp\.|company|ag|plc|s\.?r\.?l\.?)\b", re.I)
        self.NOISE = re.compile(r"[''\"()\-–—]")
        
        # Aliases
        self.ALIASES = {
            "g.e.": "ge",
            "mac tac": "mactac", 
            "pc componentes": "pccomponentes",
            "ie": "ie business school"
        }
        
        self.seen_lines = set()
        print("NER model loaded!")
    
    def line_type(self, text):
        """Classify line type without relying on headers"""
        s = text.strip()
        if self.DEGREE.search(s) or (re.search(r"\b(University|College|School|Institute)\b", s, re.I) and self.YEAR.search(s)):
            return "studies"
        if re.search(self.COURSE_HINTS, s, re.I):
            return "courses"
        if re.search(self.ROLE_WORDS + r".*,", s, re.I):
            return "corporate"
        if re.search(r"\b(University|College|School|Institute)\b", s, re.I):
            return "university"
        return "other"
    
    def canon(self, s):
        """Canonicalize text for deduplication"""
        if not s:
            return None
        x = unicodedata.normalize("NFKC", s).lower().strip()
        x = self.NOISE.sub(" ", x)
        x = self.LEGAL_SUFFIX.sub("", x)
        x = re.sub(r"\s*&\s*", " & ", x)
        x = re.sub(r"\s{2,}", " ", x).strip().strip(",.")
        x = self.ALIASES.get(x, x)
        return x
    
    def dedup_entities(self, items, key_fields, prefer_fields=None):
        """Deduplicate entities based on canonicalized keys"""
        if prefer_fields is None:
            prefer_fields = []
        out, seen = [], {}
        
        for item in items:
            key = tuple(self.canon(str(item.get(f, ""))) for f in key_fields)
            if key not in seen:
                seen[key] = item
            else:
                cur = seen[key]
                for f in prefer_fields:
                    if not cur.get(f) and item.get(f):
                        cur[f] = item[f]
        return list(seen.values())
    
    def unique_lines(self, lines):
        """Remove duplicate lines"""
        uniq = []
        for ln in lines:
            k = self.canon(ln)
            if k and k not in self.seen_lines:
                uniq.append(ln)
                self.seen_lines.add(k)
        return uniq
    
    def extract_lines_from_html(self, html_text):
        """Extract lines from HTML, preferring bullets"""
        if pd.isna(html_text):
            return []
        
        soup = BeautifulSoup(html_text, 'html.parser')
        lines = []
        
        for element in soup.find_all(['p', 'li']):
            text = element.get_text().strip()
            if not text:
                continue
                
            # Split by bullet indicators if present
            if '•' in text or '·' in text:
                bullet_splits = re.split(r'[•·]\s*', text)
                for bullet in bullet_splits:
                    bullet = bullet.strip()
                    if len(bullet) > 15:
                        lines.append(bullet)
            else:
                # Split by sentences if no bullets
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 15:
                        lines.append(sent)
        
        return self.unique_lines(lines)
    
    def merge_adjacent_orgs(self, spans, text):
        """Merge adjacent ORG spans separated by connectors"""
        orgs = [s for s in spans if s["entity_group"] == "ORG"]
        if not orgs:
            return []
        orgs.sort(key=lambda x: x["start"])
        merged = []
        cur = orgs[0]
        for nxt in orgs[1:]:
            gap = text[cur["end"]:nxt["start"]]
            if self.CONNECT.fullmatch(gap or ""):
                cur = {
                    **cur,
                    "end": nxt["end"],
                    "word": text[cur["start"]:nxt["end"]]
                }
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged
    
    def is_university_like(self, name):
        """Check if name looks like a university"""
        n = name.strip()
        return bool(self.UNI_HINT.search(n)) and not self.LAB_BLACKLIST.search(n)
    
    def is_valid_location(self, name):
        """Check if name looks like a valid location"""
        if not name or len(name) < 2:
            return False
        # Filter out academic fields, protocols, and other non-location terms
        bad_location = r"\b(Studies|Science|Engineering|Business|Administration|Economics|Law|Design|Arts|Technology|Protocol|Initiative|Panel|Committee)\b"
        if re.search(bad_location, name, re.I):
            return False
        return True
    
    def extract_field(self, degree_text, line):
        """Extract field from degree text or line"""
        m = self.FIELD_FROM_DEG.search(degree_text)
        if m:
            return m.group(1).strip()
        m2 = self.FIELD_FROM_DEG.search(line)
        return m2.group(1).strip() if m2 else None
    
    def is_valid_company(self, name):
        """Check if name looks like a valid company"""
        if not name or len(name) < 3:
            return False
        # Filter out sentences, descriptive text, and panels/initiatives
        bad_patterns = r"\b(Panel|Initiative|Committee|Board|where|a company|the|Expert|Experts|Digital|Economy)\b"
        if len(name.split()) > 4 or name.startswith(("where", "a company", "the")) or re.search(bad_patterns, name, re.I):
            return False
        return True
    
    def parse_corporate_line(self, text, orgs, locs):
        """Parse corporate line with comma structure"""
        # Only process lines that look like proper corporate entries
        if not re.search(r"^[^,]+,\s*[^,]+(?:,\s*[^,]+)*$", text) or len(text.split(",")) < 2:
            return []
        
        parts = [p.strip() for p in text.split(",")]
        
        company = None
        if len(parts) >= 2:
            seg = parts[1]
            # Find best ORG match in this segment
            for o in orgs:
                if o["word"] in seg and self.is_valid_company(o["word"]):
                    company = o["word"].strip()
                    break
            # Fallback to segment if it looks like a company
            if not company and self.is_valid_company(seg):
                company = seg
        
        location = None
        if len(parts) >= 3:
            tail = parts[-2] if self.YEAR.search(parts[-1] or "") else parts[-1]
            # Find location in this segment
            for l in locs:
                if l["word"] in tail and len(l["word"]) <= len(tail):
                    location = l["word"].strip()
                    break
            # Fallback to known countries/regions
            if not location:
                country_match = re.search(r"\b(USA|UK|U\.K\.|UAE|Spain|Belgium|Mexico|Colombia|Germany|France|Italy)\b", tail, re.I)
                if country_match:
                    location = country_match.group(1)
        
        return [{"type": "company", "name": company, "location": location}] if company else []
    
    def extract_courses(self, line, orgs):
        """Extract courses from line"""
        m = re.search(r"(?:Professor of|teaches?|taught|lectures?\s+on)\s+(.+?)(?:\s+at\s+[^,.;]+)?(?:[,.;]|$)", line, re.I)
        if not m:
            return []
        chunk = m.group(1).strip()
        items = re.split(r"\s*(?:,| and | & )\s*", chunk)
        items = [i.strip() for i in items if i.strip() and len(i.strip()) > 2]
        
        # Extract institution name only (not full description)
        inst = None
        m2 = re.search(r"\bat\s+([A-Z][^,.;]*(?:University|School|College|Institute))", line, re.I)
        if m2:
            inst = m2.group(1).strip()
        elif orgs:
            inst = orgs[0]["word"]
        
        return [{"type": "course", "title": t, "institution": inst} for t in items]
    
    def extract_entities_from_line(self, line, line_type):
        """Extract entities from a single line based on its type"""
        spans = [s for s in self.ner_pipeline(line) if s["score"] >= self.MIN_SCORE]
        orgs = self.merge_adjacent_orgs(spans, line)
        locs = [s for s in spans if s["entity_group"] in ("LOC", "MISC")]
        
        out = []
        
        if line_type == "corporate":
            out.extend(self.parse_corporate_line(line, orgs, locs))
            return out
        
        if line_type == "studies":
            # Extract degrees with years
            deg_matches = list(self.DEGREE.finditer(line))
            years = [m.group(0) for m in self.YEAR.finditer(line)]
            
            # Add universities present on this line
            for o in orgs:
                if self.is_university_like(o["word"]):
                    loc = None
                    if locs:
                        near = min(locs, key=lambda l: abs(l["start"] - o["start"]))
                        if self.is_valid_location(near["word"]):
                            loc = near["word"].strip()
                    out.append({"type": "university", "name": o["word"].strip(), "location": loc})
            
            # Studies with field + institutions
            insts = [o["word"].strip() for o in orgs if self.is_university_like(o["word"])]
            for dm in deg_matches:
                deg_text = dm.group(0).strip()
                field = self.extract_field(deg_text, line)
                out.append({
                    "type": "study",
                    "degree": deg_text,
                    "field": field,
                    "institution": insts[0] if insts else None,  # Take first institution
                    "years": years[0] if years else None  # Take first year
                })
            return out
        
        if line_type == "courses":
            out.extend(self.extract_courses(line, orgs))
            return out
        
        if line_type == "university":
            for o in orgs:
                if self.is_university_like(o["word"]) and len(o["word"]) >= 3:
                    loc = None
                    if locs:
                        near = min(locs, key=lambda l: abs(l["start"] - o["start"]))
                        if self.is_valid_location(near["word"]):
                            loc = near["word"].strip()
                    out.append({"type": "university", "name": o["word"].strip(), "location": loc})
            return out
        
        return out
    
    def process_professor(self, html_content, prof_id, alias):
        """Process single professor with improved line-based extraction"""
        lines = self.extract_lines_from_html(html_content)
        
        universities = []
        studies = []
        companies = []
        courses = []
        
        def join_or_none(v):
            if not v:
                return None
            return ", ".join(sorted(v)) if isinstance(v, list) else str(v)
        
        for line in lines:
            line_type = self.line_type(line)
            entities = self.extract_entities_from_line(line, line_type)
            
            for entity in entities:
                if entity["type"] == "university":
                    universities.append({
                        "name": entity["name"],
                        "location": entity["location"]
                    })
                elif entity["type"] == "study":
                    studies.append({
                        "degree": entity["degree"],
                        "field": entity.get("field"),
                        "institution": entity.get("institution"),
                        "years": entity.get("years")
                    })
                elif entity["type"] == "company":
                    companies.append({
                        "name": entity["name"],
                        "location": entity["location"]
                    })
                elif entity["type"] == "course":
                    courses.append({
                        "title": entity["title"],
                        "program_name": entity["institution"]
                    })
        
        # Deduplicate with improved keys - prefer entries with locations
        universities = self.dedup_entities(universities, ("name",), ["location"])
        companies = self.dedup_entities(companies, ("name",), ["location"])
        
        # Studies dedup (keep as strings, not lists)
        studies = self.dedup_entities(studies, ("degree", "field", "institution", "years"))
        
        courses = self.dedup_entities(courses, ("title", "program_name"))
        
        return {
            "id": prof_id,
            "alias": alias,
            "universities": universities,
            "studies": studies,
            "companies": companies,
            "courses": courses
        }

# Process all professors
if __name__ == "__main__":
    processor = HybridNERProcessor()
    
    # Load data
    df = pd.read_csv('data/teachers_db_practice.csv')
    data = df
    
    results = []
    
    for idx, row in data.iterrows():
        print(f"\nProcessing Professor {idx+1}: {row.get('alias', 'Unknown')}")
        
        html_content = row.get('full_info', '')
        prof_id = row.get('id', idx)
        alias = row.get('alias', f'Professor_{idx}')
        
        if html_content:
            result = processor.process_professor(html_content, prof_id, alias)
            results.append(result)
            
            # Display summary only for 50 rows
            print(f"Universities: {len(result['universities'])}, Studies: {len(result['studies'])}, Companies: {len(result['companies'])}, Courses: {len(result['courses'])}")
    
    print(f"\nProcessed {len(results)} professors.")
    
    # Save results
    with open('professor_entities_all.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to professor_entities_all.json")