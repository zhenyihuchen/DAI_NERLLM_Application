"""
HYBRID NER PROCESSOR FOR ACADEMIC BIOGRAPHIES
============================================
Combines BERT NER with regex patterns to extract structured entities from professor biographies.
Uses BERT for entity detection and regex for structure parsing and validation.
"""

import pandas as pd
import re
import json
import unicodedata
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: INITIALIZATION & CONFIGURATION
# ============================================================================

class HybridNERProcessor:
    def __init__(self):
        """Initialize BERT model and regex patterns"""
        print("Loading NER model...")
        
        # BERT NER Pipeline
        self.ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        self.MIN_SCORE = 0.85  # Confidence threshold
        
        # Initialize regex patterns
        self._init_regex_patterns()
        
        # Initialize aliases and seen lines tracker
        self.ALIASES = {
            "g.e.": "ge",
            "mac tac": "mactac", 
            "pc componentes": "pccomponentes",
            "ie": "ie business school"
        }
        self.seen_lines = set()
        
        print("NER model loaded!")
    
    def _init_regex_patterns(self):
        """Initialize all regex patterns used for entity extraction"""
        
        # DEGREE PATTERNS
        self.DEGREE = re.compile(r"""(?ix)
            \b(
                ph\.?\s*d\.?           |  # PhD, Ph.D., Ph D
                m\.?\s*b\.?\s*a\.?     |  # MBA, M.B.A., M B A
                m\.?\s*a\.?            |  # MA, M.A., M A
                m\.?\s*sc\.?           |  # MSc, M.Sc., M Sc
                mphil\.?               |  # MPhil
                ll\.?\s*m\.?           |  # LLM, LL.M.
                b\.?\s*a\.?            |  # BA, B.A.
                b\.?\s*sc\.?           |  # BSc, B.Sc.
                b\.?\s*e\.?            |  # BE, B.E.
                (?:bachelor(?:'s|s)?\s+(?:of|in)\s+[^,;]+) |
                (?:master(?:'s|s)?\s+(?:of|in)\s+[^,;]+)   |
                industrial\s+engineer
            )\b
        """)
        
        # YEAR PATTERNS (handles ranges like "2020-Present")
        self.YEAR = re.compile(r"\b(19|20)\d{2}\b(?:\s*[–-]\s*(?:Present|\b(19|20)\d{2}\b))?", re.I)
        
        # UNIVERSITY DETECTION
        self.UNI_HINT = re.compile(r"\b(University|College|School|Institute|Polytechnic|Pontifical|Tecnun|IESE)\b", re.I)
        self.LAB_BLACKLIST = re.compile(r"\b(Laboratory|Centre|Center|Panel|Committee|Board|Initiative)\b", re.I)
        
        # FIELD EXTRACTION (from degrees like "PhD in Computer Science")
        self.FIELD_FROM_DEG = re.compile(r"\b(?:in|of)\s+([^,;]+)", re.I)
        
        # LINE TYPE CLASSIFICATION
        self.ROLE_WORDS = r"(Director|Manager|Partner|Analyst|CEO|CTO|Professor|Adjunct|Assistant|Fellow|Counsel|Engineer|Officer)"
        self.COURSE_HINTS = r"(Professor of|teaches|taught|Course|Courses|Program|curriculum)"
        
        # CORPORATE LINE STRUCTURE (Position, Company, Location, Years)
        self.CORP_LINE = re.compile(
            r"^\s*[^,]+,\s*[^,]+,\s*[^,]+(?:,\s*(?:\b(19|20)\d{2}\b(?:\s*[–-]\s*(?:Present|\b(19|20)\d{2}\b))?\s*))?$",
            re.I
        )
        
        # ORGANIZATION CONNECTORS (for merging BERT entities)
        self.CONNECT = re.compile(r"\s*(?:/|&|and|of|–|-|,)\s*", re.I)
        
        # TEXT CLEANING PATTERNS
        self.LEGAL_SUFFIX = re.compile(r"\b(inc\.?|llc|ltd\.?|s\.?a\.?|gmbh|co\.|corp\.|company|ag|plc|s\.?r\.?l\.?)\b", re.I)
        self.NOISE = re.compile(r"[''\"()\-–—]")

# ============================================================================
# SECTION 2: HTML PROCESSING & LINE EXTRACTION
# ============================================================================

    def extract_lines_from_html(self, html_text):
        """Extract clean text lines from HTML content"""
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
    
    def line_type(self, text):
        """Classify line type for targeted extraction"""
        s = text.strip()
        
        # Academic background (degrees + universities + years)
        if self.DEGREE.search(s) or (re.search(r"\b(University|College|School|Institute)\b", s, re.I) and self.YEAR.search(s)):
            return "studies"
        
        # Teaching experience (courses)
        if re.search(self.COURSE_HINTS, s, re.I):
            return "courses"
        
        # Corporate experience (structured lines with roles)
        if re.search(self.ROLE_WORDS + r".*,", s, re.I):
            return "corporate"
        
        # General university mentions
        if re.search(r"\b(University|College|School|Institute)\b", s, re.I):
            return "university"
        
        return "other"

# ============================================================================
# SECTION 3: BERT ENTITY PROCESSING
# ============================================================================

    def merge_adjacent_orgs(self, spans, text):
        """Merge adjacent ORG entities separated by connectors"""
        orgs = [s for s in spans if s["entity_group"] == "ORG"]
        if not orgs:
            return []
        
        orgs.sort(key=lambda x: x["start"])
        merged = []
        cur = orgs[0]
        
        for nxt in orgs[1:]:
            gap = text[cur["end"]:nxt["start"]]
            if self.CONNECT.fullmatch(gap or ""):
                # Merge: "Massachusetts" + " Institute " + "of" + " Technology"
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

# ============================================================================
# SECTION 4: VALIDATION FUNCTIONS
# ============================================================================

    def is_university_like(self, name):
        """Check if BERT-detected organization is actually a university"""
        n = name.strip()
        return bool(self.UNI_HINT.search(n)) and not self.LAB_BLACKLIST.search(n)
    
    def is_valid_location(self, name):
        """Check if BERT-detected entity is a valid location"""
        if not name or len(name) < 2:
            return False
        
        # Filter out academic fields and other non-locations
        bad_location = r"\b(Studies|Science|Engineering|Business|Administration|Economics|Law|Design|Arts|Technology|Protocol|Initiative|Panel|Committee)\b"
        if re.search(bad_location, name, re.I):
            return False
        return True
    
    def is_valid_company(self, name):
        """Check if BERT-detected organization is a valid company"""
        if not name or len(name) < 3:
            return False
        
        # Filter out panels, committees, and descriptive text
        bad_patterns = r"\b(Panel|Initiative|Committee|Board|where|a company|the|Expert|Experts|Digital|Economy)\b"
        if len(name.split()) > 4 or name.startswith(("where", "a company", "the")) or re.search(bad_patterns, name, re.I):
            return False
        return True

# ============================================================================
# SECTION 5: ENTITY EXTRACTION METHODS
# ============================================================================

    def extract_academic_background(self, line, orgs, locs):
        """Extract degrees, universities, and years from academic background lines"""
        entities = []
        
        # Find degrees and years using regex
        deg_matches = list(self.DEGREE.finditer(line))
        years = [m.group(0) for m in self.YEAR.finditer(line)]
        
        # Find universities using BERT + validation
        for o in orgs:
            if self.is_university_like(o["word"]):
                loc = None
                if locs:
                    # Find nearest location to this university
                    near = min(locs, key=lambda l: abs(l["start"] - o["start"]))
                    if self.is_valid_location(near["word"]):
                        loc = near["word"].strip()
                entities.append({
                    "type": "university",
                    "name": o["word"].strip(),
                    "location": loc
                })
        
        # Add degrees to education
        for dm in deg_matches:
            deg_text = dm.group(0).strip()
            entities.append({
                "type": "education",
                "degree": deg_text
            })
        
        # Add years to periods
        for year in years:
            entities.append({
                "type": "period",
                "year": year
            })
        
        # Add locations separately
        for l in locs:
            if self.is_valid_location(l["word"]):
                entities.append({
                    "type": "location",
                    "name": l["word"].strip()
                })
        
        return entities
    
    def extract_corporate_experience(self, line, orgs, locs):
        """Extract companies and locations from corporate experience lines"""
        # Only process structured corporate lines (Position, Company, Location, Years)
        if not re.search(r"^[^,]+,\s*[^,]+(?:,\s*[^,]+)*$", line) or len(line.split(",")) < 2:
            return []
        
        parts = [p.strip() for p in line.split(",")]
        
        # Extract company (usually in second position)
        company = None
        if len(parts) >= 2:
            seg = parts[1]
            # Find BERT-detected organization in this segment
            for o in orgs:
                if o["word"] in seg and self.is_valid_company(o["word"]):
                    company = o["word"].strip()
                    break
            # Fallback to segment text if it looks like a company
            if not company and self.is_valid_company(seg):
                company = seg
        
        # Extract location (usually in third position or last non-year position)
        location = None
        if len(parts) >= 3:
            tail = parts[-2] if self.YEAR.search(parts[-1] or "") else parts[-1]
            # Find BERT-detected location in this segment
            for l in locs:
                if l["word"] in tail and len(l["word"]) <= len(tail):
                    location = l["word"].strip()
                    break
            # Fallback to known countries
            if not location:
                country_match = re.search(r"\b(USA|UK|U\.K\.|UAE|Spain|Belgium|Mexico|Colombia|Germany|France|Italy)\b", tail, re.I)
                if country_match:
                    location = country_match.group(1)
        
        return [{"type": "company", "name": company, "location": location}] if company else []
    
    def extract_courses(self, line, orgs):
        """Extract courses and teaching institutions"""
        # Find course pattern: "Professor of X" or "teaches Y"
        m = re.search(r"(?:Professor of|teaches?|taught|lectures?\s+on)\s+(.+?)(?:\s+at\s+[^,.;]+)?(?:[,.;]|$)", line, re.I)
        if not m:
            return []
        
        # Split multiple courses
        chunk = m.group(1).strip()
        items = re.split(r"\s*(?:,| and | & )\s*", chunk)
        items = [i.strip() for i in items if i.strip() and len(i.strip()) > 2]
        
        # Find teaching institution
        inst = None
        m2 = re.search(r"\bat\s+([A-Z][^,.;]*(?:University|School|College|Institute))", line, re.I)
        if m2:
            inst = m2.group(1).strip()
        elif orgs:
            # Use first BERT-detected organization
            inst = orgs[0]["word"]
        
        return [{"type": "course", "title": t, "institution": inst} for t in items]
    
    # Removed extract_field method - no longer needed

# ============================================================================
# SECTION 6: MAIN EXTRACTION LOGIC
# ============================================================================

    def extract_entities_from_line(self, line, line_type):
        """Main extraction method - routes to specific extractors based on line type"""
        
        # Run BERT NER on the line
        spans = [s for s in self.ner_pipeline(line) if s["score"] >= self.MIN_SCORE]
        orgs = self.merge_adjacent_orgs(spans, line)
        locs = [s for s in spans if s["entity_group"] in ("LOC", "MISC")]
        
        # Route to appropriate extractor
        if line_type == "studies":
            return self.extract_academic_background(line, orgs, locs)
        elif line_type == "corporate":
            return self.extract_corporate_experience(line, orgs, locs)
        elif line_type == "courses":
            return self.extract_courses(line, orgs)
        elif line_type == "university":
            # General university extraction
            entities = []
            for o in orgs:
                if self.is_university_like(o["word"]) and len(o["word"]) >= 3:
                    loc = None
                    if locs:
                        near = min(locs, key=lambda l: abs(l["start"] - o["start"]))
                        if self.is_valid_location(near["word"]):
                            loc = near["word"].strip()
                    entities.append({"type": "university", "name": o["word"].strip(), "location": loc})
            return entities
        
        return []

# ============================================================================
# SECTION 7: OUTPUT FORMATTING & DEDUPLICATION
# ============================================================================

    def format_structured_output(self, entities):
        """Convert extracted entities to required structure"""
        structured_result = {
            "academic_experience": {
                "Course": [],
                "Program": [],
                "Organization": []  # Teaching universities
            },
            "academic_background": {
                "Organization": [],  # Study universities
                "Location": [],
                "Education": [],
                "Period": []
            },
            "corporate_experience": {
                "Organization": [],  # Companies
                "Location": []
            }
        }
        
        for entity in entities:
            # Academic Experience
            if entity["type"] == "course":
                structured_result["academic_experience"]["Course"].append(entity["title"])
                if entity.get("institution"):
                    structured_result["academic_experience"]["Organization"].append(entity["institution"])
            
            # Academic Background
            elif entity["type"] == "education":
                structured_result["academic_background"]["Education"].append(entity["degree"])
            
            elif entity["type"] == "university":
                structured_result["academic_background"]["Organization"].append(entity["name"])
            
            elif entity["type"] == "location":
                structured_result["academic_background"]["Location"].append(entity["name"])
            
            elif entity["type"] == "period":
                structured_result["academic_background"]["Period"].append(entity["year"])
            
            # Corporate Experience
            elif entity["type"] == "company":
                structured_result["corporate_experience"]["Organization"].append(entity["name"])
                if entity.get("location"):
                    structured_result["corporate_experience"]["Location"].append(entity["location"])
        
        # Remove duplicates and None values
        for section in structured_result.values():
            for key, value_list in section.items():
                section[key] = list(set(filter(None, value_list)))
        
        return structured_result
    
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
    
    def unique_lines(self, lines):
        """Remove duplicate lines"""
        uniq = []
        for ln in lines:
            k = self.canon(ln)
            if k and k not in self.seen_lines:
                uniq.append(ln)
                self.seen_lines.add(k)
        return uniq

# ============================================================================
# SECTION 8: MAIN PROCESSING METHOD
# ============================================================================

    def process_professor(self, html_content, prof_id, alias):
        """Main method to process a single professor's biography"""
        
        # Extract lines from HTML
        lines = self.extract_lines_from_html(html_content)
        
        all_entities = []
        
        # Process each line
        for line in lines:
            line_type = self.line_type(line)
            entities = self.extract_entities_from_line(line, line_type)
            all_entities.extend(entities)
        
        # Format to required structure
        structured_result = self.format_structured_output(all_entities)
        
        return {
            "id": prof_id,
            "alias": alias,
            **structured_result
        }

# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    processor = HybridNERProcessor()
    
    # Load data
    df = pd.read_csv('data/teachers_db_practice.csv')
    
    results = []
    
    for idx, row in df.iterrows():
        print(f"\nProcessing Professor {idx+1}: {row.get('alias', 'Unknown')}")
        
        html_content = row.get('full_info', '')
        prof_id = row.get('id', idx)
        alias = row.get('alias', f'Professor_{idx}')
        
        if html_content:
            result = processor.process_professor(html_content, prof_id, alias)
            results.append(result)
            
            # Display summary
            print(f"Academic Experience - Courses: {len(result['academic_experience']['Course'])}, Organizations: {len(result['academic_experience']['Organization'])}")
            print(f"Academic Background - Organizations: {len(result['academic_background']['Organization'])}, Education: {len(result['academic_background']['Education'])}")
            print(f"Corporate Experience - Organizations: {len(result['corporate_experience']['Organization'])}")
    
    print(f"\nProcessed {len(results)} professors.")
    
    # Save results
    with open('professor_entities_structured.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to professor_entities_structured.json")