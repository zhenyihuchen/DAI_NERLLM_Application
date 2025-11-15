# DAI NERLLM Application

This application consists of two main components:
1. **Part 1**: NER (Named Entity Recognition) and Social Network Analysis
2. **Part 2**: LLM-based Answer Evaluator

## Prerequisites

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Part 1: NER and Social Network Analysis

### Overview
Extracts entities (universities, companies, studies, courses) from professor biographies and generates a social network graph.

### How to Run

1. **Navigate to Part 1 directory:**
   ```bash
   cd part1_NER_network_graph
   ```

2. **Run the complete pipeline:**
   ```bash
   python main.py
   ```

### What it does:
- **Step 1**: Preprocesses raw HTML professor biographies
- **Step 2**: Extracts entities using GLiNER model
- **Step 3**: Extracts entities using BERT+Regex hybrid approach
- **Step 4**: Merges results from both approaches with intelligent deduplication
- **Step 5**: Generates social network graph and saves visualization

### Output Files:
- `data/teachers_db_practice_processed.csv` - Preprocessed dataset
- `results/gliner_entities_results.json` - GLiNER extraction results
- `results/bert_regex_entities_results.json` - BERT+Regex extraction results
- `results/merged_entities_results.json` - Final merged results
- `results/professor_network.gexf` - Network graph file
- `results/professor_network.png` - Network visualization

### Note:
The `langextract_test/` folder contains experimental code and is not part of the main pipeline.

---

## Part 2: LLM-based Answer Evaluator

### Overview
Evaluates student answers against reference answers using LLM-based scoring across three aspects: correctness, completeness, and precision.

### How to Run

1. **Navigate to Part 2 directory:**
   ```bash
   cd part2_evaluator/approach2_LLM
   ```

2. **Set up environment variables:**
   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

### What it does:
- Provides a web interface for answer evaluation
- Uses Groq LLM API for intelligent scoring
- Evaluates answers on correctness, completeness, and precision
- Provides detailed feedback and weighted scores
- Saves evaluation sessions for tracking

### Note:
The `approach1_manual/` folder contains experimental manual evaluation code and is not the main approach.

---

## Project Structure

```
DAI_NERLLM_Application/
├── part1_NER_network_graph/          # NER and Network Analysis
│   ├── data/                         # Dataset files
│   ├── results/                      # Output files
│   ├── tools/                        # Analysis utilities
│   ├── langextract_test/            # Experimental (not used in main)
│   └── main.py                      # Main pipeline
├── part2_evaluator/                 # LLM Evaluator
│   ├── approach2_LLM/              # Main LLM approach
│   │   └── app.py                  # Streamlit application
│   ├── approach1_manual/           # Experimental (not used)
│   └── runs/                       # Evaluation sessions
├── data/                           # Raw datasets
└── requirements.txt               # Dependencies
```

## Dependencies

Key packages used:
- `pandas`, `numpy` - Data processing
- `transformers`, `torch` - NLP models
- `gliner` - Entity extraction
- `networkx`, `matplotlib` - Network analysis
- `streamlit` - Web interface
- `groq` - LLM API client
- `beautifulsoup4` - HTML processing