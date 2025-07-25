# 🧠 DigitAI: A TEI-Aware AI Tutor for XML Encoding

DigitAI is a local RAG (Retrieval-Augmented Generation) system built to help students, archivists, and researchers understand and encode TEI documents. It uses a mix of vector search and graph structure to give helpful, grounded answers without needing the cloud.
---

## 🚀 Features

- 🔎 **Hybrid Retrieval**: Combines semantic embeddings (FAISS + BGE-M3) with graph-based lookup (Neo4j)
- 🧠 **Local LLM-Compatible**: Designed to integrate with local models like Mistral or LLaMA via Ollama
- ⚙️ **Configurable Pipeline**: Driven by `digitaiCore/config.yaml` for easy control over indexing, logging, and Neo4j settings
- 🧪 **Research-Ready**: Supports embedding visualization, document analysis, and RAG-based exploration

---

## 📂 Project Structure

```
digitai/
├── digitaiCore/
│   ├── config.yaml
│   ├── config_loader.py
│   ├── embed_bge_m3.py
│   ├── neo4j_exporter.py
│   ├── faiss_index_builder.py    # [TODO] Build and save FAISS index
│   └── rag_pipeline.py           # [TODO] Hybrid FAISS + Neo4j search
├── data/
│   └── p5/
├── requirements.txt
├── dev-requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repo

```bash
git clone https://github.com/newtfire/digitai.git
cd digitai
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Core Dependencies

```bash
pip install -e .
```



---

## 🔧 Configuration

All settings are controlled in `digitaiCore/config.yaml`. This file lets you manage global options like file paths, logging, Neo4j connection details, and model settings. If you want to change where outputs go, adjust which model is used, or tweak how embeddings run, this is the place to do it.

> 🛑 **Reminder:** Never commit passwords. Use a `.env` file or local override if needed.


---

## 🧪 How to Run the Pipeline

### Export Nodes from Neo4j

```bash
python digitaiCore/neo4j_exporter.py
```

### Embed with BGE-M3 and Save to JSONL

```bash
python digitaiCore/embed_bge_m3.py
```

> Output: `data/p5/neo4jNodeEmbeddings.jsonl` (and FAISS index if configured)

### Run RAG query *(Currently in beta)*

```bash
python digitaiCore/rag_pipeline.py
```
>This grabs the most relevant TEI content using both semantic similarity and structural relationships, builds a custom context window, and sends it to your local LLM to answer or explain.

---

## 🧠 What You Can Use It For

- ✍️ **Ask for Markup Help** — Get suggestions for how to encode specific TEI structures  
- 🛠️ **Debug Your Encoding** — Review or compare your TEI markup against the official specs *(coming soon)*  
- 🎓 **Learn TEI** — Ask questions and get clear, simple explanations using real schema content  

---

## 🔮 What’s Next?

- 🎯 **Fine-Tuning Prep** — Begin curating examples and formatting data for instruction tuning  
- 🧪 **RAG Evaluation** — Test output accuracy and context relevance using real-world queries  
- 🛠️ **Interface Refinement** — Improve prompt formatting, response handling, and context window logic  
- 🧱 **Local.yaml Overrides** — Add clean support for optional secrets/config overrides  
- 📦 **(Optional) Docker Packaging** — Package for easier setup across devices, if needed
---

## 🛠 Current Maintainers & Contributors

### Alexander C. Fisher  
**Role:** Project Lead, Pipeline Developer, and Literature Review Co-Lead  
**Affiliation:** DIGIT Major @ Penn State Behrend  
**GitHub:** [@afish2003](https://github.com/afish2003)

- Designed and built the full Python pipeline: embeddings, FAISS indexing, Neo4j integration, and RAG prompting  
- Leads configuration design, system architecture, and interface logic  
- Will lead the upcoming fine-tuning phase to improve LLM performance  
- Co-leads the literature review, analyzing scholarly sources to inform system design  

---

### Hadleigh Jae Bills  
**Role:** Data Pipeline Lead and Graph Architect  
**Affiliation:** DIGIT Major @ Penn State Behrend  
**GitHub:** [@HadleighJae](https://github.com/HadleighJae)

- Prepares and structures TEI-derived JSON for both vector and graph pipelines  
- Builds and maintains the full Neo4j graph with custom Cypher logic  
- Designs the data model the pipeline relies on and supports structural debugging  
- Leads research on TEI schema logic and contributes key sources for literature review  

---

### Dr. Elisa Beshero-Bondar  
**Role:** Faculty Advisor, XSLT Architect, and Research Lead  
**Affiliation:** Faculty @ Penn State Behrend  
**GitHub:** [@ebeshero](https://github.com/ebeshero)

- Authored the XSLT transformation for converting TEI P5 XML into structured JSON  
- Provides core expertise in TEI, digital editing, and scholarly infrastructure  
- Leads the literature review and guides the team’s research direction  

> This project is part of an ongoing digital humanities research initiative at Penn State Behrend. Please cite responsibly.
---