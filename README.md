# COE Digital Enablement Prototype

An AI-powered **Retrieval-Augmented Generation (RAG) assistant** and **COE analytics prototype** designed to support enterprise digital transformation initiatives.

This project demonstrates how organizations can combine **AI-powered knowledge retrieval** with **operational analytics dashboards** to improve decision-making, knowledge access, and process efficiency.

The system allows users to ask questions about operational excellence concepts such as **Lean, Six Sigma, KPIs, and process improvement**, while retrieving evidence directly from internal documents.


# Project Objectives

The goal of this prototype is to demonstrate how AI can assist a **Center of Excellence (COE)** by:

• Providing an intelligent assistant to query internal knowledge  
• Retrieving relevant documentation using semantic search  
• Generating context-aware responses using LLMs  
• Increasing transparency through evidence-based answers  
• Improving operational insight through simple analytics dashboards  

# Enterprise Use Cases

This prototype demonstrates how AI can support enterprise Centers of Excellence by:

- Assisting employees with operational excellence knowledge
- Retrieving internal process documentation
- Supporting Lean and Six Sigma training
- Helping teams track KPIs and improvement initiatives
- Enabling faster knowledge access across departments

# Key Features

### AI Knowledge Assistant (RAG Chatbot)

- Retrieval-Augmented Generation (RAG)
- FAISS vector database for fast semantic search
- Sentence-transformer embeddings
- Evidence-based answers
- Top-3 document retrieval
- Similarity score display
- Confidence labeling (High / Medium / Low)
- Strict mode to prevent hallucinated responses
- Streamlit chat interface

### COE Digital Enablement Concepts Covered

- Lean methodology
- Six Sigma
- KPI monitoring
- Process optimization
- Continuous improvement

# Project Structure


```
coe-digital-enablement/
│
├── app.py
├── rag_pipeline.py
├── embeddings.py
├── vector_store.py
├── dashboard.py
│
├── data/
├── assets/
│
├── requirements.txt
└── README.md
```

# System Architecture

The system follows a **Retrieval-Augmented Generation workflow**.

User questions are first converted into embeddings, relevant document chunks are retrieved using a vector database, and an LLM generates a grounded response based on the retrieved context.

### RAG System Architecture
<img width="524" height="1347" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/8fc15300-b869-439a-b12f-d3284d5c7770" />


### Architecture Walk Through
<img width="2816" height="1536" alt="Gemini_Generated_Image_52no2s52no2s52no" src="https://github.com/user-attachments/assets/97694d34-9759-4b66-a629-8714a8d0d260" />

# Demo

### COE Dashboard
The Streamlit dashboard provides a simple overview of COE initiatives including:

- Initiative status distribution
- KPI achievement tracking
- Top initiatives by business impact
- Data quality indicators

### AI Assistant
The RAG assistant allows users to ask questions about operational excellence concepts such as:

- Lean methodology
- Six Sigma DMAIC framework
- KPI monitoring strategies
- Process improvement practices

### RAG Assistant Interface
<img width="966" height="648" alt="Screenshot 2026-03-06 at 1 28 24 PM" src="https://github.com/user-attachments/assets/ed19a049-5def-4360-adac-8bbaa7a96d80" />

### COE Dashboard
<img width="934" height="679" alt="Screenshot 2026-03-06 at 1 25 34 PM" src="https://github.com/user-attachments/assets/0aa1b1ad-b8f8-48c6-a9e7-369b69d2ef2f" />

Example queries:

- "What is Lean methodology?"
- "Explain DMAIC in Six Sigma"
- "How should a COE track KPIs?"
- "What are common Lean wastes?"

The assistant retrieves relevant knowledge chunks and generates an evidence-based response.

# RAG Pipeline Overview

The system implements a Retrieval-Augmented Generation (RAG) workflow consisting of the following stages:

1. **Document Processing**
   - Internal COE documents are segmented into smaller chunks.

2. **Embedding Generation**
   - Each chunk is converted into vector embeddings using the Sentence Transformers model:
   - `all-MiniLM-L6-v2`

3. **Vector Indexing**
   - Embeddings are stored in a FAISS vector database for efficient semantic search.

4. **Query Processing**
   - User questions are embedded using the same embedding model.

5. **Semantic Retrieval**
   - The system retrieves the **Top-3 most relevant document chunks** using cosine similarity.

6. **Context Construction**
   - Retrieved chunks are combined into a prompt context.

7. **LLM Answer Generation**
   - An LLM generates a grounded response using only the retrieved information.

8. **Governance Layer**
   - Similarity scores determine confidence levels:
     - High
     - Medium
     - Low

9. **Final Response**
   - The system returns:
     - AI-generated answer
     - evidence citations
     - similarity confidence score

# Technology Stack

| Layer           | Technology                               |
| --------------- | ---------------------------------------- |
| Frontend        | Streamlit                                |
| Backend         | Python                                   |
| Vector Database | FAISS                                    |
| Embedding Model | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM             | OpenAI API                               |
| Data Processing | Pandas, NumPy                            |
| Visualization   | Matplotlib                               |


---

# Example Output

Example AI assistant response:

User Query:
"What is Lean methodology?"

Assistant Response:

• Lean methodology focuses on eliminating waste in processes  
• It improves efficiency and reduces operational costs  
• Lean emphasizes continuous improvement and customer value  
• Organizations use Lean to optimize workflows and reduce delays  

Confidence: High (0.84 similarity)
Citations: lean_principles.txt

# Setup Instructions

1. Clone the repository:

git clone https://github.com/Keerthana2030/coe-digital-enablement

2. Install dependencies:

pip install -r requirements.txt

3. Add your OpenAI API key:

export OPENAI_API_KEY="your_key_here"

4. Run the Streamlit app:

streamlit run app.py

# Future Improvements

Possible enhancements include:

- Hybrid retrieval (BM25 + vector search)
- Larger knowledge base integration
- Multi-document ingestion pipeline
- Improved LLM reasoning
- Deployment on cloud infrastructure
- Role-based access to enterprise knowledge

# Author

**Keerthana Velukati**

B.Tech – Artificial Intelligence & Machine Learning  
Aspiring AI/ML Engineer

GitHub: https://github.com/Keerthana2030
LinkedIn: www.linkedin.com/in/keerthana-velukati-367710250
Email: keerthanavelukati@gmail.com

