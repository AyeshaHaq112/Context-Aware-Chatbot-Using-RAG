# Context-Aware Chatbot Using RAG (Retrieval-Augmented Generation)

## Objective
The objective of this project is to build a context-aware chatbot that can retrieve relevant information from a knowledge base and generate accurate answers. The chatbot maintains conversation history and uses retrieval techniques to enhance response quality.

---

## Dataset / Knowledge Base
A custom knowledge base was created consisting of text documents related to:
- Python programming
- Machine learning
- Natural language processing (NLP)
- Transformers and RAG
- Data science tools (Scikit-learn, Streamlit, etc.)

The knowledge base can be easily extended with additional documents such as:
- Wikipedia pages
- Internal company documents
- FAQs
- Lecture notes

---

## Methodology / Approach

### 1. Document Storage
All knowledge base documents were stored in a structured format using a Pandas DataFrame.

### 2. Vectorization
Text documents were converted into numerical vectors using:
- **TF-IDF Vectorizer**

This allows efficient similarity comparison between user queries and stored documents.

### 3. Retrieval Mechanism
Relevant documents were retrieved using:
- **Cosine Similarity**

The top 3 most relevant documents were selected for answering each query.

### 4. Question Answering
A lightweight transformer-based QA model was used:
```
distilbert-base-cased-distilled-squad
```

The retrieved documents were combined into a context, and the model generated answers based on this context.

### 5. Context Memory
The chatbot maintains conversation history:
- Stores previous user queries and responses
- Uses recent conversation context to improve follow-up questions

### 6. Retrieval-Augmented Generation (RAG)
The system follows a RAG approach:
- Retrieve relevant documents
- Generate answers using retrieved context

---

## Results / Key Findings
- The chatbot successfully retrieves relevant documents using TF-IDF and cosine similarity.
- Context-aware responses improve the quality of answers for follow-up questions.
- The system provides accurate answers for queries related to the knowledge base.
- The approach is lightweight and suitable for local execution without requiring heavy computation.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- Streamlit
---

## How to Run the Project

### 1. Install Required Libraries
```
pip install streamlit scikit-learn pandas numpy transformers torch
```

### 2. Run Jupyter Notebook
```
jupyter notebook chatbot_rag.ipynb
```

### 3. Run Streamlit App
```
streamlit run app.py
```

The chatbot will open in your browser.

---

## Features
- Context-aware conversation handling
- Document retrieval using vector similarity
- Lightweight QA model for answer generation
- Top relevant documents displayed for transparency
- Interactive UI using Streamlit

---

## Skills Demonstrated
- Retrieval-Augmented Generation (RAG)
- Information Retrieval (TF-IDF + Cosine Similarity)
- Conversational AI Development
- Context Memory Implementation
- Transformer-based Question Answering
- Deployment with Streamlit

---

## Conclusion
This project demonstrates how to build a context-aware chatbot using a Retrieval-Augmented Generation approach. By combining document retrieval and transformer-based question answering, the chatbot provides accurate and context-aware responses. The system is lightweight, efficient, and suitable for real-world applications such as customer support, knowledge assistants, and internal documentation systems.

---
