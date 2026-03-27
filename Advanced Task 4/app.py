import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -----------------------------
# Knowledge base
# -----------------------------
documents = [
    "Python is a high-level programming language used for web development, machine learning, automation, and data analysis.",
    "Machine learning is a branch of artificial intelligence that allows systems to learn patterns from data and make predictions.",
    "Streamlit is an open-source Python framework used to build interactive web apps for data science and machine learning projects.",
    "Natural language processing, or NLP, is a field of AI that focuses on the interaction between computers and human language.",
    "Retrieval-Augmented Generation, or RAG, combines information retrieval with language generation to answer questions using external knowledge.",
    "Scikit-learn is a Python machine learning library that provides tools for classification, regression, clustering, and preprocessing.",
    "Transformers are deep learning models that are widely used for NLP tasks such as text classification, translation, summarization, and question answering.",
    "A vector store is used to save numerical representations of documents so that similar content can be retrieved efficiently.",
    "Cosine similarity measures how similar two text vectors are and is commonly used in information retrieval systems.",
    "Gradio and Streamlit are popular tools for deploying machine learning applications with simple user interfaces."
]

df = pd.DataFrame({"document": documents})

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(df["document"])

# -----------------------------
# QA model
# -----------------------------
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve_documents(query, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    retrieved_docs = df.iloc[top_indices]["document"].tolist()
    return retrieved_docs

# -----------------------------
# Answer function
# -----------------------------
def answer_question(query, chat_history, top_k=3):
    recent_memory = " ".join(
        [f"User: {item['user']} Bot: {item['bot']}" for item in chat_history[-3:]]
    )
    
    enhanced_query = recent_memory + " " + query if recent_memory else query
    
    retrieved_docs = retrieve_documents(enhanced_query, top_k=top_k)
    context = " ".join(retrieved_docs)
    
    result = qa_pipeline(question=query, context=context)
    
    return {
        "answer": result["answer"],
        "confidence": result["score"],
        "retrieved_docs": retrieved_docs
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Context-Aware Chatbot Using RAG")
st.write("Ask questions from the knowledge base.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_query.strip():
        result = answer_question(user_query, st.session_state.chat_history)
        
        st.session_state.chat_history.append({
            "user": user_query,
            "bot": result["answer"]
        })
        
        st.subheader("Answer")
        st.write(result["answer"])
        
        st.subheader("Confidence Score")
        st.write(round(result["confidence"], 4))
        
        st.subheader("Retrieved Documents")
        for doc in result["retrieved_docs"]:
            st.write("- " + doc)

st.subheader("Conversation History")
for item in st.session_state.chat_history:
    st.write(f"**You:** {item['user']}")
    st.write(f"**Bot:** {item['bot']}")