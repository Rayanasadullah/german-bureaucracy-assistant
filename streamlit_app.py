import streamlit as st
import anthropic
import chromadb
import fitz
import os
from dotenv import load_dotenv

load_dotenv()

# Page setup
st.set_page_config(
    page_title="German Bureaucracy Assistant",
    page_icon="🇩🇪"
)

st.title("🇩🇪 German Bureaucracy Assistant")
st.caption("Helping immigrants navigate German official documents")

@st.cache_resource
def load_rag():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="documents")
    
    for filename in os.listdir("documents"):
        filepath = f"documents/{filename}"
        text = ""
        
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        
        elif filename.endswith(".pdf"):
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text()
            doc.close()
        
        else:
            continue
        
        chunks = []
        for i in range(0, len(text), 500):
            chunks.append(text[i:i+500])
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{i}"]
            )
    
    return collection

def search(collection, query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

collection = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about German bureaucracy..."):
    
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    relevant_chunks = search(collection, prompt)
    context = "\n\n".join(relevant_chunks)
    
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"""You are a friendly assistant helping immigrants in Germany.

RULES:
1. Answer SHORT and simple - maximum 3 lines
2. Only answer from the context below
3. End every answer with: Would you like more details?
4. If answer not in context say: I could not find this information
5. Be friendly and simple
6. Always explain in simple language even if source is complex legal text

RELEVANT CONTEXT:
{context}""",
        messages=st.session_state.messages
    )
    
    response = message.content[0].text
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })