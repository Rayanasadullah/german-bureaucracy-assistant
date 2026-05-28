import streamlit as st
import anthropic
import chromadb
import fitz
import os
import ollama
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="German Bureaucracy Assistant",
    page_icon="🇩🇪"
)

st.title("🇩🇪 German Bureaucracy Assistant")
st.caption("Helping immigrants navigate German official documents")

# Sidebar
st.sidebar.title("⚙️ Settings")
provider = st.sidebar.radio(
    "Choose AI Provider:",
    ["Claude API (Better quality)", "Ollama Llama3 (Free, Local)"]
)

api_key = None

if provider == "Claude API (Better quality)":
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input(
        "Enter your Anthropic API key:",
        type="password",
        placeholder="sk-ant-..."
    )
    if api_key:
        st.sidebar.success("✅ API key entered")
    else:
        st.sidebar.warning("⚠️ Please enter your API key")
    
    # Security message
    st.sidebar.markdown("---")
    st.sidebar.info("""
🔒 **Your API key is secure:**
- Never stored on any server
- Only used in your browser session
- Deleted when you close the tab
- We never see or log your key
""")

else:
    st.sidebar.success("✅ Ollama - 100% Local")
    st.sidebar.info("""
🔒 **Maximum Privacy:**
- Runs entirely on your computer
- No internet connection needed
- No API costs
- Your data never leaves your device
""")

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

def get_response_claude(messages, context, key):
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"""You are a friendly assistant helping immigrants in Germany.
RULES:
1. Answer SHORT and simple - maximum 3 lines
2. Only answer from the context below
3. End every answer with: Would you like more details?
4. If answer not in context say: I could not find this information
5. Be friendly and simple
6. Always explain in simple language

RELEVANT CONTEXT:
{context}""",
        messages=messages
    )
    return response.content[0].text

def get_response_ollama(messages, context):
    system = f"""You are a friendly assistant helping immigrants in Germany.
RULES:
1. Answer SHORT and simple - maximum 3 lines
2. Only answer from the context below
3. End every answer with: Would you like more details?
4. If answer not in context say: I could not find this information
5. Be friendly and simple

RELEVANT CONTEXT:
{context}"""
    
    ollama_messages = [{"role": "system", "content": system}]
    for msg in messages:
        ollama_messages.append(msg)
    
    response = ollama.chat(
        model="llama3",
        messages=ollama_messages
    )
    return response['message']['content']

collection = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about German bureaucracy..."):
    
    if provider == "Claude API (Better quality)" and not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar first!")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        relevant_chunks = search(collection, prompt)
        context = "\n\n".join(relevant_chunks)
        
        if provider == "Claude API (Better quality)":
            response = get_response_claude(
                st.session_state.messages, 
                context, 
                api_key
            )
        else:
            response = get_response_ollama(
                st.session_state.messages, 
                context
            )
        
        with st.chat_message("assistant"):
            st.write(response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })