import streamlit as st
import anthropic
import chromadb
import fitz
import os
import ollama
import base64
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="German Bureaucracy Assistant",
    page_icon="🇩🇪"
)

st.title("🇩🇪 German Bureaucracy Assistant")
st.caption("Helping immigrants navigate German official documents")

# File upload
uploaded_file = st.file_uploader(
    "📎 Upload a letter or document (optional)",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Upload a government letter you received"
)

uploaded_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            uploaded_text += page.get_text()
        st.success("✅ PDF read successfully!")
    else:
        st.image(uploaded_file)
        st.info("📸 Image uploaded! Ask your question below.")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
st.sidebar.subheader("📎 Upload your letter")
uploaded_file = st.sidebar.file_uploader(
    "Upload a government letter",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Photo or PDF of any letter you received"
)
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

def get_response_claude(messages, context, key, uploaded_text="", image_data=None, image_type=None):
    client = anthropic.Anthropic(api_key=key)
    
    system = f"""You are a friendly assistant helping immigrants in Germany.
RULES:
1. Answer SHORT and simple - maximum 3 lines
2. Only answer from the context below or the uploaded letter
3. End every answer with: Would you like more details?
4. If answer not in context say: I could not find this information
5. Be friendly and simple
6. Always explain in simple language

RELEVANT CONTEXT FROM OFFICIAL DOCUMENTS:
{context}

USER'S UPLOADED LETTER:
{uploaded_text if uploaded_text else "No letter uploaded"}"""

    api_messages = []
    for msg in messages[:-1]:
        api_messages.append(msg)
    
    last_message = messages[-1]
    if image_data and image_type:
        api_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": last_message["content"]
                }
            ]
        })
    else:
        api_messages.append(last_message)
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=api_messages
    )
    return response.content[0].text

def get_response_ollama(messages, context, uploaded_text=""):
    system = f"""You are a friendly assistant helping immigrants in Germany.
RULES:
1. Answer SHORT and simple - maximum 3 lines
2. Only answer from the context below
3. End every answer with: Would you like more details?
4. If answer not in context say: I could not find this information
5. Be friendly and simple

RELEVANT CONTEXT:
{context}

USER'S UPLOADED LETTER:
{uploaded_text if uploaded_text else "No letter uploaded"}"""
    
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
        
        image_data = None
        image_type = None
        
        if uploaded_file is not None and uploaded_file.type != "application/pdf":
            uploaded_file.seek(0)
            image_data = base64.b64encode(uploaded_file.read()).decode()
            image_type = uploaded_file.type
        
        if provider == "Claude API (Better quality)":
            response = get_response_claude(
                st.session_state.messages,
                context,
                api_key,
                uploaded_text,
                image_data,
                image_type
            )
        else:
            response = get_response_ollama(
                st.session_state.messages,
                context,
                uploaded_text
            )
        
        with st.chat_message("assistant"):
            st.write(response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })