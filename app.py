from dotenv import load_dotenv
load_dotenv()
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

def add_document(filename, text):
    chunks = []
    for i in range(0, len(text), 500):
        chunks.append(text[i:i+500])
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{filename}_{i}"]
        )
    print(f"✅ Loaded: {filename} ({len(chunks)} chunks)")

def search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

# Load ALL documents from documents folder
print("Loading documents...")
for filename in os.listdir("documents"):
    if filename.endswith(".txt"):
        with open(f"documents/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
        add_document(filename, text)

# Connect to Anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
conversation_history = []

print("\nGerman Bureaucracy Assistant ready!")
print("Type 'quit' to exit")
print("----------------------------\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    # RAG: find relevant chunks
    relevant_chunks = search(user_input)
    context = "\n\n".join(relevant_chunks)
    
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"""You are a friendly assistant helping immigrants in Germany.

RULES:
1. Answer SHORT and simple - maximum 3-4 lines
2. Only answer from the context below
3. Be friendly and simple
4. If answer not in context say: I could not find this information
5. ALWAYS end every answer with a relevant official German government link using this format:
   "📌 Official source: [title] → [URL]"
   Choose the most relevant link from this list based on the topic:
   - Registration (Anmeldung), city registration, Bürgeramt: https://www.bamf.de/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/Migrathek/ErsteDreiMonate/erste-drei-monate-node.html
   - Residence permit, visa, Aufenthaltstitel, Ausländerbehörde, eAT, Blue Card, Opportunity Card: https://www.bamf.de/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/zuwandererdrittstaaten-node.html
   - Immigration to Germany, moving to Germany, first steps, skilled workers: https://www.make-it-in-germany.com/en/
   - Jobcenter, Bürgergeld, unemployment benefits, ALG I, social welfare: https://www.jobcenter.digital
   - Employment Agency, job search, ALG I, Arbeitsagentur: https://www.arbeitsagentur.de/en/
   - Health insurance, Krankenversicherung, GKV: https://www.bundesgesundheitsministerium.de/en/health-insurance.html
   - Integration course, German language course: https://www.bamf.de/EN/Themen/Integration/ZugewanderteTeilnehmende/Integrationskurse/integrationskurse-node.html
   - Tax ID, Steuer, Finanzamt: https://www.bzst.bund.de/EN/Privatpersonen/SteuerlicheIdentifikationsnummer/steuerlicheidentifikationsnummer_node.html
   - General official Germany information portal: https://www.germany.info

RELEVANT CONTEXT:
{context}""",
        messages=conversation_history
    )
    
    response = message.content[0].text
    print(f"\nBot: {response}\n")
    
    conversation_history.append({
        "role": "assistant",
        "content": response
    })