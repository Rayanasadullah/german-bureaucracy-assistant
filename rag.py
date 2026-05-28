import chromadb
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Create ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

def add_document(filename, text):
    # Split into chunks of 500 characters
    chunks = []
    for i in range(0, len(text), 500):
        chunks.append(text[i:i+500])


    # Add each chunk to ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{filename}_{i}"]

        )
    print(f"Added {len(chunks)} chunks from {filename}")

def search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

# Test it
with open("documents/anmeldung.txt", "r", encoding="utf-8") as f:
    text = f.read()


add_document("anmeldung", text)

results = search("What documents do I need?")
for r in results:
    print("---")
    print(r)