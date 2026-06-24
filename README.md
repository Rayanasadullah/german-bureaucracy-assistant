# 🇩🇪 German Bureaucracy Assistant

An AI-powered chatbot that helps immigrants and newcomers navigate German bureaucracy. Ask questions in plain English and get simple, accurate answers — with official German government sources linked at the end of every response.

---

## What It Does

Moving to Germany involves a lot of paperwork and confusing processes. This assistant helps you understand:

- **Anmeldung** — How to register your address at the Bürgeramt
- **Residence Permits (Aufenthaltstitel)** — Types of permits, EU Blue Card, Opportunity Card, how to apply
- **Jobcenter & Unemployment** — Bürgergeld, ALG I, how to register as unemployed
- **Health Insurance (Krankenversicherung)** — Statutory vs. private insurance, what you need
- **Electronic Residence Title (eAT)** — What the card looks like, how to collect it, what to do if lost
- **Registration Certificate (Meldebescheinigung)** — What it is and how to get one

Every answer ends with a 📌 link to the **official German government website** so you can always verify the information.

---

## Features

- 💬 **Chat interface** — Ask questions in natural language
- 📄 **Upload documents** — Upload a government letter (PDF or image) and ask questions about it
- 🤖 **Three AI providers** — Choose between Claude (Anthropic), ChatGPT (OpenAI), or Ollama (free, runs locally)
- 🔍 **RAG pipeline** — Answers are grounded in official documents, not just AI guesswork
- 📌 **Official sources** — Every answer links to a relevant German government website
- 🔒 **Privacy-first** — API keys are never stored; Ollama option runs 100% on your own computer

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rayanasadullah/german-bureaucracy-assistant.git
cd german-bureaucracy-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key (optional)

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or enter your API key directly in the app sidebar — no `.env` file needed.

### 4. Run the app

```bash
python -m streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**

---

## Choosing an AI Provider

In the sidebar, you can choose between three options:

| Provider | Cost | Privacy | Quality |
|---|---|---|---|
| **Claude API** (Anthropic) | Pay-per-use | Key stays in your session | ⭐⭐⭐ Best |
| **OpenAI ChatGPT API** | Pay-per-use | Key stays in your session | ⭐⭐⭐ Great |
| **Ollama Llama3** | Free | 100% local, no internet | ⭐⭐ Good |

To get an API key:
- **Claude**: https://console.anthropic.com
- **OpenAI**: https://platform.openai.com/api-keys
- **Ollama**: Install from https://ollama.com then run `ollama pull llama3`

---

## Official Sources Used

This assistant references and links to the following official German government websites:

| Topic | Official Website |
|---|---|
| Immigration & residence permits | https://www.bamf.de |
| Moving to Germany (skilled workers) | https://www.make-it-in-germany.com |
| Jobcenter & Bürgergeld | https://www.jobcenter.digital |
| Employment Agency (ALG I) | https://www.arbeitsagentur.de |
| Health insurance | https://www.bundesgesundheitsministerium.de |
| Integration courses | https://www.bamf.de |
| Tax ID | https://www.bzst.bund.de |

---

## Project Structure

```
german-bureaucracy-assistant/
├── streamlit_app.py        # Main web app (Streamlit)
├── app.py                  # Terminal version
├── rag.py                  # RAG pipeline (ChromaDB)
├── requirements.txt
├── .env                    # Your API keys (not pushed to GitHub)
└── documents/              # Knowledge base (loaded into ChromaDB)
    ├── anmeldung.txt
    ├── residence_permit.txt
    ├── jobcenter.txt
    ├── health_insurance.txt
    ├── id_card_eat.txt
    ├── registration_certificate.txt
    ├── erste_schritte.txt
    └── BMG.pdf
```

---

## Important Disclaimer

> This assistant provides general information to help you understand German bureaucracy processes. It is **not legal advice**. Rules and requirements can change — always verify with the official government websites linked in each answer, or consult a qualified immigration lawyer for your specific situation.

---

## Built With

- [Streamlit](https://streamlit.io) — Web interface
- [ChromaDB](https://www.trychroma.com) — Vector database for RAG
- [Sentence Transformers](https://www.sbert.net) — Text embeddings
- [Anthropic Claude API](https://www.anthropic.com) — AI responses
- [OpenAI API](https://openai.com) — AI responses
- [Ollama](https://ollama.com) — Local AI option
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF reading
