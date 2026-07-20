import streamlit as st
import anthropic
import fitz
import os
import ollama
import base64
import html as html_mod
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="German Bureaucracy Assistant",
    page_icon="🇩🇪",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
#  FULL DARK THEME  —  German flag accent (black / red / gold)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg:       #080c14;
    --bg2:      #0f1724;
    --bg3:      #182030;
    --gold:     #F5C518;
    --gold-dim: rgba(245,197,24,0.12);
    --red:      #C0392B;
    --border:   #1e2d42;
    --text:     #dce6f0;
    --muted:    #57728a;
    --user-bg:  #0e2044;
    --user-bd:  rgba(56,139,253,0.3);
    --bot-bg:   #0f1724;
}

/* ── Fonts ── */
*, *::before, *::after { box-sizing: border-box; }
body, .stApp * { font-family: 'Inter', sans-serif !important; font-size: 0.9rem; }
h1, h2, h3,
.sb-title, .sb-section, .header-badge,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label span p,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label div p
    { font-family: 'Space Grotesk', sans-serif !important; }

/* ── App shell ── */
.stApp { background-color: var(--bg) !important; }
.main .block-container {
    background-color: var(--bg) !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 860px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ════════════════════════════════════
   SIDEBAR
   ════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #05090f !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
    background: #05090f !important;
    padding-top: 0 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown p { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown strong { color: #fff !important; }

/* Sidebar inputs */
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="password"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(245,197,24,0.15) !important;
    outline: none !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] .stRadio > label {
    color: var(--muted) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    margin-bottom: 5px !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    border-color: var(--gold) !important;
    background: var(--gold-dim) !important;
}
/* Radio label text size */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label span {
    font-size: 0.82rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    line-height: 1.3 !important;
}

/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 0.8rem 0 !important;
}

/* Sidebar alerts */
section[data-testid="stSidebar"] .stAlert { border-radius: 8px !important; }

/* ════════════════════════════════════
   CHAT INPUT
   ════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,0.12) !important;
}
[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: var(--gold) !important; }

/* Make default stChatMessage transparent */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
}
[data-testid="stChatMessage"] > div { background: transparent !important; }

/* ════════════════════════════════════
   ALERTS (main area)
   ════════════════════════════════════ */
.stAlert { border-radius: 10px !important; }

/* ════════════════════════════════════
   FILE UPLOADER
   ════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold) !important; }
[data-testid="stFileUploader"] * { color: var(--text) !important; }
/* Hide the verbose drag-and-drop instruction text, keep only the button */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display: none !important; }
[data-testid="stFileUploader"] button {
    background: var(--bg3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding: 4px 12px !important;
    width: 100% !important;
    text-align: center !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: var(--gold) !important;
    background: var(--gold-dim) !important;
}

/* ════════════════════════════════════
   SCROLLBAR
   ════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ════════════════════════════════════
   CUSTOM HEADER
   ════════════════════════════════════ */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.1rem 0 0.9rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
}
.flag-bar {
    width: 5px;
    height: 46px;
    border-radius: 3px;
    flex-shrink: 0;
    background: linear-gradient(to bottom, #000 33.3%, var(--red) 33.3%, var(--red) 66.6%, var(--gold) 66.6%);
}
.header-text h1 {
    font-size: 1.35rem !important;
    color: var(--text) !important;
    margin: 0 !important;
    padding: 0 !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}
.header-text p {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    margin: 2px 0 0 !important;
    padding: 0 !important;
}
.header-badge {
    margin-left: auto;
    background: rgba(245,197,24,0.1);
    border: 1px solid rgba(245,197,24,0.25);
    color: var(--gold) !important;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

/* ════════════════════════════════════
   CUSTOM CHAT BUBBLES
   ════════════════════════════════════ */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin: 0.55rem 0;
}
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

.msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.msg-avatar.user {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
    color: #fff;
}
.msg-avatar.bot {
    background: linear-gradient(135deg, #0d0d0d 0%, #7a0000 100%);
    border: 2px solid var(--gold);
    font-size: 1rem;
}

.msg-bubble {
    max-width: 80%;
    padding: 10px 15px;
    font-size: 0.915rem;
    line-height: 1.72;
    word-wrap: break-word;
}
.msg-bubble.user {
    background: var(--user-bg);
    border: 1px solid var(--user-bd);
    border-radius: 18px 18px 4px 18px;
    color: var(--text);
}
.msg-bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 4px 18px 18px 18px;
    color: var(--text);
}
.msg-bubble.bot a { color: var(--gold) !important; text-decoration: none; }
.msg-bubble.bot a:hover { text-decoration: underline; }
.msg-bubble.bot strong { color: #ffffff; }
.msg-bubble.bot em { color: #a0b4c8; font-style: italic; }
.msg-bubble.bot code {
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
    font-family: 'Courier New', monospace !important;
}
.msg-bubble.bot ul { padding-left: 1.3em; margin: 0.3em 0; }
.msg-bubble.bot li { margin: 0.15em 0; }

/* ════════════════════════════════════
   WELCOME SCREEN
   ════════════════════════════════════ */
.welcome {
    text-align: center;
    padding: 2.5rem 0.5rem 1.5rem;
}
.welcome-flag {
    display: flex;
    justify-content: center;
    height: 5px;
    width: 54px;
    margin: 0 auto 1.4rem;
    border-radius: 3px;
    overflow: hidden;
}
.wf-b { background:#000; flex:1; }
.wf-r { background:var(--red); flex:1; }
.wf-g { background:var(--gold); flex:1; }
.welcome h2 {
    font-size: 1.55rem !important;
    color: var(--text) !important;
    margin: 0 0 0.4rem !important;
    font-weight: 700 !important;
}
.welcome .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }
.eq-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 520px;
    margin: 0 auto;
}
.eq-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    color: var(--muted);
    text-align: left;
}
.eq-card .eq-icon { font-size: 1rem; margin-bottom: 3px; display: block; }

/* ════════════════════════════════════
   SIDEBAR LOGO BLOCK
   ════════════════════════════════════ */
.sb-logo {
    padding: 1.3rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.3rem;
}
.sb-flag {
    display: flex;
    height: 3px;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.8rem;
    width: 42px;
}
.sb-flag .b { background: #000; flex:1; }
.sb-flag .r { background: var(--red); flex:1; }
.sb-flag .g { background: var(--gold); flex:1; }
.sb-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sb-sub { font-size: 0.7rem !important; color: var(--muted) !important; margin-top: 2px; }
.sb-section {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.75rem 0 0.25rem;
    display: block;
}

/* ════════════════════════════════════
   FILE CHIP (attached indicator)
   ════════════════════════════════════ */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,197,24,0.08);
    border: 1px solid rgba(245,197,24,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--gold);
    margin-bottom: 0.3rem;
}
/* Make the remove button tiny */
section.main [data-testid="stButton"]:has(button[title="Remove attachment"]) button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    padding: 2px 7px !important;
    font-size: 0.72rem !important;
    border-radius: 50% !important;
    min-height: 0 !important;
    line-height: 1 !important;
}
section.main [data-testid="stButton"]:has(button[title="Remove attachment"]) button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
}

/* ════════════════════════════════════
   CHAT INPUT — native attach ("+") button inside the bar
   ════════════════════════════════════ */
[data-testid="stChatInput"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 50% !important;
}
[data-testid="stChatInput"] button:hover {
    background: rgba(245,197,24,0.1) !important;
}
[data-testid="stChatInput"] svg { fill: var(--muted) !important; }
[data-testid="stChatInput"] button:hover svg { fill: var(--gold) !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--gold) !important; }

/* ════════════════════════════════════
   NEW CHAT BUTTON
   ════════════════════════════════════ */
section[data-testid="stSidebar"] [data-testid="stButton"]:has(button[kind="secondary"]) button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 8px 0 !important;
    transition: border-color 0.15s, background 0.15s !important;
}
section[data-testid="stSidebar"] [data-testid="stButton"]:has(button[kind="secondary"]) button:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--gold) !important;
    background: var(--gold-dim) !important;
    color: var(--gold) !important;
}

/* ════════════════════════════════════
   THINKING ANIMATION
   ════════════════════════════════════ */
.thinking-bubble {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin: 0.55rem 0;
}
.thinking-dots {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    display: flex;
    gap: 5px;
    align-items: center;
}
.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--gold);
    opacity: 0.4;
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-6px); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# MARKDOWN → HTML  (for bot message bubbles)
# ═══════════════════════════════════════════
def md_to_html(text: str) -> str:
    """Convert a subset of Markdown to HTML for safe display in chat bubbles."""
    t = html_mod.escape(text)
    # Links: [label](url)
    t = re.sub(
        r'\[([^\]]+)\]\((https?://[^)]+)\)',
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        t
    )
    # Bold: **text**
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    # Italic: *text* (not touching **)
    t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', t)
    # Bullet lists: "- item" or "• item"
    lines = t.split('\n')
    in_list = False
    out = []
    for line in lines:
        if line.startswith('- ') or line.startswith('&#x2022; '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            out.append(f'<li>{line[2:].strip()}</li>')
        else:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(line)
    if in_list:
        out.append('</ul>')
    t = '\n'.join(out)
    # Line breaks
    t = re.sub(r'\n(?!<)', '<br>', t)
    return t


def render_user_msg(content: str):
    safe = html_mod.escape(content).replace('\n', '<br>')
    st.markdown(f"""
    <div class="msg-row user">
        <div class="msg-avatar user">YOU</div>
        <div class="msg-bubble user">{safe}</div>
    </div>
    """, unsafe_allow_html=True)


def render_bot_msg(content: str):
    st.markdown(f"""
    <div class="msg-row bot">
        <div class="msg-avatar bot">🇩🇪</div>
        <div class="msg-bubble bot">{md_to_html(content)}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
st.sidebar.markdown("""
<div class="sb-logo">
    <div class="sb-flag"><div class="b"></div><div class="r"></div><div class="g"></div></div>
    <div class="sb-title">Bureaucracy Assistant</div>
    <div class="sb-sub">For immigrants in Germany</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<span class="sb-section">AI Provider</span>', unsafe_allow_html=True)
provider = st.sidebar.radio(
    "Choose AI Provider",
    ["Claude API (Recommended)", "OpenAI ChatGPT API", "Ollama Llama3 (Free, Local)"],
    label_visibility="collapsed"
)

api_key = None
openai_api_key = None
openai_model = "gpt-4o-mini"

st.sidebar.markdown("---")

if provider == "Claude API (Recommended)":
    st.sidebar.markdown('<span class="sb-section">Anthropic API Key</span>', unsafe_allow_html=True)
    api_key = st.sidebar.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed"
    )
    if api_key:
        st.sidebar.success("API key set")
    else:
        st.sidebar.warning("Enter your API key to start")

elif provider == "OpenAI ChatGPT API":
    st.sidebar.markdown('<span class="sb-section">OpenAI API Key</span>', unsafe_allow_html=True)
    openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed"
    )
    st.sidebar.markdown('<span class="sb-section">Model</span>', unsafe_allow_html=True)
    openai_model = st.sidebar.selectbox(
        "ChatGPT model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        label_visibility="collapsed"
    )
    if openai_api_key:
        st.sidebar.success("API key set")
    else:
        st.sidebar.warning("Enter your OpenAI API key to start")

else:
    st.sidebar.success("Ollama — running locally")

st.sidebar.markdown("---")

# ── New Chat button ──
if st.sidebar.button("✦  New Chat", use_container_width=True, key="new_chat"):
    st.session_state.messages = []
    for k in ["doc_name", "doc_text", "doc_image", "doc_image_type"]:
        st.session_state.pop(k, None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding:0.4rem 0;font-size:0.75rem;color:#57728a;line-height:1.6;">
🔒 <strong style="color:#dce6f0;">Privacy first.</strong><br>
API keys are never stored — they exist only in your browser session and are deleted when you close the tab.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SHARED SYSTEM PROMPT
# ═══════════════════════════════════════════
SYSTEM_TEMPLATE = """You are a friendly, knowledgeable assistant helping immigrants navigate German bureaucracy.

RULES:
1. Answer SHORT and simple — maximum 3-4 lines.
2. Only answer from the context below or the uploaded letter.
3. Be friendly and use simple language.
4. If the answer is not in the context, say: "I could not find this information."
5. GREETINGS & SMALL TALK EXCEPTION: If the user says hello, hi, hey, good morning, bye, goodbye, thanks, thank you, ok, great, or any casual social phrase that is NOT asking about a bureaucratic process — respond warmly and naturally WITHOUT adding any official source link at all.
6. For all other questions, ALWAYS end your answer with a relevant official German government link in this exact format:
   "📌 Official source: [title] → [URL]"
   Pick the most relevant link:
   - Registration (Anmeldung), city registration, Bürgeramt: https://www.bamf.de/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/Migrathek/ErsteDreiMonate/erste-drei-monate-node.html
   - Residence permit, visa, Aufenthaltstitel, Ausländerbehörde, eAT, Blue Card, Opportunity Card: https://www.bamf.de/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/zuwandererdrittstaaten-node.html
   - Immigration to Germany, moving to Germany, first steps, skilled workers: https://www.make-it-in-germany.com/en/
   - Jobcenter, Bürgergeld, unemployment benefits, ALG I, social welfare: https://www.jobcenter.digital
   - Employment Agency, job search, ALG I, Arbeitsagentur: https://www.arbeitsagentur.de/en/
   - Health insurance, Krankenversicherung, GKV: https://www.bundesgesundheitsministerium.de/en/health-insurance.html
   - Integration course, German language course: https://www.bamf.de/EN/Themen/Integration/ZugewanderteTeilnehmende/Integrationskurse/integrationskurse-node.html
   - Tax ID, Steuer, Finanzamt: https://www.bzst.bund.de/EN/Privatpersonen/SteuerlicheIdentifikationsnummer/steuerlicheidentifikationsnummer_node.html
   - General official Germany information portal: https://www.germany.info

RELEVANT CONTEXT FROM OFFICIAL DOCUMENTS:
{context}

USER'S UPLOADED LETTER:
{letter}"""


# ═══════════════════════════════════════════
# RAG BACKEND
# ═══════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")


@st.cache_resource
def load_rag():
    """Build a TF-IDF index over all documents. No ML model needed — loads instantly."""
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        filepath = os.path.join(DOCS_DIR, filename)
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
        # Slightly larger chunks for better context per result
        for i in range(0, len(text), 800):
            chunk = text[i:i + 800].strip()
            if chunk:
                chunks.append(chunk)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    matrix = vectorizer.fit_transform(chunks)
    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def search(rag_data, query, n_results=3):
    query_vec = rag_data["vectorizer"].transform([query])
    scores = cosine_similarity(query_vec, rag_data["matrix"]).flatten()
    top_indices = scores.argsort()[-n_results:][::-1]
    return [rag_data["chunks"][i] for i in top_indices]


def get_response_claude(messages, context, key, uploaded_text="", image_data=None, image_type=None):
    client = anthropic.Anthropic(api_key=key)
    system = SYSTEM_TEMPLATE.format(
        context=context,
        letter=uploaded_text if uploaded_text else "No letter uploaded"
    )
    api_messages = list(messages[:-1])
    last = messages[-1]
    if image_data and image_type:
        api_messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": image_type, "data": image_data}},
                {"type": "text", "text": last["content"]}
            ]
        })
    else:
        api_messages.append(last)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=api_messages
    )
    return response.content[0].text


def get_response_openai(messages, context, key, model="gpt-4o-mini", uploaded_text=""):
    client = OpenAI(api_key=key)
    system = SYSTEM_TEMPLATE.format(
        context=context,
        letter=uploaded_text if uploaded_text else "No letter uploaded"
    )
    openai_messages = [{"role": "system", "content": system}] + list(messages)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=openai_messages
    )
    return response.choices[0].message.content


def get_response_ollama(messages, context, uploaded_text=""):
    system = SYSTEM_TEMPLATE.format(
        context=context,
        letter=uploaded_text if uploaded_text else "No letter uploaded"
    )
    ollama_messages = [{"role": "system", "content": system}] + list(messages)
    response = ollama.chat(model="llama3", messages=ollama_messages)
    return response['message']['content']


# ═══════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════
rag_data = load_rag()

# ── Header ──
st.markdown("""
<div class="app-header">
    <div class="flag-bar"></div>
    <div class="header-text">
        <h1>German Bureaucracy Assistant</h1>
        <p>Ask anything about living &amp; working in Germany</p>
    </div>
    <div class="header-badge">RAG · Official Sources</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history or welcome screen ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-flag">
            <div class="wf-b"></div><div class="wf-r"></div><div class="wf-g"></div>
        </div>
        <h2>Willkommen! How can I help?</h2>
        <p class="sub">Ask me about registration, permits, health insurance, Jobcenter, or any German official process.</p>
        <div class="eq-grid">
            <div class="eq-card"><span class="eq-icon">📍</span>How do I register my address (Anmeldung)?</div>
            <div class="eq-card"><span class="eq-icon">🪪</span>What types of residence permit are there?</div>
            <div class="eq-card"><span class="eq-icon">💊</span>How does health insurance work in Germany?</div>
            <div class="eq-card"><span class="eq-icon">💼</span>What is Bürgergeld and who can get it?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            render_user_msg(message["content"])
        else:
            render_bot_msg(message["content"])

# ── File attached indicator (shows above input when a doc is loaded) ──
if st.session_state.get("doc_name"):
    chip_col, clear_col = st.columns([0.93, 0.07])
    with chip_col:
        st.markdown(
            f'<div class="file-chip">📎 {st.session_state["doc_name"]}</div>',
            unsafe_allow_html=True
        )
    with clear_col:
        if st.button("✕", key="clear_doc", help="Remove attachment"):
            for k in ["doc_name", "doc_text", "doc_image", "doc_image_type"]:
                st.session_state.pop(k, None)
            st.rerun()

# ── Chat input (native attach button inside the bar) ──
submission = st.chat_input(
    "Ask about German bureaucracy…",
    accept_file=True,
    file_type=["pdf", "png", "jpg", "jpeg"],
)

if submission:
    prompt = submission.text if hasattr(submission, "text") else str(submission)

    # Process any file attached via the chat input's built-in attach button
    files = getattr(submission, "files", None) or []
    for new_file in files:
        if new_file.type == "application/pdf":
            doc_fitz = fitz.open(stream=new_file.read(), filetype="pdf")
            extracted = ""
            for page in doc_fitz:
                extracted += page.get_text()
            st.session_state["doc_text"] = extracted
            st.session_state["doc_image"] = None
            st.session_state["doc_image_type"] = None
        else:
            new_file.seek(0)
            st.session_state["doc_image"] = base64.b64encode(new_file.read()).decode()
            st.session_state["doc_image_type"] = new_file.type
            st.session_state["doc_text"] = ""
        st.session_state["doc_name"] = new_file.name

if submission and (submission.text if hasattr(submission, "text") else submission):

    if provider == "Claude API (Recommended)" and not api_key:
        st.error("Please enter your Anthropic API key in the sidebar first.")
    elif provider == "OpenAI ChatGPT API" and not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar first.")
    else:
        # Show user message
        render_user_msg(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve RAG context
        relevant_chunks = search(rag_data, prompt)
        context = "\n\n".join(relevant_chunks)

        # Read persisted file data from session_state
        uploaded_text = st.session_state.get("doc_text", "")
        image_data = st.session_state.get("doc_image", None)
        image_type = st.session_state.get("doc_image_type", None)

        # Thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-bubble">
            <div class="msg-avatar bot">🇩🇪</div>
            <div class="thinking-dots">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Get AI response
        if provider == "Claude API (Recommended)":
            response = get_response_claude(
                st.session_state.messages, context, api_key,
                uploaded_text, image_data, image_type
            )
        elif provider == "OpenAI ChatGPT API":
            response = get_response_openai(
                st.session_state.messages, context, openai_api_key,
                openai_model, uploaded_text
            )
        else:
            response = get_response_ollama(
                st.session_state.messages, context, uploaded_text
            )

        # Replace animation with response
        thinking_placeholder.empty()
        render_bot_msg(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
