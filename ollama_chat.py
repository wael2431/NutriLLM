import streamlit as st
import requests
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriLLM",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #FFFFFF; color: #E0C09C; }

  .chat-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #2a2a3a;
    margin-bottom: 1rem;
  }
  .chat-header h1 { font-size: 1.4rem; font-weight: 600; margin: 0; color: #F5E1C4; }

  .msg-user {
    background: #1e1e2e; border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem; margin: 0.5rem 0 0.5rem 15%;
    border: 1px solid #2a2a3a; font-size: 0.95rem;
  }
  .msg-assistant {
    background: #FFFFFF; border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1rem; margin: 0.5rem 15% 0.5rem 0;
    border: 1px solid #2a2a3a; font-size: 0.95rem; line-height: 1.6;
  }
  .msg-label {
    font-size: 0.68rem; color: #555570; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;
  }

  .stTextInput > div > div > input {
    background: #F5E1C4 !important; border: 1px solid #2a2a3a !important;
    color: #e8e8f0 !important; border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
  }
  .stButton > button {
    background: #3b3bf5; color: white; border: none;
    border-radius: 12px; padding: 0.5rem 1.5rem;
    font-family: 'Sora', sans-serif; font-weight: 600;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #5555ff; }

  section[data-testid="stSidebar"] { background: #E0C09C; border-right: 1px solid #2a2a3a; }
  section[data-testid="stSidebar"] * { color: ##E0C09C !important; }
  div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
MODEL = "nutrillm-v1:latest"


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_response(model: str, messages: list):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except Exception as e:
        yield f"⚠️ Error: {e}"


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_input = None
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <span style="font-size:1.6rem">🤖</span>
  <h1>NutriLLM</h1>
</div>
""", unsafe_allow_html=True)


# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
          <div class="msg-label">YOU</div>
          {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-assistant">
          <div class="msg-label">ASSISTANT</div>
          {msg["content"]}
        </div>""", unsafe_allow_html=True)


# ── Process pending input (runs before showing input box) ────────────────────
if st.session_state.pending_input:
    user_text = st.session_state.pending_input
    st.session_state.pending_input = None

    st.markdown(f"""
    <div class="msg-user">
      <div class="msg-label">YOU</div>
      {user_text}
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": user_text})

    full_reply = ""
    reply_placeholder = st.empty()

    for token in get_response(MODEL, st.session_state.messages):
        full_reply += token
        reply_placeholder.markdown(f"""
        <div class="msg-assistant">
          <div class="msg-label">ASSISTANT</div>
          {full_reply}▌
        </div>""", unsafe_allow_html=True)

    reply_placeholder.markdown(f"""
    <div class="msg-assistant">
      <div class="msg-label">ASSISTANT</div>
      {full_reply}
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    st.rerun()


# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message…",
        label_visibility="collapsed",
        key="user_input",
    )
with col2:
    send = st.button("Send", use_container_width=True)

if send and user_input.strip():
    st.session_state.pending_input = user_input.strip()
    st.rerun()