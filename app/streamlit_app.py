"""
app/streamlit_app.py  —  Music Mind UI
────────────────────────────────────────
"Shape your mind. One sound at a time."

Run:
    streamlit run app/streamlit_app.py

Ollama (optional local LLM):
    ollama serve && ollama pull tinyllama
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from app.models.model_loader import load_all
from app.services.hybrid        import HybridRecommenderSystem
from app.services.mood_engine   import resolve_mood
from app.services.explainability import (
    build_explanations, ollama_ready, send_to_ollama, WELLNESS_COPY,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Music Mind",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #07070f !important;
    color: #e4dff2 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #07070f !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
section[data-testid="stSidebar"] { display: none; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #2a1f50; border-radius: 2px; }

.mm-hero {
    text-align: center;
    padding: 64px 24px 44px;
    background:
        radial-gradient(ellipse 90% 55% at 50% -5%, rgba(80,50,170,0.32) 0%, transparent 65%),
        radial-gradient(ellipse 40% 30% at 15% 100%, rgba(25,12,70,0.5) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 85% 100%, rgba(12,25,85,0.4) 0%, transparent 70%);
}
.mm-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(44px, 8vw, 80px);
    font-weight: 300;
    letter-spacing: 0.2em;
    color: #ede8ff;
    text-transform: uppercase;
    margin: 0 0 8px;
    line-height: 1;
}
.mm-wordmark b { color: #9d7cf4; font-weight: 500; }
.mm-tagline {
    font-size: clamp(11px, 1.4vw, 14px);
    font-weight: 300;
    letter-spacing: 0.3em;
    color: #6d6090;
    text-transform: uppercase;
}
.mm-rule { width:40px; height:1px; margin:24px auto 0; background: linear-gradient(90deg, transparent, #6040c0, transparent); }
.mm-wrap { padding: 28px clamp(16px, 5vw, 72px); }
.mm-label { font-size:10px; font-weight:500; letter-spacing:.3em; text-transform:uppercase; color:#4e4470; margin-bottom:18px; }

.mm-card { border-radius:14px; padding:14px 10px 12px; text-align:center;
    border:1px solid rgba(255,255,255,0.06); background:rgba(255,255,255,0.025);
    cursor:pointer; transition: all .22s ease; }
.mm-card.on { border-color:rgba(157,124,244,0.55); background:rgba(100,65,200,0.2);
    box-shadow: 0 0 18px rgba(100,65,200,0.22), inset 0 1px 0 rgba(255,255,255,0.08); }
.mm-card-ico  { font-size:24px; display:block; margin-bottom:5px; }
.mm-card-name { font-size:10px; font-weight:500; letter-spacing:.1em; text-transform:uppercase; color:#b8aedc; }
.mm-card-sub  { font-size:10px; color:#5e5480; margin-top:2px; }

.stButton > button {
    position:relative; background:transparent !important; border:none !important;
    color:transparent !important; font-size:1px !important; padding:0 !important;
    margin-top:-72px !important; height:72px !important; width:100% !important;
    cursor:pointer !important; box-shadow:none !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg,#5530b4 0%,#3518888a 100%) !important;
    color:#ede8ff !important; border:none !important; border-radius:12px !important;
    font-size:13px !important; font-weight:500 !important; letter-spacing:.2em !important;
    text-transform:uppercase !important; padding:18px 32px !important;
    height:auto !important; width:100% !important; margin-top:0 !important;
    box-shadow:0 4px 24px rgba(85,48,180,0.42) !important; transition:all .28s ease !important;
}
button[kind="primary"]:hover { transform:translateY(-1px) !important; background:linear-gradient(135deg,#6640cc 0%,#4020a0 100%) !important; }
button[kind="primary"]:disabled { background:rgba(255,255,255,0.04) !important; color:#2e2848 !important; }

.stTextInput > div > div > input {
    background:rgba(255,255,255,0.035) !important; border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:10px !important; color:#e4dff2 !important; font-size:14px !important; padding:12px 16px !important;
}
.stTextInput > div > div > input:focus { border-color:rgba(157,124,244,0.45) !important; }
.stTextInput > div > div > input::placeholder { color:#3e3660 !important; }
label[data-testid="stWidgetLabel"] { font-size:10px !important; letter-spacing:.24em !important; text-transform:uppercase !important; color:#5e5480 !important; }

.mm-search-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:22px 24px; margin-bottom:20px; }
.mm-track { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:18px 20px 16px; margin-bottom:12px; position:relative; overflow:hidden; transition:all .25s ease; }
.mm-track::after { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(157,124,244,0.18),transparent); }
.mm-track:hover { background:rgba(100,65,200,0.07); border-color:rgba(157,124,244,0.18); transform:translateY(-1px); }
.mm-track-n { font-size:9px; letter-spacing:.22em; color:#3a3258; text-transform:uppercase; margin-bottom:6px; }
.mm-track-t { font-family:'Cormorant Garamond',serif; font-size:20px; font-weight:500; color:#ede8ff; line-height:1.15; }
.mm-track-a { font-size:12px; color:#6d6090; letter-spacing:.05em; margin-bottom:12px; margin-top:1px; }
.mm-tags    { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px; }
.mm-tag     { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:#9d7cf4; background:rgba(157,124,244,0.09); border:1px solid rgba(157,124,244,0.18); border-radius:20px; padding:2px 9px; }
.mm-explain { font-size:12px; color:#7a7098; line-height:1.65; border-left:2px solid rgba(100,65,200,0.25); padding-left:10px; margin-bottom:12px; font-style:italic; }
.mm-status  { background:rgba(100,65,200,0.1); border:1px solid rgba(100,65,200,0.22); border-radius:10px; padding:12px 18px; font-size:13px; color:#b8aedc; margin-bottom:20px; }
.mm-badge   { display:inline-flex; align-items:center; gap:6px; background:rgba(100,65,200,0.15); border:1px solid rgba(100,65,200,0.3); border-radius:20px; padding:4px 14px; font-size:11px; font-weight:500; letter-spacing:.12em; color:#b8aedc; text-transform:uppercase; margin-bottom:18px; }
.mm-global  { display:flex; flex-wrap:wrap; gap:7px; margin-top:6px; }
.mm-gtag    { font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#b8aedc; background:rgba(100,65,200,0.12); border:1px solid rgba(100,65,200,0.22); border-radius:20px; padding:4px 14px; }
audio { width:100% !important; height:26px !important; border-radius:6px !important; filter:invert(0.82) hue-rotate(245deg) !important; opacity:.65 !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Mood catalogue
# ─────────────────────────────────────────────────────────────────────────────
MOODS = [
    {"key":"focus",       "emoji":"🧠", "name":"Focus",    "sub":"Enter deep work"},
    {"key":"study",       "emoji":"🎓", "name":"Study",    "sub":"Clear, calm mind"},
    {"key":"gym",         "emoji":"🏋️", "name":"Power",    "sub":"Fuel the body"},
    {"key":"motivation",  "emoji":"🚀", "name":"Rise",     "sub":"Push through"},
    {"key":"chill",       "emoji":"😌", "name":"Drift",    "sub":"Let go"},
    {"key":"relax",       "emoji":"🌿", "name":"Breathe",  "sub":"Slow down"},
    {"key":"happy",       "emoji":"☀️", "name":"Joy",      "sub":"Lift the spirit"},
    {"key":"sad",         "emoji":"🌧️", "name":"Feel",     "sub":"Sit with it"},
    {"key":"heartbreak",  "emoji":"💔", "name":"Healing",  "sub":"Process & release"},
    {"key":"romantic",    "emoji":"❤️", "name":"Tender",   "sub":"Open the heart"},
    {"key":"late night",  "emoji":"🌃", "name":"Midnight", "sub":"Introspective"},
    {"key":"night drive", "emoji":"🌙", "name":"Wander",   "sub":"Moving through dark"},
    {"key":"morning",     "emoji":"🌅", "name":"Dawn",     "sub":"Begin again"},
    {"key":"meditation",  "emoji":"🧘", "name":"Still",    "sub":"Zero noise"},
    {"key":"nostalgic",   "emoji":"🕰️", "name":"Memory",   "sub":"Return somewhere"},
    {"key":"angry",       "emoji":"⚡", "name":"Release",  "sub":"Channel tension"},
    {"key":"road trip",   "emoji":"🚗", "name":"Freedom",  "sub":"Open road"},
    {"key":"coding",      "emoji":"⌨️", "name":"Flow",     "sub":"In the zone"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.spinner(""):
    try:
        data = load_all()
    except FileNotFoundError as e:
        st.error(f"Missing data: {e}")
        st.stop()

if "mood_sel" not in st.session_state:
    st.session_state.mood_sel = None
if "result" not in st.session_state:
    st.session_state.result = None

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mm-hero">
  <h1 class="mm-wordmark">Music<b>Mind</b></h1>
  <p class="mm-tagline">Shape your mind &nbsp;·&nbsp; One sound at a time</p>
  <div class="mm-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Mood grid ─────────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap">', unsafe_allow_html=True)
st.markdown('<p class="mm-label">01 — Choose your state</p>', unsafe_allow_html=True)
COLS = 6
for row_start in range(0, len(MOODS), COLS):
    chunk = MOODS[row_start:row_start + COLS]
    cols  = st.columns(len(chunk))
    for col, m in zip(cols, chunk):
        active = st.session_state.mood_sel == m["key"]
        with col:
            st.markdown(f"""
            <div class="mm-card {'on' if active else ''}">
              <span class="mm-card-ico">{m['emoji']}</span>
              <div class="mm-card-name">{m['name']}</div>
              <div class="mm-card-sub">{m['sub']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("​", key=f"m_{m['key']}", use_container_width=True):
                st.session_state.mood_sel = None if active else m["key"]
                st.session_state.result   = None
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── Search panel ──────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap" style="padding-top:4px">', unsafe_allow_html=True)
st.markdown('<p class="mm-label">02 — Or find a song / artist  <span style="color:#2e2848">(optional)</span></p>', unsafe_allow_html=True)
st.markdown('<div class="mm-search-panel">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    song_name = st.text_input("Song name", placeholder="e.g.  Lose Yourself").strip()
with c2:
    artist_name = st.text_input("Artist", placeholder="e.g.  Eminem").strip()

typed_mood = st.text_input(
    "Describe how you feel right now",
    placeholder="e.g.  I need to focus … feeling heartbroken … about to hit the gym …",
).strip()
if typed_mood:
    resolved = resolve_mood(typed_mood)
    if resolved:
        st.session_state.mood_sel = resolved
        matched = next((m for m in MOODS if m["key"] == resolved), None)
        label   = f"{matched['emoji']} {matched['name']}" if matched else resolved
        st.markdown(f'<p style="font-size:11px;color:#6d6090;letter-spacing:.1em">Matched state → {label}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:11px;color:#5e3a3a;letter-spacing:.1em">No match — try another phrase or use the cards above.</p>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ── Settings ──────────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap" style="padding-top:0">', unsafe_allow_html=True)
with st.expander("⚙  Session settings", expanded=False):
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        k = st.selectbox("Tracks per session", [5, 10, 15, 20], index=1)
    with sc2:
        diversity = st.slider("Discovery breadth  (1=close · 9=wide)", 1, 9, 5)
        content_w = 1 - diversity / 10
    with sc3:
        use_ollama = st.checkbox("🤖  TinyLlama guides", value=False, help="Start Ollama first:  ollama serve")
        if use_ollama:
            ok = ollama_ready()
            st.markdown(f'<p style="font-size:11px;color:#6d6090">{"🟢 Ready" if ok else "🔴 Not running — run: ollama serve"}</p>', unsafe_allow_html=True)
        else:
            ok = False
st.markdown('</div>', unsafe_allow_html=True)

# ── CTA button ────────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap" style="padding-top:8px;padding-bottom:36px">', unsafe_allow_html=True)
has_input = bool(song_name or artist_name or st.session_state.mood_sel)

if st.button("🎧  Start Session", disabled=not has_input, type="primary", use_container_width=True):
    recommender = HybridRecommenderSystem(
        number_of_recommendations = k,
        weight_content_based      = content_w,
        mood_blend_weight         = 0.25,
    )
    with st.spinner("Curating your session …"):
        try:
            st.session_state.result = recommender.give_recommendations(
                songs_data          = data["filtered_data"],
                track_ids           = data["track_ids"],
                transformed_matrix  = data["transformed_matrix"],
                interaction_matrix  = data["interaction_matrix"],
                original_songs_data = data["songs_data"],
                song_name           = song_name   or None,
                artist_name         = artist_name or None,
                mood_input          = st.session_state.mood_sel,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result
if not result:
    st.stop()

mood = result.get("mood")
explanation_data = build_explanations(
    seed_song       = result["seed_song"],
    seed_artist     = result["seed_artist"],
    recommendations = result["recommendations"],
    tags_per_song   = result["tags_per_song"],
    songs_data      = data["songs_data"],
    mood            = mood,
)

st.markdown('<div class="mm-wrap" style="padding-top:0">', unsafe_allow_html=True)
clean_msg = result["status_message"].replace("✅ ", "").replace("🎭 ", "")
st.markdown(f'<div class="mm-status">🎧 &nbsp;{clean_msg}</div>', unsafe_allow_html=True)

if mood:
    m_meta = next((m for m in MOODS if m["key"] == mood), None)
    if m_meta:
        st.markdown(f'<div class="mm-badge">{m_meta["emoji"]}&nbsp; {m_meta["name"]}&nbsp;·&nbsp; {m_meta["sub"]}</div>', unsafe_allow_html=True)

recs = result["recommendations"]
st.markdown(f'<p class="mm-label" style="margin-bottom:16px">Your session — {len(recs)} tracks</p>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
for i, row in recs.iterrows():
    name    = str(row.get("name",   "Unknown")).title()
    artist  = str(row.get("artist", "Unknown")).title()
    preview = row.get("spotify_preview_url", None)
    key     = f"{name} by {artist}"
    tags    = result["tags_per_song"].get(key, [])

    if use_ollama and ok:
        with st.spinner(""):
            expl_data = explanation_data["explanations"].get(key, {})
            try:
                explanation = send_to_ollama(expl_data.get("ollama_prompt", ""))
            except Exception:
                explanation = expl_data.get("inline_reason", "")
    elif mood:
        explanation = WELLNESS_COPY.get(mood, "to match your current state.")
        explanation = f"This track was chosen {explanation}"
    else:
        explanation = ""

    tag_html     = "".join(f'<span class="mm-tag">{t}</span>' for t in tags)
    explain_html = f'<div class="mm-explain">{explanation}</div>' if explanation else ""

    target = col_a if i % 2 == 0 else col_b
    with target:
        st.markdown(f"""
        <div class="mm-track">
          <div class="mm-track-n">Track {i+1:02d}</div>
          <div class="mm-track-t">{name}</div>
          <div class="mm-track-a">{artist}</div>
          <div class="mm-tags">{tag_html}</div>
          {explain_html}
        </div>""", unsafe_allow_html=True)
        if pd.notna(preview) and preview:
            st.audio(str(preview))

st.markdown('</div>', unsafe_allow_html=True)

# ── Global themes ─────────────────────────────────────────────────────────────
if result.get("global_tags"):
    st.markdown('<div class="mm-wrap" style="padding-top:4px">', unsafe_allow_html=True)
    st.markdown('<p class="mm-label">Session themes</p>', unsafe_allow_html=True)
    html = "".join(f'<span class="mm-gtag">{t}</span>' for t in result["global_tags"])
    st.markdown(f'<div class="mm-global">{html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Feedback ──────────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap" style="padding-top:0;padding-bottom:48px">', unsafe_allow_html=True)
fb1, fb2, fb3, fb4, _ = st.columns([1,1,1,1,2])
with fb1:
    if st.button("😌  Calmer",     use_container_width=True): st.toast("Glad it helped you settle.", icon="🎧")
with fb2:
    if st.button("⚡  Energised",  use_container_width=True): st.toast("Charged up — go get it.",   icon="🔥")
with fb3:
    if st.button("🎯  Focused",    use_container_width=True): st.toast("In the zone. Keep going.",   icon="🧠")
with fb4:
    if st.button("↩  New session", use_container_width=True):
        st.session_state.result   = None
        st.session_state.mood_sel = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
