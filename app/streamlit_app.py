"""
app/streamlit_app.py  —  Music Mind UI
"Shape your mind. One sound at a time."

Run locally:   streamlit run app/streamlit_app.py
Streamlit Cloud main file path: app/streamlit_app.py
"""
import sys
from pathlib import Path

# ── CRITICAL: Fix import paths for Streamlit Cloud ───────────────────────────
# On Streamlit Cloud:  CWD = /mount/src/music-mind/
#                      __file__ = /mount/src/music-mind/app/streamlit_app.py
# We need REPO ROOT in sys.path so "from app.X import Y" works everywhere.
_APP_DIR  = Path(__file__).resolve().parent          # .../app/
_REPO_ROOT = _APP_DIR.parent                         # .../music-mind/

# Insert repo root FIRST so Python finds the 'app' package correctly
for _p in [str(_REPO_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np

# Now these imports work both locally and on Streamlit Cloud
from app.models.model_loader     import load_all
from app.services.hybrid         import HybridRecommenderSystem
from app.services.mood_engine    import resolve_mood
from app.services.explainability import (
    build_explanations, ollama_ready, send_to_ollama, WELLNESS_COPY,
)

st.set_page_config(
    page_title="Music Mind",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a18 !important;
    color: #d8d0f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: #0a0a18 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
section[data-testid="stSidebar"] { display: none; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #3a2a60; border-radius: 2px; }

.mm-hero {
    text-align: center;
    padding: 64px 24px 44px;
    background:
        radial-gradient(ellipse 90% 55% at 50% -5%, rgba(90,55,200,0.35) 0%, transparent 65%),
        radial-gradient(ellipse 40% 30% at 15% 100%, rgba(30,15,80,0.5) 0%, transparent 70%);
}
.mm-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(44px, 8vw, 80px);
    font-weight: 300;
    letter-spacing: 0.2em;
    color: #f0ebff;
    text-transform: uppercase;
    margin: 0 0 8px;
    line-height: 1;
}
.mm-wordmark b { color: #b090ff; font-weight: 500; }
.mm-tagline {
    font-size: clamp(11px, 1.4vw, 14px);
    font-weight: 300;
    letter-spacing: 0.3em;
    color: #8878b8;
    text-transform: uppercase;
}
.mm-rule { width:40px; height:1px; margin:24px auto 0;
    background: linear-gradient(90deg, transparent, #7050d0, transparent); }
.mm-wrap { padding: 28px clamp(16px, 5vw, 72px); }
.mm-label {
    font-size: 11px; font-weight: 500; letter-spacing: .28em;
    text-transform: uppercase; color: #9880cc; margin-bottom: 18px;
}

.mm-card {
    border-radius: 14px; padding: 14px 10px 12px; text-align: center;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
    cursor: pointer; transition: all .22s ease;
}
.mm-card.on {
    border-color: rgba(180,140,255,0.70);
    background: rgba(110,75,220,0.25);
    box-shadow: 0 0 18px rgba(110,75,220,0.28), inset 0 1px 0 rgba(255,255,255,0.10);
}
.mm-card-ico  { font-size: 24px; display: block; margin-bottom: 5px; }
.mm-card-name { font-size: 11px; font-weight: 500; letter-spacing: .1em;
                text-transform: uppercase; color: #d0c4f0; }
.mm-card-sub  { font-size: 11px; color: #8070a8; margin-top: 2px; }

.stButton > button {
    position: relative; background: transparent !important;
    border: none !important; color: transparent !important;
    font-size: 1px !important; padding: 0 !important;
    margin-top: -72px !important; height: 72px !important;
    width: 100% !important; cursor: pointer !important;
    box-shadow: none !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #6038c8 0%, #3a1898 100%) !important;
    color: #f0ebff !important; border: none !important;
    border-radius: 12px !important; font-size: 13px !important;
    font-weight: 500 !important; letter-spacing: .2em !important;
    text-transform: uppercase !important; padding: 18px 32px !important;
    height: auto !important; width: 100% !important; margin-top: 0 !important;
    box-shadow: 0 4px 24px rgba(95,55,200,0.48) !important;
    transition: all .28s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg, #7248d8 0%, #4a28b0 100%) !important;
}
button[kind="primary"]:disabled {
    background: rgba(255,255,255,0.06) !important; color: #4a3870 !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 10px !important; color: #f0ebff !important;
    font-size: 14px !important; padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(180,140,255,0.55) !important;
}
.stTextInput > div > div > input::placeholder { color: #5a4880 !important; }
label[data-testid="stWidgetLabel"] {
    font-size: 11px !important; letter-spacing: .22em !important;
    text-transform: uppercase !important; color: #a090cc !important;
    font-weight: 500 !important;
}
.mm-search-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 22px 24px; margin-bottom: 20px;
}
.mm-track {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 16px; padding: 20px 22px 18px;
    margin-bottom: 14px; position: relative;
    overflow: hidden; transition: all .25s ease;
}
.mm-track::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,140,255,0.22), transparent);
}
.mm-track:hover {
    background: rgba(110,75,220,0.10);
    border-color: rgba(180,140,255,0.28); transform: translateY(-1px);
}
.mm-track-n { font-size: 10px; letter-spacing: .22em; color: #7060a8;
              text-transform: uppercase; margin-bottom: 7px; }
.mm-track-t { font-family: 'Cormorant Garamond', serif; font-size: 22px;
              font-weight: 500; color: #f2edff; line-height: 1.15; margin-bottom: 2px; }
.mm-track-a { font-size: 13px; color: #a090c0; letter-spacing: .05em; margin-bottom: 14px; }
.mm-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.mm-tag  {
    font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
    color: #c8aaff; background: rgba(180,140,255,0.13);
    border: 1px solid rgba(180,140,255,0.28);
    border-radius: 20px; padding: 3px 11px; font-weight: 500;
}
.mm-explain {
    font-size: 13px; color: #c0b0e0; line-height: 1.78;
    border-left: 2px solid rgba(140,100,240,0.40);
    padding-left: 12px; margin-bottom: 14px;
}
.mm-status {
    background: rgba(110,75,220,0.14);
    border: 1px solid rgba(140,100,240,0.32);
    border-radius: 10px; padding: 13px 20px;
    font-size: 14px; color: #d8c8f8; margin-bottom: 22px;
}
.mm-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(110,75,220,0.18);
    border: 1px solid rgba(140,100,240,0.38);
    border-radius: 20px; padding: 5px 16px;
    font-size: 12px; font-weight: 500; letter-spacing: .12em;
    color: #d8c8f8; text-transform: uppercase; margin-bottom: 20px;
}
.mm-global { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.mm-gtag   {
    font-size: 12px; letter-spacing: .10em; text-transform: uppercase;
    color: #c8aaff; background: rgba(110,75,220,0.15);
    border: 1px solid rgba(140,100,240,0.30);
    border-radius: 20px; padding: 5px 16px; font-weight: 500;
}
audio {
    width: 100% !important; height: 28px !important;
    border-radius: 6px !important;
    filter: invert(0.85) hue-rotate(245deg) !important;
    opacity: 0.78 !important;
}
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important; font-size: 12px !important;
    letter-spacing: .18em !important; text-transform: uppercase !important;
    color: #a090cc !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] span { color: #f0ebff !important; }
[data-testid="stAlert"] {
    background: rgba(180,50,50,0.15) !important;
    border-color: rgba(220,80,80,0.35) !important;
    border-radius: 10px !important; color: #ffb8b8 !important;
}
.mm-feedback-label {
    text-align: center; font-size: 11px; letter-spacing: .26em;
    text-transform: uppercase; color: #5a4880; margin: 28px 0 14px;
}
"""

AUDIO_JS = """
<script>
(function() {
  function stopOthers(playing) {
    document.querySelectorAll('audio').forEach(function(a) {
      if (a !== playing) { a.pause(); a.currentTime = 0; }
    });
  }
  function attachListeners() {
    document.querySelectorAll('audio').forEach(function(a) {
      if (!a._mmBound) {
        a._mmBound = true;
        a.addEventListener('play', function() { stopOthers(a); });
      }
    });
  }
  attachListeners();
  new MutationObserver(attachListeners).observe(document.body, {childList:true, subtree:true});
})();
</script>
"""

MOODS = [
    {"key":"focus",       "emoji":"🧠", "name":"Focus",     "sub":"Enter deep work"},
    {"key":"study",       "emoji":"🎓", "name":"Study",     "sub":"Clear, calm mind"},
    {"key":"gym",         "emoji":"🏋️", "name":"Power",     "sub":"Fuel the body"},
    {"key":"motivation",  "emoji":"🚀", "name":"Rise",      "sub":"Push through"},
    {"key":"chill",       "emoji":"😌", "name":"Drift",     "sub":"Let go"},
    {"key":"relax",       "emoji":"🌿", "name":"Breathe",   "sub":"Slow down"},
    {"key":"happy",       "emoji":"☀️", "name":"Joy",       "sub":"Lift the spirit"},
    {"key":"sad",         "emoji":"🌧️", "name":"Feel",      "sub":"Sit with it"},
    {"key":"heartbreak",  "emoji":"💔", "name":"Healing",   "sub":"Process & release"},
    {"key":"romantic",    "emoji":"❤️", "name":"Tender",    "sub":"Open the heart"},
    {"key":"late night",  "emoji":"🌃", "name":"Midnight",  "sub":"Introspective"},
    {"key":"night drive", "emoji":"🌙", "name":"Wander",    "sub":"Moving through dark"},
    {"key":"morning",     "emoji":"🌅", "name":"Dawn",      "sub":"Begin again"},
    {"key":"meditation",  "emoji":"🧘", "name":"Still",     "sub":"Zero noise"},
    {"key":"nostalgic",   "emoji":"🕰️", "name":"Memory",    "sub":"Return somewhere"},
    {"key":"angry",       "emoji":"⚡", "name":"Release",   "sub":"Channel tension"},
    {"key":"road trip",   "emoji":"🚗", "name":"Freedom",   "sub":"Open road"},
    {"key":"coding",      "emoji":"⌨️", "name":"Flow",      "sub":"In the zone"},
]

# ── Bootstrap ─────────────────────────────────────────────────────────────────
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(AUDIO_JS, unsafe_allow_html=True)

with st.spinner(""):
    try:
        data = load_all()
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}\n\nRun  python scripts/build_pipeline.py  first.")
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
st.markdown(
    '<p class="mm-label">02 — Or find a song / artist '
    '<span style="color:#3a2860;font-weight:400">(optional)</span></p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="mm-search-panel">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    song_name   = st.text_input("Song name", placeholder="e.g.  Lose Yourself").strip()
with c2:
    artist_name = st.text_input("Artist",    placeholder="e.g.  Eminem").strip()
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
        st.markdown(
            f'<p style="font-size:12px;color:#a090cc;letter-spacing:.1em">'
            f'Matched state → {label}</p>', unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:12px;color:#c05050;letter-spacing:.1em">'
            'No match — try another phrase or pick a card above.</p>',
            unsafe_allow_html=True,
        )
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
        use_ollama = st.checkbox("🤖  TinyLlama guides", value=False,
                                 help="Requires Ollama running locally")
        ok = ollama_ready() if use_ollama else False
        if use_ollama:
            st.markdown(
                f'<p style="font-size:12px;color:#a090cc">'
                f'{"🟢 Ready" if ok else "🔴 Not running — run: ollama serve"}</p>',
                unsafe_allow_html=True,
            )
st.markdown('</div>', unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="mm-wrap" style="padding-top:8px;padding-bottom:36px">', unsafe_allow_html=True)
has_input = bool(song_name or artist_name or st.session_state.mood_sel)
if not has_input:
    st.markdown(
        '<p style="text-align:center;color:#3a2860;font-size:12px;'
        'letter-spacing:.22em;text-transform:uppercase;padding:6px 0">'
        'Select a state or enter a song above</p>', unsafe_allow_html=True,
    )

if st.button("🎧  Start Session", disabled=not has_input,
             type="primary", use_container_width=True):
    recommender = HybridRecommenderSystem(
        number_of_recommendations=k,
        weight_content_based=content_w,
        mood_blend_weight=0.25,
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
        st.markdown(
            f'<div class="mm-badge">{m_meta["emoji"]}&nbsp; {m_meta["name"]}'
            f'&nbsp;·&nbsp; {m_meta["sub"]}</div>', unsafe_allow_html=True,
        )

recs = result["recommendations"]
st.markdown(
    f'<p class="mm-label" style="margin-bottom:18px">Your session — {len(recs)} tracks</p>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)
for i, row in recs.iterrows():
    name    = str(row.get("name",   "Unknown")).title()
    artist  = str(row.get("artist", "Unknown")).title()
    preview = row.get("spotify_preview_url", None)
    key     = f"{name} by {artist}"
    tags    = result["tags_per_song"].get(key, [])
    expl    = explanation_data["explanations"].get(key, {})

    if use_ollama and ok:
        with st.spinner(""):
            try:
                explanation = send_to_ollama(expl.get("ollama_prompt", ""))
            except Exception:
                explanation = expl.get("inline_reason", "")
    else:
        explanation = expl.get("inline_reason", "")

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
        if pd.notna(preview) and str(preview).startswith("http"):
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
st.markdown('<p class="mm-feedback-label">How did this session feel?</p>', unsafe_allow_html=True)
st.markdown('<div class="mm-wrap" style="padding-top:0;padding-bottom:48px">', unsafe_allow_html=True)
fb1, fb2, fb3, fb4, _ = st.columns([1, 1, 1, 1, 2])
with fb1:
    if st.button("😌  Calmer",     use_container_width=True): st.toast("Glad it helped.", icon="🎧")
with fb2:
    if st.button("⚡  Energised",  use_container_width=True): st.toast("Charged up.", icon="🔥")
with fb3:
    if st.button("🎯  Focused",    use_container_width=True): st.toast("In the zone.", icon="🧠")
with fb4:
    if st.button("↩  New session", use_container_width=True):
        st.session_state.result   = None
        st.session_state.mood_sel = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)