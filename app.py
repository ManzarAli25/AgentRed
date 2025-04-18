import streamlit as st
import time
from AgentRed import (
    extract_opinions,
    parse_debate_output,
    create_support_chain,
    create_counter_chain,
    run_debate,
    evaluate_debate
)

# Set page config
st.set_page_config(
    page_title="AgentRed - Reddit Opinion Debate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 10px;
    }
    .debate-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .support-card {
        background-color: #FFE6E6;
        border-left: 5px solid #FF4B4B;
    }
    .counter-card {
        background-color: #E6FFE6;
        border-left: 5px solid #4CAF50;
    }
    .judgment-box {
        padding: 20px;
        background-color: #F0F2F6;
        border-radius: 10px;
        margin-top: 20px;
    }
    .progress-bar {
        height: 4px;
        background-color: #FF4B4B;
        margin: 10px 0;
    }
    .emoji-large {
        font-size: 1.5em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'debate_axis' not in st.session_state:
    st.session_state.debate_axis = None
if 'opinion_1' not in st.session_state:
    st.session_state.opinion_1 = None
if 'opinion_2' not in st.session_state:
    st.session_state.opinion_2 = None
if 'debate_history' not in st.session_state:
    st.session_state.debate_history = None
if 'judgment' not in st.session_state:
    st.session_state.judgment = None
if 'opinions_extracted' not in st.session_state:
    st.session_state.opinions_extracted = False
if 'debate_setup' not in st.session_state:
    st.session_state.debate_setup = False
if 'debate_completed' not in st.session_state:
    st.session_state.debate_completed = False
if 'judgment_given' not in st.session_state:
    st.session_state.judgment_given = False
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = None

# App header
st.title("🤖 AgentRed - Reddit Opinion Debate")
st.markdown("""
<div style="background-color: #F0F2F6; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h3 style="color: #2E4053; margin:0;">Extract Reddit opinions ➔ Simulate AI debate ➔ Get expert judgment</h3>
</div>
""", unsafe_allow_html=True)

# Progress tracker
progress_text = ["Opinion Extraction", "Debate Setup", "Debate Simulation", "Judgment"]
current_progress = sum([
    st.session_state.opinions_extracted,
    st.session_state.debate_setup,
    st.session_state.debate_completed,
    st.session_state.judgment_given
])
st.markdown(f"""
<div style="margin: 20px 0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
        {''.join([f'<div style="text-align: center;"><div style="background-color: {"#FF4B4B" if i <= current_progress else "#D3D3D3"}; border-radius: 50%; width: 30px; height: 30px; margin: 0 auto; display: flex; align-items: center; justify-content: center; color: white;">{i+1}</div>{t}</div>' for i, t in enumerate(progress_text)])}
    </div>
    <div class="progress-bar" style="width: {25 * current_progress}%;"></div>
</div>
""", unsafe_allow_html=True)

# Step 1: Topic Input & Opinion Extraction
with st.container():
    st.markdown("<h2 class='header'>🔍 Step 1: Topic Input & Opinion Extraction</h2>", unsafe_allow_html=True)
    topic = st.text_input("**Enter a debate topic:**", placeholder="e.g., Climate change, Universal basic income, AI regulation...")
    
    if st.button("🚀 Extract Opinions from Reddit", type="primary", disabled=not topic):
        with st.spinner("🔎 Scanning Reddit for opposing viewpoints... This may take 1-2 minutes"):
            try:
                messages, extracted_output = extract_opinions(topic)
                st.session_state.raw_output = extracted_output.content
                st.session_state.debate_axis, st.session_state.opinion_1, st.session_state.opinion_2 = parse_debate_output(extracted_output.content)
                st.session_state.opinions_extracted = True
                
                with st.expander("📋 View Raw Extracted Opinions", expanded=False):
                    st.markdown(st.session_state.raw_output)
            except Exception as e:
                st.error(f"❌ Error extracting opinions: {str(e)}")

# Step 2: Debate Setup
if st.session_state.opinions_extracted:
    with st.container():
        st.markdown("<h2 class='header'>⚔️ Step 2: Debate Setup</h2>", unsafe_allow_html=True)
        
        if st.button("🎮 Configure Debate Agents", type="primary"):
            st.session_state.debate_setup = True
        
        if st.session_state.debate_setup:
            st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3>📌 Debate Axis</h3>
                <p style="font-size: 1.1em; color: #2E4053;">{st.session_state.debate_axis}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="debate-card support-card">
                    <h3>🔴 Supporting Viewpoint</h3>
                    <h4>{st.session_state.opinion_1['title']}</h4>
                    <p><strong>Summary:</strong> {st.session_state.opinion_1['summary']}</p>
                    <details>
                        <summary><strong>Key Arguments</strong></summary>
                        <ul>
                            {"".join([f"<li>{arg}</li>" for arg in st.session_state.opinion_1['arguments']])}
                        </ul>
                    </details>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="debate-card counter-card">
                    <h3>🟢 Opposing Viewpoint</h3>
                    <h4>{st.session_state.opinion_2['title']}</h4>
                    <p><strong>Summary:</strong> {st.session_state.opinion_2['summary']}</p>
                    <details>
                        <summary><strong>Key Arguments</strong></summary>
                        <ul>
                            {"".join([f"<li>{arg}</li>" for arg in st.session_state.opinion_2['arguments']])}
                        </ul>
                    </details>
                </div>
                """, unsafe_allow_html=True)

# Step 3: Debate Simulation
if st.session_state.debate_setup:
    with st.container():
        st.markdown("<h2 class='header'>💬 Step 3: Debate Simulation</h2>", unsafe_allow_html=True)
        
        if st.button("🎬 Start Live Debate", type="primary"):
            with st.spinner("🤖 AI agents are debating... Grab some popcorn, this may take 2-3 minutes"):
                try:
                    support_chain = create_support_chain(st.session_state.debate_axis, st.session_state.opinion_1)
                    counter_chain = create_counter_chain(st.session_state.debate_axis, st.session_state.opinion_2)
                    debate_history = run_debate(support_chain, counter_chain, num_rounds=3)
                    st.session_state.debate_history = debate_history
                    st.session_state.debate_completed = True
                except Exception as e:
                    st.error(f"❌ Debate error: {str(e)}")
        
        if st.session_state.debate_completed:
            st.markdown("<h3 style='margin-top: 30px;'>🗣️ Debate Transcript</h3>", unsafe_allow_html=True)
            for i, message in enumerate(st.session_state.debate_history):
                is_support = i % 2 == 0
                st.markdown(f"""
                <div style="margin: 10px 0; padding: 15px; border-radius: 10px; background-color: {'#FFE6E6' if is_support else '#E6FFE6'}; border-left: 5px solid {'#FF4B4B' if is_support else '#4CAF50'};">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="background-color: {'#FF4B4B' if is_support else '#4CAF50'}; color: white; border-radius: 5px; padding: 5px 10px; font-weight: bold;">
                            {'🔴 Support' if is_support else '🟢 Counter'} • Round {i//2 +1}
                        </div>
                    </div>
                    {message.content}
                """, unsafe_allow_html=True)

# Step 4: Judgment Phase
if st.session_state.debate_completed:
    with st.container():
        st.markdown("<h2 class='header'>⚖️ Step 4: Judgment Phase</h2>", unsafe_allow_html=True)
        
        if st.button("🏆 Get Final Judgment", type="primary"):
            with st.spinner("🧑⚖️ Analyzing debate quality..."):
                try:
                    judgment = evaluate_debate(st.session_state.debate_history)
                    st.session_state.judgment = judgment
                    st.session_state.judgment_given = True
                except Exception as e:
                    st.error(f"❌ Judgment error: {str(e)}")
        
        if st.session_state.judgment_given:
            st.markdown(f"""
            <div class="judgment-box">
                <h3 style="color: #2E4053; margin-top: 0;">🎯 Final Verdict</h3>
                <div style="background-color: white; padding: 20px; border-radius: 10px;">
                    {st.session_state.judgment}
                
            """, unsafe_allow_html=True)

# Reset functionality
if st.session_state.opinions_extracted:
    st.markdown("---")
    if st.button("🔄 Start New Debate", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()