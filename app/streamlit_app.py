"""
Streamlit UI for SHL Assessment Recommendation System
"""

import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
USE_LOCAL_ENGINE = False  # Set to True if API is not available

# Try to import local engine as fallback
if USE_LOCAL_ENGINE:
    try:
        from src.recommender import RecommendationEngine
        local_engine = RecommendationEngine()
        st.sidebar.success("✓ Running in standalone mode")
    except Exception as e:
        st.sidebar.error(f"Error loading local engine: {e}")
        local_engine = None
else:
    local_engine = None

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender", 
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .assessment-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .assessment-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .assessment-score {
        font-size: 0.9rem;
        color: #28a745;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 SHL Assessment Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find the perfect talent assessments for your hiring needs</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system recommends the top SHL talent assessments based on:
    - Natural language queries
    - Job description text
    - Job description URLs
    
    The system uses **semantic search** with **diversity constraints** to ensure balanced recommendations across different assessment categories.
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    top_k = st.slider("Number of recommendations", min_value=5, max_value=15, value=10, step=1)
    use_llm = st.checkbox("Use LLM enhancement (if available)", value=False, 
                          help="Enhance query using LLM for better results (requires Groq API key)")
    
# Main content
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📝 Input")
    
    input_type = st.radio(
        "Select input type:",
        options=["Natural Language Query", "Job Description Text", "Job Description URL"],
        help="Choose how you want to provide your requirements"
    )
    
    query = None
    jd_text = None
    jd_url = None
    
    if input_type == "Natural Language Query":
        query = st.text_area(
            "Enter your query:",
            placeholder="Example: I want to hire Java developers with 3 years of experience...",
            height=150,
            help="Describe what kind of assessment you're looking for"
        )
    elif input_type == "Job Description Text":
        jd_text = st.text_area(
            "Paste job description:",
            placeholder="Paste the complete job description here...",
            height=300,
            help="Paste the full job description text"
        )
    else:  # JD URL
        jd_url = st.text_input(
            "Enter job description URL:",
            placeholder="https://example.com/job-posting",
            help="Enter URL of the job posting"
        )
    
    submit_button = st.button("🔍 Get Recommendations", type="primary")

with col2:
    st.subheader("✨ Recommendations")
    
    if submit_button:
        # Validate input
        if not any([query, jd_text, jd_url]):
            st.error("Please provide at least one input!")
        else:
            with st.spinner("Searching for the best assessments..."):
                try:
                    if local_engine:
                        # Use local engine
                        result = local_engine.recommend(
                            query=query,
                            jd_text=jd_text,
                            jd_url=jd_url,
                            top_k=top_k,
                            use_llm=use_llm
                        )
                    else:
                        # Call API
                        payload = {
                            "query": query,
                            "jd_text": jd_text,
                            "jd_url": jd_url,
                            "top_k": top_k,
                            "use_llm": use_llm
                        }
                        
                        response = requests.post(
                            f"{API_URL}/recommend",
                            json=payload,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")
                            result = None
                    
                    if result and 'recommendations' in result:
                        recommendations = result['recommendations']
                        
                        # Display summary
                        st.success(f"Found {len(recommendations)} relevant assessments!")
                        
                        # Create DataFrame for table view
                        df_data = []
                        for i, rec in enumerate(recommendations, 1):
                            df_data.append({
                                'Rank': i,
                                'Assessment Name': rec['assessment_name'],
                                'Type': rec['test_type'],
                                'Duration': f"{rec['duration_minutes']} min" if rec['duration_minutes'] else 'N/A',
                                'Score': f"{rec['relevance_score']:.3f}",
                                'Adaptive': rec['adaptive_support'],
                                'Remote': rec['remote_support']
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        # Display table
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        st.divider()
                        
                        # Display detailed cards
                        st.subheader("📋 Detailed View")
                        
                        # Category distribution
                        categories = {}
                        for rec in recommendations:
                            cat = rec['test_type']
                            categories[cat] = categories.get(cat, 0) + 1
                        
                        st.write("**Category Distribution:**")
                        cat_col1, cat_col2, cat_col3 = st.columns(3)
                        for i, (cat, count) in enumerate(sorted(categories.items(), key=lambda x: x[1], reverse=True)):
                            if i % 3 == 0:
                                cat_col1.metric(cat, count)
                            elif i % 3 == 1:
                                cat_col2.metric(cat, count)
                            else:
                                cat_col3.metric(cat, count)
                        
                        st.divider()
                        
                        # Display cards
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{i} - {rec['assessment_name']}", expanded=(i <= 3)):
                                st.markdown(f"**Type:** {rec['test_type']}")
                                st.markdown(f"**Relevance Score:** {rec['relevance_score']:.4f}")
                                st.markdown(f"**Duration:** {rec['duration_minutes']} minutes" if rec['duration_minutes'] else "**Duration:** Not specified")
                                st.markdown(f"**Adaptive Support:** {rec['adaptive_support']}")
                                st.markdown(f"**Remote Support:** {rec['remote_support']}")
                                st.markdown(f"**Description:** {rec['description']}")
                                st.markdown(f"**Link:** [View Assessment]({rec['url']})")
                    
                except requests.exceptions.ConnectionError:
                    st.error(f"""
                    ❌ **Connection Error**
                    
                    Could not connect to the API at {API_URL}.
                    
                    **Options:**
                    1. Make sure the API server is running locally
                    2. Start the API with: `uvicorn api.main:app --reload`
                    3. Or set USE_LOCAL_ENGINE=True in the code to run standalone
                    """)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Enter your requirements on the left and click 'Get Recommendations'")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with Streamlit • Powered by Semantic Search & AI
</div>
""", unsafe_allow_html=True)
