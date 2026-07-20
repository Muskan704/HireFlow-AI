"""
Streamlit Web Interface for Recruitment Intelligence Platform.

Run with: streamlit run app.py
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.pipeline import run_pipeline
from app.models.results import PipelineResult

# Page config
st.set_page_config(
    page_title="Recruitment Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .score-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 Recruitment Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown("### Upload resumes and job description to get ranked candidates with detailed analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This interface uses the full 7-stage pipeline:\n\n1. Parse\n2. Extract\n3. Hard Filter\n4. Score\n5. Rank\n6. Summarise\n7. Knowledge Brief")
    
    st.divider()
    st.header("📊 Scoring Weights")
    st.caption("Live from weights.json:")
    try:
        from app.services.ranker import load_weights
        _weights = load_weights()
        _display_weights = {
            **_weights.get("skill_matching", {}),
            **_weights.get("experience_matching", {}),
            **_weights.get("additional_components", {}),
        }
        st.json({k: f"{v:.0%}" for k, v in _display_weights.items()})
    except Exception as e:
        st.caption(f"(Could not load weights.json: {e})")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Job Description")
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=['pdf', 'docx'],
        help="Upload the job description (PDF or DOCX)"
    )

with col2:
    st.header("👥 Resumes")
    resume_files = st.file_uploader(
        "Upload Resumes",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload one or more resume files"
    )

# Process button
st.divider()

if st.button("🚀 Run Pipeline", type="primary", use_container_width=True, disabled=not (jd_file and resume_files)):
    if not jd_file:
        st.error("Please upload a Job Description")
    elif not resume_files:
        st.error("Please upload at least one Resume")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📋 Processing Job Description...")
        progress_bar.progress(10)
        
        try:
            # Prepare files
            jd_bytes = jd_file.read()
            jd_source = (jd_bytes, jd_file.name)
            
            resume_sources = []
            total_resumes = len(resume_files)
            
            for i, resume_file in enumerate(resume_files):
                status_text.text(f"📄 Processing resume {i+1}/{total_resumes}: {resume_file.name}")
                progress_bar.progress(10 + int(30 * (i / total_resumes)))
                resume_bytes = resume_file.read()
                resume_sources.append((resume_bytes, resume_file.name))
            
            status_text.text("🔄 Running pipeline... This may take a few minutes.")
            progress_bar.progress(50)
            
            # Run pipeline
            result = run_pipeline(resume_sources, jd_source)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Store result in session state
            st.session_state['result'] = result
            
        except Exception as e:
            st.error(f"❌ Pipeline Error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Display results if available
if 'result' in st.session_state:
    result: PipelineResult = st.session_state['result']
    
    st.divider()
    st.header("📊 Results")
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("JD Title", result.jd_title[:30] + "..." if len(result.jd_title or "") > 30 else result.jd_title)
    
    with col2:
        st.metric("Total Processed", result.total_resumes_processed)
    
    with col3:
        st.metric("Passed Filter", result.total_passed_filter, delta_color="normal")
    
    with col4:
        st.metric("Rejected", result.total_rejected, delta_color="inverse")
    
    # Ranked candidates
    st.divider()
    st.header("🏆 Ranked Candidates")
    
    if result.ranked_candidates:
        for candidate in result.ranked_candidates:
            score_percent = round(candidate.overall_score * 100)
            
            # Candidate card
            with st.expander(f"#{candidate.rank} {candidate.candidate_name} - Score: {score_percent}%", expanded=(candidate.rank == 1)):
                
                # Score progress bar
                st.progress(score_percent / 100, text=f"Overall Score: {score_percent}%")
                
                # Section scores
                st.subheader("📈 Component Scores")
                score_cols = st.columns(3)
                section_items = list(candidate.section_scores.items())
                
                for i, (section, score) in enumerate(section_items):
                    with score_cols[i % 3]:
                        st.metric(
                            section.replace("_", " ").title(),
                            f"{round(score * 100)}%"
                        )
                
                # Fit summary
                if candidate.fit_summary:
                    st.subheader("📝 Summary")
                    st.info(candidate.fit_summary)
                
                # Knowledge brief
                brief = next((b for b in result.knowledge_briefs if b.resume_id == candidate.resume_id), None)
                
                if brief:
                    st.subheader("📋 Knowledge Brief")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Role Overview**")
                        st.write(brief.role_overview)
                        
                        st.markdown("**Career Summary**")
                        st.write(brief.career_summary)
                        
                        if brief.key_achievements:
                            st.markdown("**Key Achievements**")
                            for achievement in brief.key_achievements:
                                st.write(f"✅ {achievement}")
                    
                    with col2:
                        if brief.areas_to_probe:
                            st.markdown("**🔍 Areas to Probe**")
                            for area in brief.areas_to_probe:
                                st.write(f"❓ {area}")
                        
                        if brief.suggested_talking_points:
                            st.markdown("**💬 Talking Points**")
                            for point in brief.suggested_talking_points:
                                st.write(f"💡 {point}")
                    
                    # Quick reference
                    st.markdown("---")
                    st.markdown("**Quick Reference**")
                    ref_cols = st.columns(4)
                    with ref_cols[0]:
                        st.caption("Experience")
                        st.write(f"{brief.years_of_experience or 0} years")
                    with ref_cols[1]:
                        st.caption("Current Role")
                        st.write(brief.current_or_last_role or "N/A")
                    with ref_cols[2]:
                        st.caption("Education")
                        st.write(brief.education_highlight or "N/A")
                    with ref_cols[3]:
                        st.caption("Location")
                        st.write(brief.location or "N/A")
    else:
        st.warning("No candidates passed the hard filter.")
    
    # Rejected candidates
    if result.rejected_candidates:
        st.divider()
        st.header("❌ Rejected Candidates")
        
        for rejected in result.rejected_candidates:
            close_miss_tag = " 🟡 CLOSE MISS" if rejected.is_close_miss else ""
            with st.expander(f"🚫 {rejected.candidate_name}{close_miss_tag}"):
                st.markdown("**Rejection Reasons:**")
                for reason in rejected.reject_reasons:
                    st.write(f"• {reason}")

                if rejected.rejection_summary:
                    if rejected.is_close_miss:
                        st.warning(
                            f"🟡 **Close miss — may be worth a second look**\n\n{rejected.rejection_summary}"
                        )
                    else:
                        st.info(rejected.rejection_summary)

                if rejected.checks:
                    st.markdown("**Check Results:**")
                    st.json(rejected.checks)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Recruitment Intelligence Platform • Pre-Call Setup Module</p>
    <p>Built with ❤️ using Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)