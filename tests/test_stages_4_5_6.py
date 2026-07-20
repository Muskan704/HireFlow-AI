"""
Automated tests for Stage 4 (Scoring), Stage 5 (Ranking), Stage 6 (Summary).

These tests use mock data to ensure the pipeline stages work correctly
without requiring actual LLM API calls during testing.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.models.schemas import ResumeData, JDData, WorkExperience, Project, Certification
from app.models.results import (
    CandidateScores,
    SectionScore,
    RankedCandidate,
    CandidateSummary,
    FilterResult,
)
from app.services.scorer_semantic import score_candidate_semantic
from app.services.scorer_llm import score_candidate_llm
from app.services.ranker import rank_candidates, load_weights, _blend_section_score
from app.services.summariser import summarise_candidate


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_resume() -> ResumeData:
    """Create a sample resume for testing."""
    return ResumeData(
        resume_id="test-resume-001",
        candidate_name="John Doe",
        email="john@example.com",
        phone="123-456-7890",
        location="New York, NY",
        summary="Experienced software engineer with 5 years in full-stack development.",
        skills=["Python", "JavaScript", "React", "Node.js", "PostgreSQL", "Docker"],
        programming_languages=["Python", "JavaScript"],
        frameworks_and_tools=["React", "Node.js", "Docker"],
        work_experience=[
            WorkExperience(
                title="Senior Software Engineer",
                company="Tech Corp",
                duration_months=36,
                responsibilities=["Built microservices", "Led team of 5"],
                technologies=["Python", "Docker", "PostgreSQL"],
            ),
            WorkExperience(
                title="Software Engineer",
                company="Startup Inc",
                duration_months=24,
                responsibilities=["Developed React frontend", "Built REST APIs"],
                technologies=["JavaScript", "React", "Node.js"],
            ),
        ],
        projects=[
            Project(
                name="E-commerce Platform",
                description="Full-stack e-commerce application",
                technologies=["React", "Node.js", "PostgreSQL"],
                impact="Increased sales by 30%",
            ),
        ],
        education=[],
        certifications=[
            Certification(name="AWS Solutions Architect", issuer="AWS", year=2023),
        ],
        total_experience_months=60,
    )


@pytest.fixture
def sample_jd() -> JDData:
    """Create a sample job description for testing."""
    return JDData(
        job_title="Senior Full Stack Engineer",
        company="Hiring Corp",
        location="Remote",
        location_strict=False,
        responsibilities=["Build scalable web applications", "Lead technical projects"],
        must_have_skills=["Python", "JavaScript", "React", "PostgreSQL"],
        nice_to_have_skills=["Docker", "AWS"],
        must_have_skill_groups=[],
        required_certifications=[],
        required_education=None,  # Optional[str], not a list
        min_experience_months=36,
    )


@pytest.fixture
def sample_resume2() -> ResumeData:
    """Create a second sample resume for ranking tests."""
    return ResumeData(
        resume_id="test-resume-002",
        candidate_name="Jane Smith",
        email="jane@example.com",
        phone="987-654-3210",
        location="San Francisco, CA",
        summary="Junior developer with strong fundamentals.",
        skills=["Python", "JavaScript", "React"],
        programming_languages=["Python", "JavaScript"],
        frameworks_and_tools=["React"],
        work_experience=[
            WorkExperience(
                title="Junior Developer",
                company="Small Co",
                duration_months=12,
                responsibilities=["Built React components"],
                technologies=["JavaScript", "React"],
            ),
        ],
        projects=[],
        education=[],
        certifications=[],
        total_experience_months=12,
    )


# ── Stage 4A: Semantic Scoring Tests ─────────────────────────────────────────────

class TestSemanticScoring:
    """Tests for Stage 4A - Semantic scoring using embeddings."""

    def test_semantic_score_returns_candidate_scores(self, sample_resume, sample_jd):
        """Test that semantic scoring returns properly structured CandidateScores."""
        result = score_candidate_semantic(sample_resume, sample_jd)

        assert isinstance(result, CandidateScores)
        assert result.method == "semantic"
        assert result.resume_id == sample_resume.resume_id
        assert result.candidate_name == sample_resume.candidate_name
        assert len(result.section_scores) == 2  # skills and experience

    def test_semantic_score_section_ranges(self, sample_resume, sample_jd):
        """Test that all section scores are in valid 0-1 range."""
        result = score_candidate_semantic(sample_resume, sample_jd)

        for section in result.section_scores:
            assert 0.0 <= section.score <= 1.0, f"Section {section.section_name} score out of range"
            assert section.method == "semantic"
            assert section.reasoning is None  # Semantic has no reasoning

    def test_semantic_score_skills_section(self, sample_resume, sample_jd):
        """Test that skills section is scored."""
        result = score_candidate_semantic(sample_resume, sample_jd)
        skills_section = result.get_section("skills")

        assert skills_section is not None
        assert skills_section.section_name == "skills"
        # High overlap should produce reasonable score
        assert skills_section.score > 0.0

    def test_semantic_score_experience_section(self, sample_resume, sample_jd):
        """Test that experience section is scored."""
        result = score_candidate_semantic(sample_resume, sample_jd)
        exp_section = result.get_section("experience")

        assert exp_section is not None
        assert exp_section.section_name == "experience"


# ── Stage 4B: LLM Scoring Tests ──────────────────────────────────────────────────

class TestLLMScoring:
    """Tests for Stage 4B - LLM-based scoring."""

    @patch("app.services.scorer_llm._call_llm_scorer")
    def test_llm_score_returns_candidate_scores(self, mock_llm, sample_resume, sample_jd):
        """Test that LLM scoring returns properly structured CandidateScores."""
        mock_llm.return_value = MagicMock(
            skills_score=8.0,
            skills_reasoning="Good skill match",
            experience_score=7.0,
            experience_reasoning="Solid experience",
        )

        result = score_candidate_llm(sample_resume, sample_jd)

        assert isinstance(result, CandidateScores)
        assert result.method == "llm"
        assert result.resume_id == sample_resume.resume_id
        assert len(result.section_scores) == 2

    @patch("app.services.scorer_llm._call_llm_scorer")
    def test_llm_score_normalizes_to_unit_range(self, mock_llm, sample_resume, sample_jd):
        """Test that LLM scores (0-10) are normalized to 0-1."""
        mock_llm.return_value = MagicMock(
            skills_score=8.0,
            skills_reasoning="",
            experience_score=6.0,
            experience_reasoning="",
        )

        result = score_candidate_llm(sample_resume, sample_jd)

        skills = result.get_section("skills")
        exp = result.get_section("experience")

        assert skills.score == 0.8  # 8/10
        assert exp.score == 0.6     # 6/10
        assert skills.raw_score == 8.0
        assert exp.raw_score == 6.0

    @patch("app.services.scorer_llm._call_llm_scorer")
    def test_llm_score_includes_reasoning(self, mock_llm, sample_resume, sample_jd):
        """Test that LLM scores include reasoning strings."""
        mock_llm.return_value = MagicMock(
            skills_score=9.0,
            skills_reasoning="All required skills present",
            experience_score=8.0,
            experience_reasoning="Directly relevant experience",
        )

        result = score_candidate_llm(sample_resume, sample_jd)

        skills = result.get_section("skills")
        exp = result.get_section("experience")

        assert skills.reasoning == "All required skills present"
        assert exp.reasoning == "Directly relevant experience"


# ── Stage 5: Ranking Tests ────────────────────────────────────────────────────────

class TestRanking:
    """Tests for Stage 5 - Weighted ranking."""

    def test_load_weights_returns_dict(self):
        """Test that load_weights returns a dictionary."""
        weights = load_weights()

        assert isinstance(weights, dict)
        assert "section_weights" in weights

    def test_rank_candidates_single(self, sample_resume):
        """Test ranking with a single candidate."""
        semantic = CandidateScores(
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            method="semantic",
            section_scores=[
                SectionScore(section_name="skills", score=0.8, method="semantic"),
                SectionScore(section_name="experience", score=0.7, method="semantic"),
            ],
        )
        llm = CandidateScores(
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            method="llm",
            section_scores=[
                SectionScore(section_name="skills", score=0.9, method="llm"),
                SectionScore(section_name="experience", score=0.8, method="llm"),
            ],
        )

        result = rank_candidates([sample_resume], [semantic], [llm])

        assert len(result) == 1
        assert result[0].rank == 1
        assert result[0].resume_id == sample_resume.resume_id

    def test_rank_candidates_multiple_sorts_correctly(self, sample_resume, sample_resume2):
        """Test that multiple candidates are sorted by score descending."""
        # Higher scoring candidate
        semantic1 = CandidateScores(
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            method="semantic",
            section_scores=[
                SectionScore(section_name="skills", score=0.9, method="semantic"),
                SectionScore(section_name="experience", score=0.9, method="semantic"),
            ],
        )
        llm1 = CandidateScores(
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            method="llm",
            section_scores=[
                SectionScore(section_name="skills", score=0.9, method="llm"),
                SectionScore(section_name="experience", score=0.9, method="llm"),
            ],
        )

        # Lower scoring candidate
        semantic2 = CandidateScores(
            resume_id=sample_resume2.resume_id,
            candidate_name=sample_resume2.candidate_name,
            method="semantic",
            section_scores=[
                SectionScore(section_name="skills", score=0.5, method="semantic"),
                SectionScore(section_name="experience", score=0.4, method="semantic"),
            ],
        )
        llm2 = CandidateScores(
            resume_id=sample_resume2.resume_id,
            candidate_name=sample_resume2.candidate_name,
            method="llm",
            section_scores=[
                SectionScore(section_name="skills", score=0.5, method="llm"),
                SectionScore(section_name="experience", score=0.4, method="llm"),
            ],
        )

        result = rank_candidates(
            [sample_resume, sample_resume2],
            [semantic1, semantic2],
            [llm1, llm2],
        )

        assert len(result) == 2
        assert result[0].rank == 1
        assert result[0].overall_score > result[1].overall_score
        assert result[1].rank == 2

    def test_blend_section_score_semantic_mode(self):
        """Test blending in semantic mode returns semantic score."""
        semantic = CandidateScores(
            resume_id="test",
            method="semantic",
            section_scores=[SectionScore(section_name="skills", score=0.8, method="semantic")],
        )
        llm = CandidateScores(
            resume_id="test",
            method="llm",
            section_scores=[SectionScore(section_name="skills", score=0.6, method="llm")],
        )

        result = _blend_section_score("skills", semantic, llm, "semantic", 0.5)

        assert result == 0.8

    def test_blend_section_score_llm_mode(self):
        """Test blending in LLM mode returns LLM score."""
        semantic = CandidateScores(
            resume_id="test",
            method="semantic",
            section_scores=[SectionScore(section_name="skills", score=0.8, method="semantic")],
        )
        llm = CandidateScores(
            resume_id="test",
            method="llm",
            section_scores=[SectionScore(section_name="skills", score=0.6, method="llm")],
        )

        result = _blend_section_score("skills", semantic, llm, "llm", 0.5)

        assert result == 0.6

    def test_blend_section_score_both_mode(self):
        """Test blending in both mode blends scores."""
        semantic = CandidateScores(
            resume_id="test",
            method="semantic",
            section_scores=[SectionScore(section_name="skills", score=0.8, method="semantic")],
        )
        llm = CandidateScores(
            resume_id="test",
            method="llm",
            section_scores=[SectionScore(section_name="skills", score=0.6, method="llm")],
        )

        # blend_ratio=0.5: 0.5 * 0.6 + 0.5 * 0.8 = 0.7
        result = _blend_section_score("skills", semantic, llm, "both", 0.5)

        assert result == 0.7


# ── Stage 6: Summary Tests ────────────────────────────────────────────────────────

class TestSummary:
    """Tests for Stage 6 - LLM-generated summaries."""

    @patch("app.services.summariser._call_llm_summariser")
    def test_summarise_candidate_returns_summary(self, mock_llm, sample_resume, sample_jd):
        """Test that summarise_candidate returns CandidateSummary."""
        mock_llm.return_value = MagicMock(
            strengths=["Strong skills", "Good experience"],
            gaps=["Missing AWS"],
            recommendation="Recommended for interview.",
        )

        ranked = RankedCandidate(
            rank=1,
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            overall_score=0.85,
            section_scores={"skills": 0.9, "experience": 0.8},
        )

        result = summarise_candidate(sample_resume, sample_jd, ranked)

        assert isinstance(result, CandidateSummary)
        assert result.resume_id == sample_resume.resume_id
        assert result.candidate_name == sample_resume.candidate_name
        assert len(result.strengths) == 2
        assert len(result.gaps) == 1

    @patch("app.services.summariser._call_llm_summariser")
    def test_summary_handles_empty_gaps(self, mock_llm, sample_resume, sample_jd):
        """Test that summary handles candidates with no gaps."""
        mock_llm.return_value = MagicMock(
            strengths=["Perfect match"],
            gaps=[],
            recommendation="Strongly recommended.",
        )

        ranked = RankedCandidate(
            rank=1,
            resume_id=sample_resume.resume_id,
            candidate_name=sample_resume.candidate_name,
            overall_score=0.95,
            section_scores={"skills": 1.0, "experience": 0.9},
        )

        result = summarise_candidate(sample_resume, sample_jd, ranked)

        assert result.gaps == []


# ── Integration Tests ─────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Integration tests for the full pipeline flow."""

    def test_filter_result_passed_candidate(self, sample_resume, sample_jd):
        """Test that a qualified candidate passes the hard filter."""
        from app.services.hard_filter import run_hard_filter

        result = run_hard_filter(sample_resume, sample_jd)

        assert result.passed is True
        assert len(result.reject_reasons) == 0

    def test_filter_result_rejected_candidate(self, sample_jd):
        """Test that an unqualified candidate is rejected."""
        from app.services.hard_filter import run_hard_filter

        # Resume missing required skills
        unqualified_resume = ResumeData(
            resume_id="unqualified-001",
            candidate_name="Unqualified Candidate",
            skills=["Python"],  # Missing most required skills (JavaScript, React, PostgreSQL)
            work_experience=[],
            projects=[],
            education=[],
            certifications=[],
            total_experience_months=6,  # Below minimum of 36
        )

        result = run_hard_filter(unqualified_resume, sample_jd)

        assert result.passed is False
        assert len(result.reject_reasons) > 0
