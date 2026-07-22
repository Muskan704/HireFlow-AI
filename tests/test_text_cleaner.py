from app.services.text_cleaner import clean_document_text


def test_clean_document_text_removes_email_headers_and_keeps_jd_content():
    raw = """
Outlook
EXTERNAL-Re: Right to Represent
From: recruiter@example.com
Sent: Tuesday, July 21, 2026 10:00 AM
To: candidate@example.com
Subject: Wabtec Corporation - Accounting II

Job Description
Controllership/Accounting II
Responsibilities:
Prepare reconciliations and month-end close activities.
Requirements:
Excel
Oracle or similar ERP system experience

-----Original Message-----
From: old-thread@example.com
This message and any attachments are confidential.
"""

    cleaned = clean_document_text(raw)

    assert "From:" not in cleaned
    assert "Sent:" not in cleaned
    assert "Subject:" not in cleaned
    assert "Original Message" not in cleaned
    assert "Job Description" in cleaned
    assert "Controllership/Accounting II" in cleaned
    assert "month-end close" in cleaned
    assert "Oracle or similar ERP system experience" in cleaned
