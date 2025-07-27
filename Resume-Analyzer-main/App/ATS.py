
def calculate_ats_score(resume_text):
    """
    Calculate ATS compatibility score (0-100) based on key resume factors
    """
    score = 0
    
    # 1. Keyword Optimization (30 points max)
    keywords = ["skills", "experience", "education", "projects", 
                "achievements", "certifications", "summary"]
    found_keywords = [kw for kw in keywords if kw in resume_text.lower()]
    score += min(30, (len(found_keywords)/len(keywords)*30))
    
    # 2. Section Presence (20 points max)
    sections = ["work experience", "education", "skills", "contact information"]
    found_sections = [sec for sec in sections if sec in resume_text.lower()]
    score += min(20, (len(found_sections)/len(sections)*20))
    
    # 3. Length Check (10 points)
    page_count = len(resume_text.split("\f"))  # Count page breaks
    if 1 <= page_count <= 2:
        score += 10
    
    # 4. Formatting (20 points)
    formatting_checks = {
        "consistent_bullets": "- " in resume_text or "* " in resume_text,
        "consistent_dates": any(x in resume_text.lower() for x in ["present", "20", "jan", "feb"]),
        "no_tables": "<table>" not in resume_text.lower(),
        "standard_fonts": "times new roman" in resume_text.lower() or "arial" in resume_text.lower()
    }
    score += sum(5 for check in formatting_checks.values() if check)
    
    # 5. Contact Info (10 points)
    contact_checks = {
        "has_email": "@" in resume_text,
        "has_phone": any(x in resume_text for x in ["phone", "mobile", "tel"]),
    }
    score += sum(5 for check in contact_checks.values() if check)
    
    # 6. Action Verbs (10 points)
    action_verbs = ["managed", "led", "developed", "implemented", 
                   "increased", "reduced", "optimized", "created"]
    found_verbs = [verb for verb in action_verbs if verb in resume_text.lower()]
    score += min(10, (len(found_verbs)/len(action_verbs)*10))
    
    return min(100, int(score))
