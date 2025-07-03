"""
Date utility functions for healthcare assistant
"""
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

def get_current_date_info() -> Dict[str, str]:
    """
    Get current date information for the healthcare assistant.
    Returns current date, tomorrow, and day of week information.
    """
    # Current date is July 3rd, 2025
    current_date = datetime(2025, 7, 3)
    
    # Calculate relative dates
    tomorrow = current_date + timedelta(days=1)
    next_week_start = current_date + timedelta(days=(7 - current_date.weekday()))  # Next Monday
    
    return {
        "today": current_date.strftime("%Y-%m-%d"),
        "today_formatted": current_date.strftime("%A, %B %d, %Y"),
        "tomorrow": tomorrow.strftime("%Y-%m-%d"),
        "tomorrow_formatted": tomorrow.strftime("%A, %B %d, %Y"),
        "current_weekday": current_date.strftime("%A"),
        "next_monday": next_week_start.strftime("%Y-%m-%d"),
        "current_time": current_date.strftime("%H:%M"),
        "current_year": "2025"
    }

def parse_relative_date(date_input: str, reference_date: Optional[datetime] = None) -> str:
    """
    Parse relative date expressions like 'tomorrow', 'next Tuesday', etc.
    Returns date in YYYY-MM-DD format.
    """
    if reference_date is None:
        reference_date = datetime(2025, 7, 3)  # July 3rd, 2025
    
    date_input = date_input.lower().strip()
    
    # Handle common relative dates
    if date_input in ['today']:
        return reference_date.strftime("%Y-%m-%d")
    
    elif date_input in ['tomorrow']:
        return (reference_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    elif 'next week' in date_input:
        # Next Monday
        days_ahead = 7 - reference_date.weekday()
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    elif 'monday' in date_input:
        # Next Monday
        days_ahead = (0 - reference_date.weekday()) % 7
        if days_ahead == 0:  # If today is Monday, get next Monday
            days_ahead = 7
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    elif 'tuesday' in date_input:
        days_ahead = (1 - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    elif 'wednesday' in date_input:
        days_ahead = (2 - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    elif 'thursday' in date_input:
        days_ahead = (3 - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    elif 'friday' in date_input:
        days_ahead = (4 - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (reference_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # If no relative date found, try to parse as absolute date
    try:
        # Try common date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"]:
            try:
                parsed_date = datetime.strptime(date_input, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except:
        pass
    
    # If all else fails, return tomorrow as default
    logging.warning(f"Could not parse date '{date_input}', defaulting to tomorrow")
    return (reference_date + timedelta(days=1)).strftime("%Y-%m-%d")

def get_date_context_for_llm() -> str:
    """
    Get formatted date context string for LLM prompts
    """
    info = get_current_date_info()
    return f"""
CURRENT DATE CONTEXT:
- Today: {info['today_formatted']} ({info['today']})
- Tomorrow: {info['tomorrow_formatted']} ({info['tomorrow']})
- Current day of week: {info['current_weekday']}
- Current year: {info['current_year']}

When users mention relative dates:
- "tomorrow" = {info['tomorrow']}
- "next Monday" = {info['next_monday']}
- Always calculate from today's date: {info['today']}
"""
