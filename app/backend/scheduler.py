# Appointment scheduling logic

from app.backend.calendar_toolkit import calendar_tools
from datetime import datetime
from typing import Optional, Dict
import logging
from pydantic import BaseModel, ValidationError, validator

def find_or_suggest_time(requested_date: str, requested_time: str, patient_name: str, search_days: int = 14) -> Dict:
    """
    Check if a slot is available or suggest the next available slot up to `search_days` ahead.
    Handles both absolute dates (YYYY-MM-DD) and relative dates (tomorrow, next week, etc.)
    """
    import datetime
    from app.utils.date_utils import parse_relative_date

    # Parse the date - handle both absolute and relative dates
    if requested_date.lower() in ['tomorrow', 'today'] or 'next' in requested_date.lower() or any(day in requested_date.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
        # It's a relative date, parse it
        parsed_date = parse_relative_date(requested_date)
        logging.info(f"[Scheduler] Parsed relative date '{requested_date}' to {parsed_date}")
    else:
        # It's already an absolute date
        parsed_date = requested_date

    logging.info(f"[Scheduler] Checking slot for {parsed_date} at {requested_time} for {patient_name}")
    # Use the search and create tools
    search_tool = None
    create_tool = None
    for tool in calendar_tools:
        if hasattr(tool, 'name') and 'search' in tool.name.lower():
            search_tool = tool
        if hasattr(tool, 'name') and 'create' in tool.name.lower():
            create_tool = tool
    if not search_tool or not create_tool:
        logging.error("Calendar tools for search/create not found.")
        return {"status": "none", "message": "Calendar tools not available"}

    # Parse input date/time using the parsed date
    dt = datetime.datetime.strptime(f"{parsed_date} {requested_time}", "%Y-%m-%d %H:%M")
    clinic_start_hour = 9
    clinic_end_hour = 18
    slot_found = False
    for day_offset in range(search_days):
        for hour in range(dt.hour, clinic_end_hour):
            start_dt = dt.replace(hour=hour, minute=0, second=0)
            end_dt = start_dt + datetime.timedelta(hours=1)
            start_datetime = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_datetime = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            calendars_info = '[{"id": "primary"}]'
            search_result = search_tool.run({
                "calendars_info": calendars_info,
                "min_datetime": start_datetime,
                "max_datetime": end_datetime,
                "max_results": 1
            })
            logging.info(f"[Scheduler] Calendar search result: {search_result}")
            # If no events found, slot is available
            if (
                (isinstance(search_result, dict) and not search_result.get('events')) or
                (isinstance(search_result, str) and 'No events found' in search_result) or
                (isinstance(search_result, list) and len(search_result) == 0)
            ):
                slot = {
                    "date": start_dt.strftime("%Y-%m-%d"),
                    "time": start_dt.strftime("%H:%M"),
                    "doctor": "Dr. Smith",
                    "patient": patient_name
                }
                # Create the appointment event
                event_data = {
                    "summary": f"Doctor Appointment - {patient_name}",
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                    "timezone": "America/New_York"
                }
                create_result = create_tool.run(event_data)
                logging.info(f"[Scheduler] Calendar create result: {create_result}")
                return {"status": "booked", "slot": slot, "event": create_result, "message": f"Appointment booked for {patient_name}"}
        # Move to next day, reset hour to clinic_start_hour
        dt = dt + datetime.timedelta(days=1)
        dt = dt.replace(hour=clinic_start_hour, minute=0, second=0)
    # If no slot found after search_days
    return {"status": "none", "message": f"No slots available in the next {search_days} days."}

def confirm_schedule(slot: Dict, patient_name: str) -> Dict:
    """
    Simulate confirming a schedule (would save to DB in real app).
    """
    logging.info(f"[Scheduler] Confirming schedule for {patient_name} at {slot}")
    return {"status": "confirmed", "slot": slot, "patient": patient_name}

class AppointmentRequest(BaseModel):
    patient_name: str
    date: str  # Expecting 'YYYY-MM-DD'
    time: str  # Expecting 'HH:MM'
    reason: str
    phone: str

    @validator('date')
    def validate_date(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

    @validator('time')
    def validate_time(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, '%H:%M')
        except ValueError:
            raise ValueError('Time must be in HH:MM format')
        return v

    @validator('phone')
    def validate_phone(cls, v):
        import re
        if not re.match(r'^\\+?1?\\d{9,15}$', v):
            raise ValueError('Phone number must be valid and include country code if possible')
        return v

def validate_appointment_fields(data: dict) -> dict:
    """
    Validate and parse appointment fields using Pydantic.
    Returns dict with 'valid': True/False and 'errors' or 'appointment'.
    """
    try:
        appointment = AppointmentRequest(**data)
        return {'valid': True, 'appointment': appointment}
    except ValidationError as e:
        return {'valid': False, 'errors': e.errors()}
