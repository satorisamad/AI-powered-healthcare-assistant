from langchain_google_community import CalendarToolkit

from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)
# Set up Google Calendar credentials (reuse your Gmail OAuth flow)
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)
calendar_resource = build_resource_service(credentials=credentials)
toolkit = CalendarToolkit(api_resource=calendar_resource)

calendar_tools = toolkit.get_tools()

# Example: calendar_tools will include functions for creating events, checking availability, etc.
