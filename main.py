#!/usr/bin/env python3
"""
Entry point for the AI-Powered Healthcare Voice Assistant.

This is the main entry point for the healthcare voice assistant application.
Currently serves as a placeholder for the main application logic.
"""


def main():
    """Main entry point for the healthcare voice assistant."""
    print("Welcome to the AI-Powered Healthcare Voice Assistant!")
    print("This is a placeholder main function.")
    print("To run the voice assistant server, use:")
    print("  uvicorn app.voice.twilio_deepgram_router:app --reload --port 8000")
    print("To run the Streamlit chatbot, use:")
    print("  streamlit run streamlit_chatbot.py")


if __name__ == "__main__":
    main()
