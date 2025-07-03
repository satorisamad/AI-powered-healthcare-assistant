# AI-Powered Healthcare Voice Assistant

A comprehensive voice-enabled healthcare assistant that handles appointment scheduling, insurance verification, and clinic inquiries through natural phone conversations. Built with OpenAI GPT-4o-mini, Twilio, Deepgram, and Cartesia for real-time voice interactions.

## 🎯 Features

- **Real-time Voice Conversations**: Sub-1-second response times via WebSocket streaming
- **Appointment Scheduling**: Integrates with Google Calendar for actual bookings
- **Insurance Verification**: Checks accepted providers and coverage
- **Clinic Information**: Answers FAQs about hours, location, services, and policies
- **Emergency Detection**: Automatically detects and responds to medical emergencies
- **Natural Conversation Flow**: Maintains context and handles complex multi-turn dialogues

## 🏗️ Architecture

```
Phone Call → Twilio → Deepgram STT → OpenAI GPT → Backend Tools → Cartesia TTS → Voice Response
```

### Key Components
- **Voice Processing**: Twilio (telephony) + Deepgram (STT) + Cartesia (TTS)
- **AI Agent**: OpenAI GPT-4o-mini with function calling
- **Backend Services**: Appointment scheduler, insurance verifier, FAQ system
- **Session Management**: Multi-user conversation handling
- **Response Caching**: Optimized for common queries

## 📋 Prerequisites

- **Python 3.8+**
- **API Keys Required**:
  - OpenAI API key
  - Cartesia API key
  - Deepgram API key (optional, for advanced features)
  - Twilio account (for phone integration)
- **Google Calendar API** (for appointment booking)

## 🚀 Quick Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd TakeHomeVoiceAssistant
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the root directory:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here

# Optional (for advanced features)
DEEPGRAM_API_KEY=your_deepgram_api_key_here
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
```

### 3. Set Up Google Calendar (Optional)

For real appointment booking:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Calendar API
4. Create credentials (OAuth 2.0)
5. Download `credentials.json` to project root
6. Run the calendar setup:

```bash
python -c "from app.backend.calendar_toolkit import calendar_tools; print('Calendar setup complete')"
```

### 4. Test the System

#### Option A: Web Interface (Recommended for Testing)
```bash
streamlit run streamlit_chatbot.py
```
Open http://localhost:8501 in your browser

#### Option B: Voice Server (For Phone Integration)
```bash
uvicorn app.voice.twilio_deepgram_router:app --reload --port 8000
```

#### Option C: Direct Testing
```bash
python main.py
```

## 🧪 Testing Scenarios

Use the Streamlit interface to test these scenarios:

### Basic Interactions
- **Greeting**: "Hello"
- **Clinic Hours**: "What are your hours?"
- **Location**: "Where are you located?"
- **Services**: "What services do you offer?"

### Appointment Scheduling
- **Simple Booking**: "I need an appointment tomorrow at 2 PM"
- **Complex Booking**: "Can I schedule a check-up with Dr. Smith next Tuesday morning?"
- **Availability Check**: "What times are available this week?"

### Insurance Verification
- **Coverage Check**: "Do you accept Blue Cross Blue Shield?"
- **Plan Verification**: "I have Aetna insurance, am I covered?"

### Edge Cases
- **Emergency**: "I'm having chest pain" (tests emergency detection)
- **Unclear Input**: "Um, uh, maybe..." (tests noise handling)
- **Off-topic**: "What's the weather?" (tests redirection)

## 📞 Phone Integration Setup

### Twilio Configuration

1. **Create Twilio Account**: Sign up at [twilio.com](https://twilio.com)
2. **Get Phone Number**: Purchase a Twilio phone number
3. **Configure Webhook**: Set webhook URL to your server:
   ```
   https://your-domain.com/voice
   ```
4. **Test Call**: Call your Twilio number to test the voice assistant

### Ngrok for Local Testing
```bash
# Install ngrok
npm install -g ngrok

# Expose local server
ngrok http 8000

# Use the ngrok URL in Twilio webhook configuration
```

## 🔧 Configuration Options

### Voice Optimization Settings
Edit `app/ai/llm_client.py` to adjust:
- **Response Length**: `max_tokens` parameter
- **Voice Speed**: TTS speaking rate
- **Conversation Style**: Temperature and top_p values

### Data Customization
- **FAQ Answers**: Edit `data/faq.json`
- **Insurance Plans**: Edit `data/insurance_plans.json`
- **Available Slots**: Edit `data/slots.json`

## 📊 Monitoring and Logs

The system provides detailed logging for debugging:

```bash
# View real-time logs
tail -f app.log

# Check specific components
grep "[Agent]" app.log      # AI agent decisions
grep "[Voice]" app.log      # Voice processing
grep "[Scheduler]" app.log  # Appointment booking
```

## 🚨 Troubleshooting

### Common Issues

**"Cannot connect to FastAPI server"**
- Ensure the server is running: `uvicorn app.voice.twilio_deepgram_router:app --reload --port 8000`
- Check if port 8000 is available

**"Calendar tools not available"**
- Verify `credentials.json` is in the root directory
- Run Google OAuth flow: `python -c "from app.backend.calendar_toolkit import calendar_tools"`

**"API key not found"**
- Check `.env` file exists and contains required keys
- Verify API keys are valid and have sufficient credits

**Voice quality issues**
- Check internet connection stability
- Verify Twilio webhook URL is accessible
- Test with different phone numbers/carriers

### Performance Optimization

For production deployment:
- Use Redis for session storage
- Implement connection pooling
- Add load balancing
- Enable response compression
- Set up monitoring and alerting

## 📁 Project Structure

```
TakeHomeVoiceAssistant/
├── app/
│   ├── ai/                 # AI agent and LLM client
│   ├── backend/            # Business logic (scheduler, insurance, FAQ)
│   ├── utils/              # Utility functions
│   └── voice/              # Voice processing (STT, TTS, WebSocket)
├── data/                   # JSON data files
├── audio_files/            # Generated audio files
├── credentials.json        # Google API credentials
├── token.json             # Google OAuth token
├── streamlit_chatbot.py   # Web testing interface
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
└── SYSTEM_DESIGN.md       # Detailed architecture documentation
```
