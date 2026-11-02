# 🏥 CityCare Clinic – AI Voice Agent

An AI-powered clinic voice assistant built using **Python, LiveKit Agents, and FastAPI**. CityCare enables patients to interact with a virtual clinic receptionist through real-time voice conversations.

The agent helps patients with appointment booking, appointment checking, rescheduling, and cancellation while providing clinic information through an AI-powered conversational interface.

## 🚀 Key Features

- **Real-Time Voice Conversations:** Natural voice interactions between patients and the AI assistant.
- **Appointment Booking:** Helps patients schedule clinic appointments.
- **Appointment Management:** Supports checking, rescheduling, and cancelling appointments.
- **Caller Identity Verification:** Collects and verifies caller details before accessing private appointment information.
- **Clinic Information:** Answers questions about clinic services, working hours, doctors, and facilities.
- **RAG Integration:** Retrieves relevant information from the clinic knowledge base.
- **Database Persistence:** Uses PostgreSQL to store appointment records.
- **Google Calendar Integration:** Synchronizes confirmed appointments with Google Calendar.
- **LLM Fallback:** Supports primary and fallback language models for improved reliability.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Voice Agent Framework | LiveKit Agents |
| Speech-to-Text (STT) | Deepgram Nova-3 |
| Text-to-Speech (TTS) | Cartesia Sonic-3 |
| Large Language Models | Groq / Gemini |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Calendar Integration | Google Calendar |
| AI Knowledge Retrieval | RAG |
| Vector Database | ChromaDB / InMemoryVectorStore |

## 🏗️ Project Architecture

The system connects patients to a LiveKit voice agent, which processes spoken requests using speech recognition, an LLM, and speech synthesis.

The FastAPI backend handles appointment operations, while PostgreSQL stores records and Google Calendar synchronizes appointments.

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/saiyarramsetti55-ship-it/livekit_voice_agent.git
cd livekit_voice_agent
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env.local` file and configure the required API keys and service credentials.

Never commit API keys, passwords, or private credentials.

### 4. Start the Backend

```bash
uv run uvicorn backend.main:app --port 8000 --reload
```

### 5. Start the LiveKit Voice Agent

```bash
lk agent dev
```

## 🔐 Security

- Caller verification before accessing private appointment details.
- API credentials stored in environment variables.
- No medical diagnosis or medication dosage recommendations.
- Appointment operations should be confirmed before execution.

## 🎯 Project Goal

To build a reliable, low-latency AI voice receptionist that improves clinic accessibility and automates routine patient interactions through natural voice communication.

---

**Built with Python, LiveKit, FastAPI, and Generative AI.**