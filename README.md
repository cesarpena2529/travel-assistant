# 🏋 Travel Assistant

An intelligent, real-time travel assistant built with FastAPI and OpenAI, leveraging Retrieval-Augmented Generation (RAG) and multiple public APIs. Designed to provide rich, location-based recommendations with memory of past conversations.

---

## 🔍 Why This Project?

When planning a trip or navigating a new city, travelers often have fragmented information: weather from one app, attractions from another, and no memory of preferences. This assistant aims to unify that into a seamless experience by combining:

- Real-time data from public APIs
- Long-form memory using vector search
- Conversational planning via GPT

It’s built for:

- ✈️ Curious travelers
- 🧠 AI/ML enthusiasts
- 🔧 Engineers looking to learn RAG-based architectures

---

## 🌟 Features

- 🔎 **Location-aware Q&A**: Ask anything about your current or future travel destination
- ☀️ **Weather + Attractions**: Integrated real-time weather and top points of interest
- 🧠 **Memory & Personalization**: Powered by vector search with OpenAI embeddings
- 🛠 **Local + Cloud Deployment**: Docker Compose or EC2 with domain and TLS

---

## 🧱 Architecture Overview

### 📡 Data Pipeline

- **GeoDB API**: Resolves city/country details
- **OpenTripMap API**: Gets attractions and metadata
- **VisualCrossing API**: Historical and real-time weather data
- Initial articles were scraped and processed from `enwikivoyage-latest-pages-articles`, an open-source dataset from Wikimedia.
- **(Optional)**: Considered Amadeus API for flights

### 🧠 Knowledge Base Construction

- Articles and attraction descriptions are **chunked** and embedded
- Uses `text-embedding-3-small` from OpenAI
- Stored in **Qdrant** (local vector DB via Docker)
- Indexed by `location_id` and relevant metadata

---

## 🧪 Optional: Build Your Own Knowledge Base

You can skip this if you're using the pre-embedded KB.

### 📋 Prerequisites

- Python 3.8+
- `.env` with OpenAI API key and relevant config



### ⚙️ Steps

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:

```bash
cp .env.example .env
# Fill in your keys (OPENAI_API_KEY, etc.)
```

3. **Process knowledge base**:

```bash
python process_kb_for_vector_search.py
```

This creates:

- `kb_flattened.json` — embedded text chunks
- `metadata_lookup.json` — corresponding metadata like weather, attractions, etc.

4. **Qdrant will load the embedded vectors automatically at runtime.**

---

## 🚀 Backend Logic (FastAPI)

- Main logic lives in `claude_test_server.py` (in project root)
- Accepts queries via `/chat` endpoint
- Retrieves top-k relevant chunks using `qdrant_service.py`
- Assembles prompt + context, sends to Claude 3 Haiku (or GPT fallback)
- Returns answer + references + metadata

### 🖥️ Frontend

- (Optional) React frontend under development (e.g. map, cards, chat UI)
- Currently accessible via HTTP or Postman/cURL

---

## 🐳 Deployment (EC2-Based)

### ⚖️ Orchestration

- **Docker Compose**: Two services
  - `travel-assistant` (FastAPI backend)
  - `qdrant` (vector database)

### 🔐 HTTPS & Domain

- **Nginx** reverse proxy
- **Certbot** with Let’s Encrypt for TLS
- **Custom Domain**: `travel-assistant.site`

### ⚙️ EC2 Setup

- Hosted on t3a.medium instance
- Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)
- `.env` loaded with systemd unit for auto-restart

### 🔍 Monitoring Tools

- `htop` for CPU/memory
- `iftop` for network traffic

---

## 📁 Project Structure

```
travel-assistant/
├── backend/
│   ├── qdrant_service.py           # Qdrant client wrapper used in claude_test_server.py
│   ├── config.py, models.py        # Shared utilities
│   ├── (legacy) main.py, main_qdrant.py, search_service.py, etc.
├── claude_test_server.py           # Main FastAPI app used in Dockerfile.prod
├── data/
│   └── kb_flattened.json, metadata.json, etc.
├── Dockerfile.prod
├── Dockerfile.qdrant
├── docker-compose.yml
├── nginx.conf
├── certs/                          # Auto-managed by Certbot
├── requirements.txt
└── README.md
```

---

## 🧪 Run Locally (via Docker Compose)

```bash
# Step 1: Clone and navigate
$ git clone https://github.com/yourname/travel-assistant.git
$ cd travel-assistant

# Step 2: Add .env file
OPENAI_API_KEY=sk-xxx
AWS_ACCESS_KEY_ID=...
...

# Step 3: Build and launch
$ docker-compose up --build

# Access endpoints:
# http://localhost:8000/chat
# http://localhost:8000/health
```

---

## 🔎 Example Prompt + Response

> **User:** What's the weather and best places to visit near Cusco?

```json
{
  "weather": {
    "July": { "temp": 63, "humidity": 55 }
  },
  "attractions": [
    { "name": "Sacsayhuamán", "kinds": "historic", "rate": 3 },
    { "name": "Qorikancha", "kinds": "museums", "rate": 3 }
  ],
  "summary": "In July, Cusco is mild and dry. Be sure to visit Sacsayhuamán and Qorikancha for history and culture."
}
```

---

## 🔐 Environment Variables

| Variable                                      | Description                        |
| --------------------------------------------- | ---------------------------------- |
| `OPENAI_API_KEY`                              | For embeddings and chat completion |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | For future expansion               |
| `QDRANT_HOST`                                 | Set to `qdrant` (Docker network)   |

---

## 📀 Architecture Diagram

> Coming soon: full EC2 + Docker + TLS + Nginx deployment diagram

---

## 📣 Credits & Notes

- Built with ❤️ by Cesar Pena
- APIs used: OpenTripMap, GeoDB, VisualCrossing (Weather Data), OpenAI
- Vector DB: Qdrant (self-hosted)
- Designed for portfolio and educational use

---

## 🧼 Next Steps

- ECS + Fargate deployment variant
- React-based frontend UI
- Chat memory persistence
- Vacation package planner

---

## 📜 License

MIT License (for educational and demo use)

---

## 🥂 Cheers!

This project is a culmination of engineering, travel curiosity, and long nights debugging Nginx. May it inspire your next adventure — virtual or real.

