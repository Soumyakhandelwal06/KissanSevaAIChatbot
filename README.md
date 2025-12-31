# KissanSeva AI Chatbot 🌾🤖

KissanSeva is an intelligent AI-powered chatbot designed to assist farmers by providing instant advisory services, answering agricultural queries, and offering solutions for crop management, pest control, and more. The system leverages advanced NLP models to understand and respond to farmer queries effectively.

## 🚀 Features

- **AI-Driven Chat Interface**: Interactive chatbot for real-time assistance.
- **Crop Advisory**: Expert advice on crop health, sowing, and harvesting.
- **Pest & Disease Management**: Identification and solutions for common plant diseases.
- **Multilingual Support**: (Intended/Implemented) support for local languages using translation services.
- **Robust Backend**: Powered by FastAPI and advanced ML models (TensorFlow, Transformers).

## 🛠️ Tech Stack

### Frontend
- **Framework**: React.js
- **Build Tool**: Vite
- **Styling**: CSS / Styled Components (inferred)
- **State Management**: React Hooks

### Backend
- **Framework**: FastAPI (Python)
- **ML/AI Libraries**: 
  - TensorFlow / Keras
  - HuggingFace Transformers
  - Sentence Transformers
  - Scikit-learn
- **Database**: MongoDB
- **Server**: Uvicorn

## 📂 Project Structure

```
KissanSeva-AI-Ai-farmers-chatbot-/
├── backend/            # FastAPI application and ML models
│   ├── models/         # Trained ML models
│   ├── main.py         # Application entry point
│   ├── requirements.txt # Python dependencies
│   └── ...
├── frontend/           # React frontend application
│   ├── src/            # Source code
│   ├── package.json    # Node dependencies
│   └── ...
└── setup_backend.sh    # Script to set up backend environment
```

## ⚙️ Installation & Setup

### Prerequisites
- Node.js & npm
- Python 3.9+
- MongoDB (running locally or cloud URI)

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables:
Copy `.env.example` to `.env` (if available) or create a `.env` file with necessary configurations (API keys, DB URL).

Run the server:
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
The application will be accessible via the link provided in the terminal (usually `http://localhost:5173`).

## 🐳 Docker Support

You can also run the backend using Docker:

```bash
cd backend
docker-compose up -d --build
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## 📄 License

[MIT License](LICENSE) (Assuming MIT, or verify)