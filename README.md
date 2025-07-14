# 🤰 Maternal Health Companion Chatbot

An AI-powered chatbot designed to support pregnant individuals with empathetic, emotional, and informational guidance. It uses Google Gemini and Hugging Face models to detect mood and provide supportive responses tailored to the emotional context of the user.

## 💡 Features

- Detects user emotions using a fine-tuned Hugging Face emotion classifier
- Responds contextually with help from Google's Gemini API
- Gradio-based user interface with chatbot experience
- Dynamic mood detection and visualization
- Offline fallback to intelligent template-based responses

## 🧠 Technologies Used

- Python
- [Gradio](https://gradio.app/) – for the frontend interface
- [Google Gemini API](https://ai.google.dev/) – for intelligent conversational responses
- [Hugging Face Transformers](https://huggingface.co/) – for emotion detection
- `lru_cache` – for performance optimization

## 🏗️ Project Structure
1) ├── maternal_chatbot.py # Main chatbot code
2) ├── README.md # Project documentation
3) └── requirements.txt # Python dependencies

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/maternal-chatbot.git
   cd maternal-chatbot


2. Install dependencies:
3. pip install -r requirements.txt
4. Add your Gemini API key in the code (GEMINI_API_KEY = "YOUR_KEY
5. Run the chatbot: python maternal_chatbot.py


🔒 Disclaimer
This is not a replacement for professional medical advice. Always consult a doctor for serious health concerns.
