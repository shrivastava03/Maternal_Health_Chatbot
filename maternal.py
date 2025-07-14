import gradio as gr
import google.generativeai as genai
from transformers import pipeline
from functools import lru_cache
import re
import os
from typing import List, Tuple

# Configure Gemini
GEMINI_API_KEY = "_"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_available = True
except Exception as e:
    print(f"Gemini error: {e}")
    gemini_available = False

# Load Hugging Face emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Response templates
RESPONSE_TEMPLATES = {
    'scared': [
        "It's completely normal to feel {mood} during pregnancy. Many expectant mothers experience similar feelings. Try taking some deep breaths - inhale for 4 counts, hold for 4, exhale for 6.",
        "I hear that you're feeling {mood}. Pregnancy can bring up many emotions. Remember, you're stronger than you think. Would you like me to suggest some relaxation techniques?",
        "Feeling {mood} is understandable. Have you tried talking to your healthcare provider about these concerns? They can offer professional reassurance."
    ],
    'confused': [
        "Let me clarify that for you. {info} If you need more details, your prenatal classes or healthcare provider can offer additional guidance.",
        "That's a common question. The key points are: {info} Does this help explain things better?",
        "I understand the confusion. Here's what you should know: {info} Bookmark this information for future reference."
    ],
    'frustrated': [
        "Pregnancy discomforts can definitely be frustrating. Try {suggestion} to help alleviate this. Many moms find this helpful.",
        "I hear your frustration. These feelings are valid. Remember to be gentle with yourself during this time.",
        "This sounds challenging. Have you considered {suggestion}? It might help ease the frustration you're feeling."
    ],
    'medical': [
        "For this {symptom}, I recommend contacting your healthcare provider for proper evaluation. It's always best to get medical advice directly.",
        "This sounds like something to discuss with your doctor. Would you like help finding nearby clinics?",
        "Pregnancy symptoms can vary, but for {symptom}, professional medical advice is important."
    ],
    'positive': [
        "That's wonderful to hear! Enjoy these positive moments in your pregnancy journey.",
        "I'm so glad you're feeling {mood}! This is a special time to cherish.",
        "What a beautiful sentiment! Savor these happy pregnancy moments."
    ],
    'neutral': [
        "I'm here to support you throughout your pregnancy. Feel free to share anything on your mind.",
        "Thank you for sharing. How are you feeling about your pregnancy today?",
        "I'm listening. Pregnancy brings many experiences - would you like to talk more?"
    ]
}

MOOD_CONFIG = {
    'scared': {
        'color': '#ffdddd',
        'keywords': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'fear'],
        'info': {
            'default': 'Pregnancy can bring many new feelings and concerns',
            'pain': 'Some discomfort is normal, but persistent pain should be checked',
            'baby': 'Your baby is well protected in the womb'
        }
    },
    'confused': {
        'color': '#fff4dd',
        'keywords': ['confused', 'unsure', 'understand', 'question', 'what if', 'how to'],
        'info': {
            'default': 'Pregnancy involves many changes that can be confusing',
            'nutrition': 'Focus on balanced meals with folate, iron and calcium',
            'exercise': 'Moderate exercise like walking is generally safe'
        }
    },
    'frustrated': {
        'color': '#ffe5dd',
        'keywords': ['frustrated', 'angry', 'annoyed', 'irritated', 'fed up'],
        'suggestions': {
            'default': 'gentle prenatal yoga or meditation',
            'sleep': 'using pregnancy pillows for better support',
            'discomfort': 'warm baths or maternity support belts'
        }
    },
    'medical': {
        'color': '#ffebee',
        'keywords': ['pain', 'bleeding', 'contraction', 'pressure', 'symptom', 'doctor'],
        'symptoms': {
            'default': 'symptoms',
            'severe': 'severe symptoms',
            'bleeding': 'any bleeding'
        }
    }
}

@lru_cache(maxsize=50)
def detect_emotion(text: str) -> str:
    result = emotion_classifier(text)[0]
    top_emotion = max(result, key=lambda x: x['score'])['label']
    if top_emotion in ['fear']: return 'scared'
    elif top_emotion in ['anger']: return 'frustrated'
    elif top_emotion in ['sadness', 'confusion']: return 'confused'
    elif top_emotion in ['joy']: return 'positive'
    return 'neutral'

def analyze_mood(text: str) -> Tuple[str, str]:
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in MOOD_CONFIG['medical']['keywords']):
        symptom = next((s for s in MOOD_CONFIG['medical']['keywords'] if s in text_lower), 'default')
        return 'medical', symptom
    try:
        mood = detect_emotion(text)
    except:
        mood = 'neutral'
    context = 'default'
    return mood, context

@lru_cache(maxsize=50)
def get_gemini_response(prompt: str) -> str:
    return model.generate_content(prompt).text

def generate_response(message: str, chat_history: List[Tuple[str, str]] = []) -> str:
    mood, context = analyze_mood(message)
    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history[-3:]])
    prompt = f"""
As a maternal health assistant, provide ONE complete, supportive response to this {mood} message.
Be emotionally appropriate and informative.

Your job include:
 -Always take care of tone
 -Never let patient panic and talk supportively

Conversation:
{history_text}

User: {message}
Assistant:
"""
    try:
        if gemini_available:
            return get_gemini_response(prompt)
    except Exception as e:
        print(f"Gemini error: {e}")

    # fallback to template
    templates = RESPONSE_TEMPLATES.get(mood, RESPONSE_TEMPLATES['neutral'])
    return templates[0].format(
        mood=mood,
        info=MOOD_CONFIG.get(mood, {}).get('info', {}).get(context, ''),
        suggestion=MOOD_CONFIG.get(mood, {}).get('suggestions', {}).get(context, ''),
        symptom=MOOD_CONFIG.get(mood, {}).get('symptoms', {}).get(context, '')
    )

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .chatbot { min-height: 500px; border-radius: 12px; }
    .mood-indicator {
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.9em;
        margin-top: 8px;
        display: inline-block;
    }
    .warning {
        background: #fff3e0;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
""") as demo:
    gr.Markdown("""
    # 🤰 Maternal Health Companion
    *Supportive, judgment-free pregnancy support*
    """)

    if not gemini_available:
        gr.Markdown("""
        <div class="warning">
        ⚠️ Running in limited mode. For full functionality, please configure your Gemini API key.
        </div>
        """)

    chatbot = gr.Chatbot(label="Conversation", bubble_full_width=False)
    msg = gr.Textbox(label="Share your thoughts", placeholder="How are you feeling today?")

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    mood_display = gr.HTML()

    def respond(message: str, chat_history: List[Tuple[str, str]]):
        if not message.strip():
            return chat_history, "", ""
        response = generate_response(message, chat_history)
        mood, _ = analyze_mood(message)
        mood_data = MOOD_CONFIG.get(mood, {'color': '#e3f2fd'})
        chat_history.append((message, response))
        mood_html = f"""
        <div class=\"mood-indicator\" style=\"background:{mood_data['color']}\">
            Detected: {mood.capitalize()}
        </div>
        """
        return chat_history, "", mood_html

    msg.submit(respond, [msg, chatbot], [chatbot, msg, mood_display])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg, mood_display])
    clear_btn.click(lambda: ([], "", ""), None, [chatbot, msg, mood_display])

if __name__ == "__main__":
    demo.launch()
