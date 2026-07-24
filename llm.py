# llm.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def rewrite_script(context_text: str, user_prompt: str = "", provider: str = None, target_minutes: int = 1, vocal_tone: str = "Soft Spoken") -> str:
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    
    # ASMR speech is slow, roughly 80-90 words per minute.
    target_word_count = target_minutes * 85 
    
    # Strict system prompt removing noise and focusing on the user prompt
    system_msg = (
        "You are an expert ASMR scriptwriter generating scripts for an automated text-to-speech engine.\n"
        f"Your goal is to write a script that takes exactly {target_minutes} minutes to speak slowly (strictly aim for ~{target_word_count} words).\n"
        f"The REQUIRED vocal tone is: {vocal_tone.upper()}.\n"
        "If the tone is WHISPERING, write using words that emphasize sibilance (s, sh, f, th, h) and highly intimate, breathy pacing.\n"
        "CRITICAL RULES:\n"
        "1. ONLY output the spoken script. NO titles, NO introductions, NO concluding remarks.\n"
        "2. NO meta-text, NO asterisks, NO markdown formatting (like **bold** or *italics*). Do NOT use quotes.\n"
        "3. Insert [pause] or [pause:2s] (or up to [pause:4s]) frequently to dictate pacing.\n"
        "4. The user's specific instructions MUST be the main subject of your script.\n"
    )
    
    user_msg = f"User Instructions / Topic: {user_prompt if user_prompt else 'Create a general relaxing experience.'}\n\n"
    if context_text:
        user_msg += f"Background Context Information:\n{context_text}"
        
    if provider == "ollama":
        return _call_ollama(system_msg, user_msg)
    elif provider == "openai":
        return _call_openai(system_msg, user_msg)
    elif provider == "gemini":
        return _call_gemini(system_msg, user_msg)
        
    return f"[pause:2s] {user_prompt if user_prompt else 'Just relax and breathe.'}"

def _call_ollama(system_msg: str, user_msg: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "ministral-3:8b")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }
    
    # Fallback through common Ollama endpoints to avoid 404s
    endpoints = [f"{base_url}/api/chat", f"{base_url}/v1/chat/completions"]
    
    last_err = None
    for ep in endpoints:
        try:
            resp = requests.post(ep, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                if "message" in data:
                    return data["message"]["content"]
                elif "choices" in data:
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            continue
            
    raise RuntimeError(f"Ollama LLM call failed on all endpoints. Last error: {last_err}. Check if model '{model}' is pulled.")

def _call_openai(system_msg: str, user_msg: str) -> str:
    # Placeholder for OpenAI native implementation
    return "OpenAI routing placeholder."

def _call_gemini(system_msg: str, user_msg: str) -> str:
    # Placeholder for Gemini native implementation
    return "Gemini routing placeholder."