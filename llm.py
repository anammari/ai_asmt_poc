# llm.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_GEMINI_FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")

# dialect -> (dialect rule for the prompt, dialect idiom label)
_DIALECT_RULES = {
    "Arabic": ("Standard Arabic (no dialect)", "Standard Arabic"),
    "Syrian": ("Syrian/Levantine Arabic dialect", "Syrian/Levantine"),
    "Egyptian": ("Egyptian Arabic dialect", "Egyptian"),
}


def _redact_api_key(value: str) -> str:
    if not value:
        return value
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return value.replace(key, "***REDACTED***")
    return value


def _is_placeholder_key(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return not normalized or "your_gemini_api_key_here" in normalized


def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts") or []
    for part in parts:
        text = part.get("text")
        if text:
            return text
    return ""

def _local_fallback_script(user_prompt: str) -> str:
    topic = user_prompt if user_prompt else "Just relax and breathe."
    return f"[pause:2s] {topic}"

def _is_transient_gemini_error(message: str) -> bool:
    lowered = (message or "").lower()
    return (
        "http 429" in lowered
        or "http 500" in lowered
        or "http 503" in lowered
        or "request failed" in lowered
        or "timed out" in lowered
        or "api unavailable" in lowered
        or "unavailable after retries" in lowered
    )

def _is_transient_status(status_code: int) -> bool:
    return status_code in (429, 500, 503)

def _normalize_model_name(model: str) -> str:
    normalized = (model or "").strip()
    if normalized.startswith("models/"):
        return normalized[len("models/") :]
    return normalized

def _get_gemini_candidate_models(primary_model: str) -> list[str]:
    raw_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
    configured_fallbacks = [_normalize_model_name(m) for m in raw_fallbacks.split(",") if m.strip()]
    all_candidates = [_normalize_model_name(primary_model), *configured_fallbacks, *_DEFAULT_GEMINI_FALLBACK_MODELS]

    deduped: list[str] = []
    for model in all_candidates:
        if model not in deduped:
            deduped.append(model)
    return deduped


def validate_gemini_api_key(api_key: str | None = None, model: str | None = None, timeout: int = 20) -> dict:
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    target_model = _normalize_model_name(model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))

    if _is_placeholder_key(key):
        return {
            "ok": False,
            "reason": "missing_or_placeholder",
            "needs_new_key": True,
            "status_code": None,
            "message": "Gemini API key is missing or still using placeholder text.",
        }

    url = f"{_GEMINI_BASE_URL}/{target_model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": "Respond with exactly: ok"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8},
    }

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "reason": "network_error",
            "needs_new_key": False,
            "status_code": None,
            "message": _redact_api_key(str(exc)),
        }

    if resp.status_code == 200:
        return {
            "ok": True,
            "reason": "ok",
            "needs_new_key": False,
            "status_code": 200,
            "message": f"Gemini key is valid for model '{target_model}'.",
        }
    if resp.status_code in (401, 403):
        return {
            "ok": False,
            "reason": "invalid_key_or_permission",
            "needs_new_key": True,
            "status_code": resp.status_code,
            "message": _redact_api_key(resp.text),
        }
    if resp.status_code == 429:
        return {
            "ok": False,
            "reason": "quota_or_rate_limited",
            "needs_new_key": False,
            "status_code": 429,
            "message": _redact_api_key(resp.text),
        }
    if resp.status_code in (500, 503):
        return {
            "ok": False,
            "reason": "provider_unavailable",
            "needs_new_key": False,
            "status_code": resp.status_code,
            "message": _redact_api_key(resp.text),
        }
    if resp.status_code == 404:
        return {
            "ok": False,
            "reason": "model_unavailable",
            "needs_new_key": False,
            "status_code": 404,
            "message": _redact_api_key(resp.text),
        }
    return {
        "ok": False,
        "reason": "unexpected_error",
        "needs_new_key": False,
        "status_code": resp.status_code,
        "message": _redact_api_key(resp.text),
    }


def rewrite_script(
    context_text: str,
    user_prompt: str = "",
    provider: str = None,
    target_minutes: int = 1,
    vocal_tone: str = "Soft Spoken",
    user_info: str = "",
    language: str = "English",
    dialect: str = "Arabic",
) -> str:
    is_arabic = "arabic" in language.strip().lower()
    if provider is None:
        provider = os.getenv(
            "ARABIC_LLM_PROVIDER" if is_arabic else "ENGLISH_LLM_PROVIDER",
            "ollama",
        )
    target_word_count = target_minutes * 85

    system_msg = (
        "You are an expert ASMR scriptwriter generating scripts for an automated text-to-speech engine.\n"
        f"Your goal is write a script that takes exactly {target_minutes} minutes to speak slowly (strictly aim for ~{target_word_count} words).\n"
        f"The REQUIRED vocal tone is: {vocal_tone.upper()}.\n"
        f"Language rule: Write the spoken script in {language} only.\n"
    )

    if vocal_tone == "Whispering":
        system_msg += "Write using words that emphasize sibilance (s, sh, f, th, h) and highly intimate, breathy pacing.\n"

    system_msg += (
        "CRITICAL RULES:\n"
        "1. ONLY output the spoken script. NO titles, NO introductions, NO concluding remarks.\n"
        "2. NO meta-text, NO asterisks, NO markdown formatting (like **bold** or *italics*). Do NOT use quotes.\n"
        "3. Insert [pause] or [pause:2s] (or up to [pause:4s]) frequently to dictate pacing.\n"
        "4. NEVER use elongated words like 'shhhhh', 'sssss', or 'hmmmm'. The TTS engine will spell them out letter-by-letter. Use standard words only.\n"
        "5. You MUST stay strictly relevant to the provided Background Context and User Instructions.\n"
    )

    if is_arabic:
        dialect_rule, dialect_idiom = _DIALECT_RULES.get(dialect, _DIALECT_RULES["Arabic"])
        system_msg += (
            f"Language rule: Write the spoken script in Arabic only using {dialect_rule}.\n"
            f"DIALECT FIDELITY (CRITICAL): The ENTIRE script — every sentence of BOTH the narration and the dialogue — must be written in {dialect_rule}. Do NOT write narration in Standard/Formal Arabic while only the dialogue uses the dialect. Sustain the requested dialect consistently from the first word to the last, using {dialect_idiom} vocabulary, word order, and everyday speech patterns throughout.\n"
            "6. STRUCTURE: Divide the script into 5-6 paragraphs of roughly equal length, separated by blank lines.\n"
            "7. TIMING: The script MUST reach the target word count — do NOT finish early. Expand with sensory detail, dialogue, and scene-setting to reach the target.\n"
            "8. RICHNESS: Use vivid sensory language (sights, sounds, smells, textures). Include dialogue and a clear narrative arc: introduction, discovery, encounter, lesson, resolution. Use natural "
            f"{dialect_idiom} idioms and expressions to make the story feel authentic.\n"
            "9. VARY PAUSE DURATIONS: Use [pause], [pause:2s], [pause:3s], and [pause:4s] throughout — not just the default [pause]. This creates natural pacing.\n"
        )

    if user_info.strip():
        if is_arabic:
            system_msg += "10. PERSONAL ATTENTION: You have been provided with the User's Personal Info. Address the listener directly and by name. Weave the provided personal details (e.g. name, age, family, work, hobbies) naturally and prominently throughout the script — not just once — so the listener feels personally addressed. If the info mentions family or loved ones, frame the story as one told to them.\n"
        else:
            system_msg += "6. Personal Attention: You have been provided with the User's Personal Info. You MUST seamlessly and naturally weave these details (like their name, age, or hobbies) into the ASMR script to provide a deeply personal, comforting experience.\n"

    user_msg = f"User Instructions / Topic: {user_prompt if user_prompt else 'Create a general relaxing experience.'}\n\n"

    if user_info.strip():
        user_msg += f"User's Personal Info for Personal Attention:\n{user_info.strip()}\n\n"

    if context_text.strip():
        user_msg += f"Background Context Information:\n{context_text.strip()}"

    if provider == "ollama":
        ollama_model = os.getenv(
            "ARABIC_OLLAMA_MODEL" if is_arabic else "ENGLISH_OLLAMA_MODEL",
            "arabic-asmr-syria:latest" if is_arabic else "ministral-3:8b",
        )
        return _call_ollama(system_msg, user_msg, model=ollama_model)
    elif provider == "openai":
        try:
            return _call_openai(system_msg, user_msg)
        except NotImplementedError:
            raise
    elif provider == "openrouter":
        if is_arabic:
            return _local_fallback_script(user_prompt)
        try:
            return _call_openrouter(system_msg, user_msg)
        except RuntimeError:
            try:
                return _call_ollama(system_msg, user_msg)
            except RuntimeError:
                return _local_fallback_script(user_prompt)
    elif provider == "gemini":
        try:
            return _call_gemini(system_msg, user_msg)
        except RuntimeError as gemini_error:
            if _is_transient_gemini_error(str(gemini_error)):
                try:
                    return _call_ollama(system_msg, user_msg)
                except RuntimeError:
                    return _local_fallback_script(user_prompt)
            raise

    return _local_fallback_script(user_prompt)

def _call_ollama(system_msg: str, user_msg: str, model: str | None = None) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "ministral-3:8b")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }
    
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

def _call_openrouter(system_msg: str, user_msg: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or _is_placeholder_key(api_key):
        raise RuntimeError("OPENROUTER_API_KEY is missing or is a placeholder.")
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free").strip()
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            if _is_transient_status(resp.status_code):
                last_err = RuntimeError(f"OpenRouter HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter LLM call failed after retries. Last error: {last_err}")

def _call_openai(system_msg: str, user_msg: str) -> str:
    raise NotImplementedError(
        "OpenAI provider is not yet implemented. "
        "Set ARABIC_LLM_PROVIDER or ENGLISH_LLM_PROVIDER in .env."
    )

def _call_gemini(system_msg: str, user_msg: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    primary_model = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    if _is_placeholder_key(api_key):
        raise RuntimeError("Gemini API key is missing or placeholder. Update GEMINI_API_KEY in .env.")

    payload = {
        "systemInstruction": {"parts": [{"text": system_msg}]},
        "contents": [{"parts": [{"text": user_msg}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.95,
            "maxOutputTokens": 1200,
        },
    }

    candidate_models = _get_gemini_candidate_models(primary_model)
    transient_errors: list[str] = []
    fallback_model_errors: list[str] = []
    max_attempts = 4

    for model in candidate_models:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"

        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=90,
                )
            except requests.RequestException as exc:
                if attempt < max_attempts - 1:
                    time.sleep(2**attempt)
                    continue
                transient_errors.append(f"Model='{model}' network error: {_redact_api_key(str(exc))}")
                break

            if resp.status_code == 200:
                data = resp.json()
                text = _extract_gemini_text(data)
                if text.strip():
                    return text.strip()
                raise RuntimeError(f"Gemini response did not include generated text for model '{model}'.")

            if _is_transient_status(resp.status_code) and attempt < max_attempts - 1:
                time.sleep(2**attempt)
                continue

            safe_body = _redact_api_key(resp.text)
            if _is_transient_status(resp.status_code):
                transient_errors.append(
                    f"Model='{model}' HTTP {resp.status_code}: {safe_body[:220]}"
                )
                break

            if model != _normalize_model_name(primary_model):
                fallback_model_errors.append(
                    f"Model='{model}' HTTP {resp.status_code}: {safe_body[:220]}"
                )
                break

            raise RuntimeError(
                f"Gemini API call failed with HTTP {resp.status_code}. "
                f"Model='{model}'. Response='{safe_body[:300]}'"
            )

    details_pool = transient_errors + fallback_model_errors
    details = " | ".join(details_pool[-4:]) if details_pool else "No additional details."
    raise RuntimeError(
        "Gemini API unavailable after retries across candidate models. "
        f"Tried: {', '.join(candidate_models)}. Details: {details}"
    )