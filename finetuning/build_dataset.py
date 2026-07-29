# finetuning/build_dataset.py
"""
Builds the Arabic ASMR supervised fine-tuning (SFT) dataset.

Output: JSONL in chat format (one sample per line):
  {"messages": [
      {"role": "system",    "content": <ASMR scriptwriter system prompt>},
      {"role": "user",      "content": "User Instructions / Topic: <topic>"},
      {"role": "assistant", "content": <Arabic ASMR script with [pause] markers>},
  ]}

The system prompt mirrors the structure the app uses at inference time
(llm.rewrite_script with language="Arabic (العربية)"), so the fine-tuned
model learns the exact distribution the app will query.

Sources of samples:
  1. Built-in hand-written seed exemplars (always included, also used in
     --offline mode so the training pipeline can be smoke-tested).
  2. Gemini-augmented generation: reuses the app's hardened Gemini caller
     (llm._call_gemini with retries/fallbacks) over a topic bank.

Every candidate is validated (Arabic ratio, [pause] markers, no markdown,
word-count band, dedup) before it lands in the JSONL.

Usage:
  python finetuning/build_dataset.py --count 60      # needs GEMINI_API_KEY
  python finetuning/build_dataset.py --offline       # seeds only
"""
import argparse
import hashlib
import json
import os
import sys

# Allow `python finetuning/build_dataset.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "arabic_asmr_sft.jsonl")

# System prompt shape mirrors llm.rewrite_script's Arabic configuration.
SYSTEM_PROMPT = (
    "You are an expert ASMR scriptwriter generating scripts for an automated "
    "text-to-speech engine.\n"
    "Your goal is to write a script that takes exactly 2 minutes to speak "
    "slowly (strictly aim for ~170 words).\n"
    "The REQUIRED vocal tone is: WHISPERING.\n"
    "Language rule: Write the spoken script in Arabic (العربية) only.\n"
    "CRITICAL RULES:\n"
    "1. ONLY output the spoken script. NO titles, NO introductions, NO "
    "concluding remarks.\n"
    "2. NO meta-text, NO asterisks, NO markdown formatting. Do NOT use quotes.\n"
    "3. Insert [pause] or [pause:2s] (or up to [pause:4s]) frequently to "
    "dictate pacing.\n"
    "4. NEVER use elongated words. The TTS engine will spell them out "
    "letter-by-letter. Use standard words only.\n"
    "5. You MUST stay strictly relevant to the provided User Instructions."
)

# Hand-written Arabic ASMR exemplars (Modern Standard Arabic, whisper pacing).
SEED_SAMPLES = [
    (
        "صوت المطر على النافذة لمساعدتك على النوم",
        "[pause:2s] أغمض عينيك الآن، وخذ نفساً عميقاً وبطيئاً [pause] "
        "استمع إلى قطرات المطر وهي تلامس الزجاج برفق، واحدة تلو الأخرى، "
        "كأصابع صغيرة تطرق باباً ناعماً [pause:2s] كل قطرة تحمل معها "
        "همسة هدوء، وتغسل عنك تعب اليوم الطويل [pause] تخيل أن كل نسمة "
        "هواء تدخل صدرك تملؤه سلاماً، وكل زفير يخرج يأخذ معه القلق "
        "[pause:2s] أنت في مكان آمن، دافئ، بعيد عن كل ضجيج [pause] "
        "دع صوت المطر يحيط بك كبطانية ناعمة، واسمح لجفونك أن تثقل "
        "شيئاً فشيئاً [pause:3s] نم الآن، فالعالم من حولك ينام أيضاً، "
        "والمطر يحرس أحلامك حتى الصباح [pause:2s]",
    ),
    (
        "جلسة استرخاء في مكتبة قديمة هادئة",
        "[pause:2s] تخيل نفسك في مكتبة قديمة، تفوح منها رائحة الورق "
        "والخشب العتيق [pause] أنت وحدك تماماً، والصمت يحيط بك من كل "
        "جانب [pause:2s] اسمع صوت أصابعك وهي تمر ببطء على أغلفة الكتب، "
        "لمسة خفيفة، ناعمة، حنونة [pause] تسحب كتاباً قديماً برفق، "
        "وتسمع حفيف صفحاته وهي تنفتح أمامك [pause:2s] كل صفحة تقلبها "
        "تصدر صوتاً رقيقاً، كأجنحة فراشة تحط على زهرة [pause] الضوء "
        "الخافت يداعب الحروف، ونفسك يهدأ أكثر فأكثر [pause:2s] لا "
        "مواعيد، لا أصوات، لا شيء يستعجلك [pause] فقط أنت، والكتب، "
        "وهذا الصمت الجميل الذي يحتضنك بكل رقة [pause:3s]",
    ),
    (
        "همسات مسائية لتهدئة الأعصاب قبل النوم",
        "[pause:2s] مساؤك هادئ يا صديقي [pause] أنا هنا لأهمس في أذنك "
        "بكلمات ناعمة، تذيب كل توتر النهار [pause:2s] أولاً، أرخِ "
        "كتفيك، ودعهما يهبطان ببطء [pause] أغلق عينيك بلطف، كما "
        "تُغلق ستارة حريرية في أمسية صيفية [pause:2s] لاحظ كيف يتنفس "
        "جسدك وحده، دون أي مجهود منك [pause] شهيق يملؤك براحة، "
        "وزفير يحررك من كل ثقل [pause:2s] كل عضلة في وجهك تسترخي "
        "الآن: جبهتك، وجنتاك، وفكك [pause] أنت تغرق في هدوء عميق، "
        "كحجر يسكن قاع بحيرة صافية [pause:3s] وفي هذا الهدوء، تستحق "
        "أن تكون لطيفاً مع نفسك، فقد كان يومك طويلاً بما فيه الكفاية "
        "[pause:2s]",
    ),
    (
        "وصف هادئ لشاطئ بحر في الصباح الباكر",
        "[pause:2s] تخيل أنك تقف على شاطئ رملي في الصباح الباكر، "
        "والعالم لم يستيقظ بعد [pause] الرمال تحت قدميك باردة وناعمة، "
        "كمسحوق حريري [pause:2s] الأمواج تصل إلى الشاطئ بإيقاع بطيء "
        "ومتكرر: تقترب بهدوء، ثم تنسحب برقة [pause] اسمع صوتها الخافت، "
        "كأنه تنفس عميق لكائن عظيم نائم [pause:2s] نسيم البحر يلامس "
        "وجهك بلطف، يحمل رائحة الملح والنقاء [pause] مع كل موجة، يذوب "
        "قلقك أكثر في الرمل الرطب [pause:2s] لا أحد هنا سواك، ولا "
        "شيء يطلب منك شيئاً [pause] فقط البحر، والرمال، وأنت، في "
        "لحظة سلام كاملة [pause:3s]",
    ),
    (
        "جلسة عناية شخصية بأصوات ناعمة ومريحة",
        "[pause:2s] دعني أعتني بك قليلاً [pause] سأمشط شعرك ببطء "
        "شديد، خصلة بعد خصلة، بمشط خشبي ناعم [pause:2s] اسمع الصوت "
        "الرقيق للمشط وهو ينساب بين الخصلات، حريرياً وهادئاً [pause] "
        "كل تمريرة تأخذ معها جزءاً من توترك [pause:2s] والآن، أربت "
        "برفق على كتفيك، بضغط خفيف متكرر، كدقات قلب هادئة [pause] "
        "اشعر بالدفء يسري من أطراف أصابعي إلى عضلاتك المتعبة [pause:2s] "
        "تنفس بعمق، واستقبل هذه العناية، فأنت تستحق كل لحظة منها "
        "[pause] لا حاجة لأن تفعل شيئاً، فقط استسلم لهذا الدلال "
        "الهادئ [pause:3s] وأغمض عينيك، فأنت في أيدٍ أمينة [pause:2s]",
    ),
]

TOPIC_BANK = [
    "صوت نار المخيم في ليلة هادئة",
    "الاستلقاء في حديقة مزهرة في الربيع",
    "صوت عاصفة رعدية بعيدة وأنت في مكان دافئ",
    "الجلوس بجانب نافذة طائرة ليلاً",
    "صوت شلال ماء في غابة هادئة",
    "رحلة هادئة في قطار ليلي",
    "وصف إعداد كوب شاي بالنعناع ببطء",
    "الاستلقاء على عشب أخضر تحت سماء صافية",
    "صوت الرياح بين أغصان الأشجار في الخريف",
    "جلسة قراءة هادئة بجانب المدفأة",
    "المشي على أوراق الشجر الجافة ببطء",
    "وصف حديقة يابانية هادئة ببركة ماء",
    "صوت النوافير في ساحة قديمة",
    "الاستماع إلى صوت موج البحر من غرفة فندق",
    "جلسة تدليك هادئة لفروة الرأس",
    "وصف متحف هادئ بعد ساعات الإغلاق",
    "صوت الثلج يتساقط في قرية نائية",
    "الاستلقاء في أرجوحة تحت ظل شجرة",
    "وصف مقهى هادئ في الصباح الباكر",
    "جلسة تنفس عميق موجهة للنوم",
]

_GENERATOR_SYSTEM = (
    "You create training data for an Arabic ASMR scriptwriter. Write ONE "
    "immersive Arabic (Modern Standard Arabic) ASMR script for the given "
    "topic. Style: second-person, intimate whisper pacing, rich sensory "
    "detail, soothing and calming. Requirements: 90-170 words; insert "
    "[pause], [pause:2s] or [pause:3s] markers frequently; absolutely no "
    "titles, no markdown, no asterisks, no quotes, no meta commentary; "
    "output ONLY the spoken script text."
)


def _arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if "ء" <= c <= "ي")
    return arabic / len(letters)


def validate_script(script: str) -> tuple[bool, str]:
    """Quality gate for a candidate assistant response."""
    words = script.split()
    if not (60 <= len(words) <= 260):
        return False, f"word count {len(words)} outside 60-260"
    if _arabic_ratio(script) < 0.8:
        return False, "arabic letter ratio < 0.8"
    if "[pause" not in script:
        return False, "missing [pause] markers"
    for bad in ("**", "#", "```", "http", "العنوان", "Title:"):
        if bad in script:
            return False, f"contains forbidden artifact: {bad}"
    return True, "ok"


def _dedup_key(script: str) -> str:
    normalized = " ".join(script.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _to_record(topic: str, script: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User Instructions / Topic: {topic}"},
            {"role": "assistant", "content": script},
        ]
    }


def generate_with_gemini(topic: str, few_shot: str) -> str:
    """Generates one candidate script via the app's hardened Gemini caller."""
    from llm import _call_gemini

    user_msg = (
        f"Here is an example of the expected style:\n{few_shot}\n\n"
        f"Now write a NEW script (different imagery, same style) for the "
        f"topic: {topic}"
    )
    return _call_gemini(_GENERATOR_SYSTEM, user_msg).strip()


def build_dataset(count: int, offline: bool) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()

    def _try_add(topic: str, script: str) -> bool:
        script = script.strip()
        ok, reason = validate_script(script)
        key = _dedup_key(script)
        if not ok:
            print(f"  [skip] {topic[:40]}... -> {reason}")
            return False
        if key in seen:
            print(f"  [skip] {topic[:40]}... -> duplicate")
            return False
        seen.add(key)
        records.append(_to_record(topic, script))
        return True

    for topic, script in SEED_SAMPLES:
        _try_add(topic, script)
    print(f"[seeds] {len(records)} built-in exemplars accepted")

    if not offline:
        few_shot = SEED_SAMPLES[0][1]
        topic_index = 0
        attempts = 0
        max_attempts = max(count * 3, 10)
        while len(records) < count and attempts < max_attempts:
            topic = TOPIC_BANK[topic_index % len(TOPIC_BANK)]
            topic_index += 1
            attempts += 1
            try:
                script = generate_with_gemini(topic, few_shot)
            except Exception as exc:
                print(f"  [warn] Gemini generation failed for '{topic}': {exc}")
                continue
            if _try_add(topic, script):
                print(f"  [ok] {len(records)}/{count}: {topic}")

    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=60,
                        help="Target number of samples (default: 60).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path.")
    parser.add_argument("--offline", action="store_true",
                        help="Only emit built-in seed exemplars (no Gemini calls).")
    args = parser.parse_args(argv)

    records = build_dataset(count=args.count, offline=args.offline)
    if not records:
        print("error: no valid samples produced.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(records)} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
