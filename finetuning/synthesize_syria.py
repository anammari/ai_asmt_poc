# finetuning/synthesize_syria.py
"""
Synthesize Syrian ASMR training records to augment real ingested records.

Reads real records from finetuning/data/ingested/syria/ and seed templates
from finetuning/synthetic_seeds.json. Expands each seed to match the average
word count of the real data.

Append-only: re-running only adds new synthetic records. Existing records
(real or synthetic) are never modified or removed. Each batch uses
deterministic seeding so the same inputs always produce the same output.

Usage:
  python finetuning/synthesize_syria.py                           # default: 6 or len(real)
  python finetuning/synthesize_syria.py --count 10                # target 10 synthetic total
  python finetuning/synthesize_syria.py --count 10 --force         # regenerate from scratch
"""
import argparse
import hashlib
import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

DIALECT_DISPLAY = "Syrian Arabic"
INGESTED_DIR = os.path.join(SCRIPT_DIR, "data", "ingested", "syria")
JSONL_PATH = os.path.join(SCRIPT_DIR, "data", "training", "syria", "arabic_asmr_sft.jsonl")
SEEDS_PATH = os.path.join(SCRIPT_DIR, "synthetic_seeds.json")

# ---- Expansion phrase libraries (same as before) ----

SLOW_DOWN_PHRASES = [
    "بهدوء شديد", "على مهلك", "شوي شوي", "ببطء", "نفس عميق",
    "براحه", "بهدوء", "برفق", "بنعومه", "براحه تامه",
]

BREATH_PROMPTS = [
    "خذ شهيق عميق من انفك", "زفر ببطء من فمك",
    "نفس عميق وبطيء", "تنفس بعمق وراحه",
    "خلي النفس يملأ رئتك", "اطلع الهواء بهدوء",
]

COMFORT_PHRASES = [
    "انت بأمان", "كل شي تمام", "لا تقلق", "انا معك",
    "انت مرتاح", "هادئ بالك", "ريح بالك", "انت بخير",
]

PHYSICAL_SENSATIONS = [
    "احس بوزن جسمك عالسرير", "لاحظ كيف كتفيك نزلوا لتحت",
    "اشعر بالدفا ينتشر بجسمك", "احس بارتخاء عضلات وجهك",
    "لاحظ كيف رجليك صارت ثقيله", "اشعر بالهدوء يغلفك",
]

SALON_PHRASES = [
    "خليني اشتغل على هالخصله بهدوء", "المشط بيمشي بخفه على شعرك",
    "شو رايك بهالطريقه بتناسبك", "بعد ما نخلص رح تكوني مرتاحه",
    "المقص بيمشي بهدوء على الاطراف", "شعرك ناعم وحريري وبيستاهل العنايه",
    "وبعد هيك ننتقل للجهه التانيه", "وانا هون عم اشتغل بهدوء وانت مرتاحه",
    "كل خصله بتاخد وقتها وراحتها", "اللمسات الناعمه هي يلي بتفرق",
]

MAMA_PHRASES = [
    "انا هون جنبك وانا معك", "لا تخاف انا موجوده",
    "نام بهدوء وانت بأمان", "انا رح ضل ساهر جنبك حتى تنام",
    "كل شيء تمام وما في داعي للقلق", "انت ولدي وانت نور عيوني",
    "سكر عيونك بهدوء وخلينا نغفو سوا", "انا بحبك وانا رح ضل معك دايما",
    "ما في شي بيستاهل قلقك", "انت بأمان تام وهون في حضني",
]

FRIEND_PHRASES = [
    "لما نحكي سوا بحس بكل شي", "الصداقه كنز ما الها ثمن",
    "انا فخره فيك يلي صديقتي", "بتعرفي اني بحبك موت",
    "ايامنا مع بعض احلى ايام", "لما نضحك سوا بنسى كل همومي",
    "انت اجمل هديه من رب العالمين", "نضل سوا مهما صار",
    "حتى لو صار شي بترجع لبعض", "لان الصداقه الحقيقيه ما بتنتهي",
]

SEA_PHRASES = [
    "الموج عم ييجي ويروح بهدوء", "تأمل الافق البعيد",
    "لاحظ كيف القمر عم يضيء المي", "النسيم البارد عم يمسح ع وجهك",
    "صوت الموج يريح القلب", "كل موجه بتاخد معها همه",
    "النجوم عم تضوي بالسما", "البحر واسع وكبير مثلك",
    "انت جزء من هاد الجمال", "وكل شيء في مكانه الصحيح",
]

YOGA_PHRASES = [
    "اجلسي بوضعيه مريحه", "سكري عيونك بهدوء",
    "لاحظي كيف جسمك عم يرتاح", "كل عضله في جسمك بترتخي",
    "من قدميك لراسك كل شي مرتاح", "التنفس العميق بيدخل الراحه",
    "مع كل شهيق بتدخلي طاقه", "مع كل زفير بتطردي التعب",
    "خلي الافكار تمر زي الغيوم", "وانت هون مرتاحه وهادئه",
]

GRANDMA_PHRASES = [
    "تيتا دايم كانت تقول هيك", "ريحه الاكل بتعبي البيت",
    "الاكل البيتي فيه روح", "الوصفه من تيتا وبتعيش",
    "كل مره اطبخ هالاكله بتذكرها", "تيتا كانت طبخها بالحب",
    "البيت كان دايم يفوح برايحه", "هالوصفه عمرها ميه سنه",
    "تيتا علمتني انه الطبخ عنايه", "والحب بيضل موجود للابد",
]

TOPIC_EXPANSIONS: dict[str, list[str]] = {
    "كوافيره": SALON_PHRASES,
    "ماما": MAMA_PHRASES,
    "صديقه": FRIEND_PHRASES,
    "بحر": SEA_PHRASES,
    "يوجا": YOGA_PHRASES,
    "تيتا": GRANDMA_PHRASES,
}

# ---- Data loading ----

def load_ingested_records() -> list[dict]:
    """Read real records from ingested directory."""
    records = []
    if not os.path.isdir(INGESTED_DIR):
        return records
    for filename in sorted(os.listdir(INGESTED_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(INGESTED_DIR, filename), encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def load_existing_jsonl() -> list[dict]:
    """Read existing training JSONL."""
    if not os.path.exists(JSONL_PATH):
        return []
    records = []
    with open(JSONL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_seeds() -> list[dict]:
    """Read seed templates from synthetic_seeds.json."""
    if not os.path.exists(SEEDS_PATH):
        raise FileNotFoundError(
            f"Seeds file not found: {SEEDS_PATH}. "
            "Run `uv run python finetuning/synthesize_syria.py` to generate it."
        )
    with open(SEEDS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def extract_user_topic(record: dict) -> str | None:
    """Extract the user message content from a chat-format record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"].strip()
    return None


def user_topic(title: str) -> str:
    return f"User Instructions / Topic: [{DIALECT_DISPLAY}] {title}"


def _to_record(title: str, transcript: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_topic(title)},
            {"role": "assistant", "content": transcript},
        ]
    }

# ---- Expansion engine ----

def expand_transcript(
    seed_text: str,
    expansion_pool: list[str],
    target_words: int,
    rng: random.Random,
) -> str:
    """Expand a seed transcript to target word count using deterministic RNG."""
    words = seed_text.split()
    if len(words) >= target_words:
        return seed_text

    lines = [line.strip() for line in seed_text.split(" ") if line.strip()]
    result_lines = []
    needed = target_words - len(words)

    while needed > 0:
        for line in lines:
            result_lines.append(line)
            if rng.random() < 0.25 and expansion_pool:
                phrase = rng.choice(expansion_pool)
                result_lines.append(phrase)
                needed -= len(phrase.split())
            if rng.random() < 0.08:
                bp = rng.choice(BREATH_PROMPTS)
                result_lines.append(bp)
                needed -= len(bp.split())
            if rng.random() < 0.06:
                cf = rng.choice(COMFORT_PHRASES)
                result_lines.append(cf)
                needed -= len(cf.split())
            if needed <= 0:
                break
        rng.shuffle(expansion_pool)
        if needed > 0 and rng.random() < 0.1:
            ps = rng.choice(PHYSICAL_SENSATIONS)
            result_lines.append(ps)
            needed -= len(ps.split())

    result = " ".join(result_lines)
    current_words = len(result.split())
    while current_words < target_words - 50:
        if expansion_pool:
            result += " " + rng.choice(expansion_pool)
        result += " " + rng.choice(SLOW_DOWN_PHRASES)
        current_words = len(result.split())

    if current_words > target_words + 100:
        words_list = result.split()
        result = " ".join(words_list[:target_words])

    return result


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


def _generate_batch_title(base_title: str, batch_index: int, used_titles: set[str]) -> str:
    """Generate a unique synthetic title by appending a batch suffix."""
    suffix = f" (الجزء {batch_index + 1})"
    candidate = base_title + suffix
    # Ensure uniqueness by incrementing if needed.
    attempt = 0
    while _title_hash(candidate) in used_titles or candidate in used_titles:
        attempt += 1
        candidate = base_title + f" (الجزء {batch_index + 1} - {attempt})"
    return candidate


def get_synthetic_count(existing_records: list[dict], ingested_titles: set[str]) -> int:
    """Count how many existing JSONL records are synthetic (not from ingested data)."""
    count = 0
    for rec in existing_records:
        topic = extract_user_topic(rec)
        if topic:
            # Synthetic topics are those whose stripped title hash does not match any ingested title.
            title_part = topic.split("Topic:")[-1].strip()
            # Check if this matches any ingested title
            is_real = any(title_part == user_topic(t) for t in ingested_titles)
            if not is_real:
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize Syrian ASMR records to augment real ingested data (append-only)."
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Target total number of synthetic records. Default: max(6, num_real_records).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all synthetic records from scratch (removes existing synthetic records first).",
    )
    args = parser.parse_args(argv)

    # Load real ingested records.
    ingested = load_ingested_records()
    if not ingested:
        print("error: no ingested records found", file=sys.stderr)
        return 1

    ingested_titles = set()
    real_records_data = []  # list of (title, transcript)
    for item in ingested:
        title = (item.get("title") or "").strip()
        transcript = (item.get("transcript") or "").strip()
        if title and transcript:
            ingested_titles.add(title)
            real_records_data.append((title, transcript))

    if not real_records_data:
        print("error: no valid real records found", file=sys.stderr)
        return 1

    avg_wc = sum(len(t) for _, t in real_records_data) // len(real_records_data)

    # Determine target synthetic count.
    target_total = args.count if args.count > 0 else max(6, len(real_records_data))

    # Load existing JSONL and separate.
    existing = load_existing_jsonl()
    synthetic_exists = [r for r in existing if extract_user_topic(r) and
                        not any(extract_user_topic(r) == user_topic(t) for t in ingested_titles)]
    real_exists = [r for r in existing if extract_user_topic(r) and
                   any(extract_user_topic(r) == user_topic(t) for t in ingested_titles)]

    if args.force:
        # Discard all existing synthetic records, keep only real.
        synthetic_exists = []
        existing = real_exists
        # Rewrite the JSONL with only real records.
        os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
        with open(JSONL_PATH, "w", encoding="utf-8") as fh:
            for record in real_exists:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    current_synthetic = len(synthetic_exists)
    needed = target_total - current_synthetic

    if needed <= 0:
        print(f"[info] synthetic target already met ({current_synthetic}/{target_total})")
        print(f"[done] 0 new synthetic records; no changes to {JSONL_PATH}")
        return 0

    # Load seed templates.
    seeds = load_seeds()
    if not seeds:
        print("error: no seeds found in synthetic_seeds.json", file=sys.stderr)
        return 1

    # Collect existing synthetic user topics for dedup.
    used_topics: set[str] = set()
    for r in synthetic_exists:
        topic = extract_user_topic(r)
        if topic:
            used_topics.add(topic)
    for r in real_exists:
        topic = extract_user_topic(r)
        if topic:
            used_topics.add(topic)

    # Use deterministic seeding: unique seed per batch.
    batch_seed = 42 + current_synthetic
    rng = random.Random(batch_seed)
    print(f"[setup] batch seed={batch_seed}, target={target_total}, "
          f"existing synthetic={current_synthetic}, generating={needed}")

    new_records: list[dict] = []
    seed_index = 0
    for i in range(needed):
        # Cycle through seeds, generate unique titles.
        seed_template = seeds[seed_index % len(seeds)]
        seed_index += 1

        base_title = seed_template["title"]
        pool_key = seed_template["pool"]
        expansion_pool = TOPIC_EXPANSIONS.get(pool_key, list(TOPIC_EXPANSIONS.keys()))
        seed_text = seed_template["seed"]

        new_title = _generate_batch_title(base_title, current_synthetic + i, used_topics)
        topic_str = user_topic(new_title)

        if topic_str in used_topics:
            continue  # Safety: skip if collision

        expanded = expand_transcript(seed_text, expansion_pool.copy(), avg_wc, rng)
        new_records.append(_to_record(new_title, expanded))
        used_topics.add(topic_str)

    if not new_records:
        print("[info] no new synthetic records generated (all topics collided)")
        return 0

    # Append to JSONL.
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
    with open(JSONL_PATH, "a", encoding="utf-8") as fh:
        for record in new_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[stats] batch: {len(new_records)} new synthetic records")
    for i, rec in enumerate(new_records):
        text = rec["messages"][2]["content"]
        topic = extract_user_topic(rec)
        print(f"  synth [{current_synthetic + i + 1}] {topic[:70]}...")
        print(f"             {len(text.split())} words, {len(text)} chars")

    new_total = current_synthetic + len(new_records)
    print(f"[done] appended {len(new_records)} synthetic records -> {JSONL_PATH} "
          f"(synthetic total: {new_total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())