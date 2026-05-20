"""Ollama-based translation for video dubbing.

- Works with any source language -> any target language.
- Strict prompt: model MUST translate every word into the target language.
- Post-pass retry: if a line is still mostly in the wrong script, retry individually.
- Strips thinking/reasoning tokens from Qwen3/Gemma4.
"""
import json
import logging
import os
import re
import time

import httpx

log = logging.getLogger("tachidubb.translator")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

LANG_NAMES = {
    "en": "English", "ru": "Russian", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi", "it": "Italian",
    "tr": "Turkish", "uk": "Ukrainian", "pl": "Polish", "nl": "Dutch",
    "sv": "Swedish", "th": "Thai", "vi": "Vietnamese", "da": "Danish",
    "fi": "Finnish", "el": "Greek", "he": "Hebrew", "id": "Indonesian",
    "ms": "Malay", "no": "Norwegian", "tl": "Tagalog", "sw": "Swahili",
    "cs": "Czech", "ro": "Romanian", "hu": "Hungarian", "bg": "Bulgarian",
    "ka": "Georgian", "hy": "Armenian", "az": "Azerbaijani", "kk": "Kazakh",
    "uz": "Uzbek", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
    "ur": "Urdu", "fa": "Persian", "sr": "Serbian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "lv": "Latvian", "lt": "Lithuanian",
    "et": "Estonian", "is": "Icelandic", "ca": "Catalan",
}

# Script blocks per language family. Used to detect "forgot to translate".
SCRIPTS = {
    "cyrillic":   r"[\u0400-\u04ff]",
    "latin":      r"[A-Za-z]",
    "cjk":        r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]",
    "arabic":     r"[\u0600-\u06ff]",
    "hebrew":     r"[\u0590-\u05ff]",
    "greek":      r"[\u0370-\u03ff]",
    "devanagari": r"[\u0900-\u097f]",
    "bengali":    r"[\u0980-\u09ff]",
    "tamil":      r"[\u0b80-\u0bff]",
    "telugu":     r"[\u0c00-\u0c7f]",
    "thai":       r"[\u0e00-\u0e7f]",
    "georgian":   r"[\u10a0-\u10ff]",
    "armenian":   r"[\u0530-\u058f]",
}

LANG_SCRIPT = {
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic",
    "kk": "cyrillic",
    "en": "latin", "es": "latin", "fr": "latin", "de": "latin", "it": "latin",
    "pt": "latin", "pl": "latin", "nl": "latin", "sv": "latin", "da": "latin",
    "fi": "latin", "no": "latin", "cs": "latin", "ro": "latin", "hu": "latin",
    "hr": "latin", "sk": "latin", "sl": "latin", "lv": "latin", "lt": "latin",
    "et": "latin", "tr": "latin", "vi": "latin", "id": "latin", "ms": "latin",
    "tl": "latin", "sw": "latin", "is": "latin", "ca": "latin",
    "az": "latin", "uz": "latin",
    "zh": "cjk", "ja": "cjk", "ko": "cjk",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "he": "hebrew", "el": "greek",
    "hi": "devanagari", "bn": "bengali",
    "ta": "tamil", "te": "telugu",
    "th": "thai", "ka": "georgian", "hy": "armenian",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code[:2].lower(), code)


def _clean_response(text: str) -> str:
    """Strip thinking/reasoning tokens produced by Qwen3 and Gemma4."""
    # Qwen3 style
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Gemma4 channel markers (multiple syntax variants seen in the wild)
    text = re.sub(r"<\|channel\|?>\s*thought.*?<\|?channel\|?>", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel\|?>.*?<\|?channel\|?>", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Any stray <|...|> control tokens
    text = re.sub(r"<\|[a-z_\-]+\|>", "", text, flags=re.IGNORECASE)

    # Gemma4 often dumps visible thinking blocks as plain markdown, like:
    #   Thinking...
    #   Thinking Process:
    #   1. **Analyze the input:** ...
    #   2. **Determine the context:** ...
    #   ...done thinking.
    # Strip from "Thinking" heading to "...done thinking." (inclusive),
    # case-insensitive, spans multiple lines. If no closing marker, cut
    # up to the first blank line followed by actual translation content.
    text = re.sub(
        r"(?is)(?:^|\n)\s*thinking(?:\s+process)?[:\.]*.*?(?:\.\.\.\s*done\s+thinking\.?|\n\s*\n)",
        "\n", text,
    )
    # Also strip leading "Thinking..." standalone line
    text = re.sub(r"(?im)^\s*thinking\.{0,3}\s*$", "", text)
    return text.strip()


def _looks_untranslated(text: str, target_code: str) -> bool:
    """True if text is mostly NOT in the target script (heuristic)."""
    if not text.strip():
        return False
    tgt_script = LANG_SCRIPT.get(target_code[:2].lower())
    if not tgt_script:
        return False
    pattern = SCRIPTS.get(tgt_script)
    if not pattern:
        return False
    target_chars = len(re.findall(pattern, text))
    letter_chars = len(re.findall(r"[^\s\d\.,!?:;\-()\"'\[\]]", text))
    if letter_chars < 3:
        return False
    ratio = target_chars / letter_chars
    return ratio < 0.5  # >50% chars outside target script = not translated


async def check_ollama(url: str = "") -> tuple[bool, list[str]]:
    """Check Ollama health; list installed models."""
    target = url or OLLAMA_URL
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{target}/api/tags")
            if r.status_code == 200:
                return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return False, []


async def unload_ollama_model(model: str, url: str = "") -> bool:
    """Force Ollama to unload a model from memory RIGHT NOW.

    Ollama normally keeps the last-used model in VRAM for 5 minutes
    (OLLAMA_KEEP_ALIVE). On a 12 GB GPU shared with VoxCPM this causes
    memory pressure during TTS. Sending a request with keep_alive=0
    triggers immediate unload. We use num_predict=1 so the request is
    cheap; Ollama treats it as "last request before unload".

    Additionally polls /api/ps to confirm the model actually dropped
    out of VRAM before returning, with up to 10s wait. Without this
    wait the model is still in VRAM when VoxCPM tries to load.

    Returns True if the model is confirmed unloaded from VRAM.
    """
    import asyncio as _asyncio
    target = url or OLLAMA_URL
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(f"{target}/api/generate", json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,  # KEY: unload immediately
                "options": {"num_predict": 1},
            })
            if r.status_code != 200:
                log.warning(f"[ollama] Unload request HTTP {r.status_code}")
                return False

            # Poll /api/ps until the model is no longer listed or 10s elapse
            for attempt in range(20):  # 20 x 500ms = 10s max
                await _asyncio.sleep(0.5)
                try:
                    ps = await c.get(f"{target}/api/ps")
                    if ps.status_code != 200:
                        continue
                    loaded = {m.get("name", "") for m in ps.json().get("models", [])}
                    still_loaded = any(
                        name == model or name.startswith(model)
                        for name in loaded
                    )
                    if not still_loaded:
                        log.info(
                            f"[ollama] Unloaded model '{model}' from VRAM "
                            f"(confirmed after {(attempt + 1) * 0.5:.1f}s)"
                        )
                        return True
                except Exception:
                    continue
            log.warning(
                f"[ollama] Unload requested for '{model}' but still in VRAM "
                f"after 10s — VoxCPM may compete for GPU memory"
            )
            return False
    except Exception as e:
        log.debug(f"Unload request failed: {e}")
        return False


async def ollama_pull_stream(model: str, url: str = ""):
    """Async generator that streams pull progress lines."""
    target = url or OLLAMA_URL
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{target}/api/pull",
                            json={"name": model, "stream": True}) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ═══════════════════════════════════════════════════════════════════════
#  Domain glossaries
# ═══════════════════════════════════════════════════════════════════════
# Gemma on its own sometimes mistranslates sport/technical terminology
# into nonsense or literal translations ("sweep" → "метла" instead of
# "свип"). When the user's context_hint names a known domain, we inject
# an authoritative term list so the model uses the right words.
# Add more domains as needed — format is:
#   (trigger_keywords_in_context_hint,
#    {en_term: preferred_target_translation_for_ru})
# ═══════════════════════════════════════════════════════════════════════
_DOMAIN_GLOSSARIES_RU = [
    (
        ("bjj", "jiu-jitsu", "jiu jitsu", "jujitsu", "grappling", "mma",
         "submission grappling", "nogi", "no-gi", "brazilian jiu"),
        {
            # ─── Sweeps (свипы) ───────────────────────────────────
            "sweep": "свип",
            "sweeps": "свипы",
            "hip bump sweep": "хип-бамп свип",
            "hip bump": "хип-бамп",
            "scissor sweep": "ножничный свип",
            "flower sweep": "флауэр свип",
            "pendulum sweep": "маятниковый свип",
            "butterfly sweep": "баттерфляй свип",
            "x-guard sweep": "свип из икс-гарда",
            "reverse de la riva": "реверс де ла рива",
            "de la riva sweep": "свип де ла рива",
            "lasso sweep": "лассо свип",
            "tripod sweep": "трипод свип",
            "tomoe nage": "томое наге",
            "balloon sweep": "балун свип",

            # ─── Guards (гарды) ───────────────────────────────────
            "guard": "гард",
            "full guard": "фулл-гард",
            "closed guard": "клоуз-гард",
            "open guard": "опен-гард",
            "half guard": "хаф-гард",
            "deep half guard": "дип хаф-гард",
            "z-guard": "z-гард",
            "knee shield": "ни-шилд",
            "butterfly guard": "баттерфляй-гард",
            "spider guard": "спайдер-гард",
            "lasso guard": "лассо-гард",
            "x-guard": "икс-гард",
            "single leg x": "сингл-лег икс",
            "single-leg x": "сингл-лег икс",
            "de la riva": "де ла рива",
            "reverse de la riva": "реверс де ла рива",
            "dlr": "ДЛР",
            "rdlr": "РДЛР",
            "50/50": "фифти-фифти",
            "50-50": "фифти-фифти",
            "rubber guard": "раббер-гард",
            "worm guard": "ворм-гард",
            "k-guard": "k-гард",
            "collar and sleeve": "коллар-слив",

            # ─── Positions (позиции) ──────────────────────────────
            "mount": "маунт",
            "full mount": "фулл-маунт",
            "s-mount": "s-маунт",
            "technical mount": "техникал-маунт",
            "side control": "сайд-контроль",
            "side mount": "сайд-маунт",
            "cross side": "кросс-сайд",
            "knee on belly": "ни-он-белли",
            "knee mount": "ни-маунт",
            "north-south": "норт-саут",
            "north south": "норт-саут",
            "back mount": "бэк-маунт",
            "back control": "бэк-контроль",
            "back take": "бэк-тейк",
            "turtle": "тёртл",
            "turtle position": "позиция тёртл",
            "front headlock": "фронт-хедлок",
            "seatbelt": "ситбелт",
            "body triangle": "боди-трайэнгл",
            "hooks": "хуки",

            # ─── Submissions / Chokes (сабмишны) ──────────────────
            "submission": "сабмишн",
            "submissions": "сабмишны",
            "sub": "саб",
            "tap": "сдался",
            "tap out": "сдаться",
            "tapped": "сдался",
            "tapping": "сдача",
            "choke": "удушающий",
            "chokes": "удушающие",
            "strangle": "удушение",
            "rear naked choke": "удушающий сзади",
            "rnc": "удушающий сзади",
            "bow and arrow": "бау-энд-эрроу",
            "bow and arrow choke": "бау-энд-эрроу",
            "triangle": "треугольник",
            "triangle choke": "треугольник",
            "reverse triangle": "реверс-треугольник",
            "mounted triangle": "треугольник из маунта",
            "arm triangle": "арм-треугольник",
            "d'arce": "дарс",
            "d'arce choke": "дарс",
            "darce": "дарс",
            "anaconda": "анаконда",
            "anaconda choke": "анаконда",
            "guillotine": "гильотина",
            "guillotine choke": "гильотина",
            "ezekiel": "эзекил",
            "ezekiel choke": "эзекил",
            "cross choke": "кросс-чок",
            "loop choke": "луп-чок",
            "baseball choke": "бейсбол-чок",
            "clock choke": "клок-чок",
            "north-south choke": "норт-саут чок",

            # ─── Joint locks (болевые) ────────────────────────────
            "armbar": "армбар",
            "armlock": "армлок",
            "juji gatame": "джуджи гатаме",
            "kimura": "кимура",
            "americana": "американа",
            "omoplata": "омоплата",
            "wristlock": "вристлок",
            "wrist lock": "вристлок",
            "heel hook": "хил-хук",
            "inside heel hook": "внутренний хил-хук",
            "outside heel hook": "внешний хил-хук",
            "knee bar": "нибар",
            "kneebar": "нибар",
            "toe hold": "то-холд",
            "straight ankle": "прямой анкл-лок",
            "ankle lock": "анкл-лок",
            "calf slicer": "калф-слайсер",
            "bicep slicer": "байсепс-слайсер",
            "leg lock": "леглок",
            "leg locks": "леглоки",

            # ─── Grips (захваты) ──────────────────────────────────
            "grip": "захват",
            "grips": "захваты",
            "collar grip": "захват за воротник",
            "sleeve grip": "захват за рукав",
            "lapel grip": "захват за лацкан",
            "cross collar": "кросс-коллар",
            "gable grip": "гейбл-захват",
            "s-grip": "s-захват",
            "ball and socket": "бол-энд-сокет",
            "underhook": "андерхук",
            "underhooks": "андерхуки",
            "overhook": "оверхук",
            "overhooks": "оверхуки",
            "double underhooks": "двойные андерхуки",
            "pummel": "прокачка",
            "pummeling": "пламмелинг",
            "cross-face": "кросс-фейс",
            "crossface": "кросс-фейс",
            "whizzer": "визер",
            "kimura grip": "кимура-захват",

            # ─── Takedowns / standup (тейкдауны) ──────────────────
            "takedown": "тейкдаун",
            "takedowns": "тейкдауны",
            "shot": "проход",
            "shoot": "бросок в ноги",
            "double leg": "дабл-лег",
            "double leg takedown": "дабл-лег",
            "single leg": "сингл-лег",
            "high crotch": "хай-кроч",
            "ankle pick": "анкл-пик",
            "foot sweep": "подножка",
            "hip throw": "бросок через бедро",
            "seoi nage": "сеои нагэ",
            "uchi mata": "учи мата",
            "o-goshi": "о-гоши",
            "trip": "подсечка",
            "leg trip": "подсечка",
            "sprawl": "спроул",
            "sprawling": "спроулинг",

            # ─── Guard passes (проходы) ───────────────────────────
            "pass": "проход",
            "passing": "прохождение",
            "guard pass": "проход гарда",
            "pass the guard": "пройти гард",
            "torreando": "торреандо",
            "toreando": "торреандо",
            "bullfighter pass": "торреандо",
            "knee slice": "ни-слайс",
            "knee slide": "ни-слайс",
            "knee cut": "ни-кат",
            "over-under": "овер-андер",
            "over under pass": "овер-андер проход",
            "double under": "дабл-андер",
            "double underpass": "дабл-андер проход",
            "stack pass": "стэк-проход",
            "leg drag": "лег-драг",
            "smash pass": "смэш-проход",
            "long step": "лонг-степ",
            "x-pass": "x-проход",

            # ─── Concepts / slang (сленг) ─────────────────────────
            "roll": "роллинг",
            "rolling": "роллинг",
            "roll with": "покатать с",
            "drill": "дрилл",
            "drilling": "дриллить",
            "drills": "дриллы",
            "flow roll": "флоу-роллинг",
            "flow": "флоу",
            "live roll": "лайв-роллинг",
            "sparring": "спарринг",
            "scramble": "скрамбл",
            "scrambling": "скрамблинг",
            "escape": "эскейп",
            "escapes": "эскейпы",
            "bridge": "бридж",
            "bridging": "бриджинг",
            "shrimp": "шримп",
            "shrimping": "шримпинг",
            "hip escape": "шримп",
            "technical stand-up": "техникал стэнд-ап",
            "technical standup": "техникал стэнд-ап",
            "frame": "фрейм",
            "framing": "фрейминг",
            "posture": "поза",
            "post": "пост",
            "base": "база",
            "basing": "базинг",
            "off-balance": "оффбэланс",
            "kuzushi": "кузуши",
            "transition": "переход",
            "transitions": "переходы",
            "chain": "цепочка",
            "chaining": "связка",
            "combo": "связка",
            "setup": "сетап",
            "set up": "сетап",
            "setting up": "готовлю",
            "entry": "вход",
            "finish": "финиш",
            "finishing": "финиш",
            "control": "контроль",
            "pressure": "давление",
            "heavy": "тяжёлый",
            "light": "лёгкий",
            "tight": "плотный",
            "loose": "свободный",
            "slippery": "скользкий",
            "timing": "тайминг",
            "connection": "коннекшн",
            "leverage": "рычаг",

            # ─── Slang / culture (сленг, жаргон) ──────────────────
            "tap or snap": "сдавайся или ломайся",
            "tap, nap, or snap": "сдавайся, засыпай или ломайся",
            "gi": "ги",
            "kimono": "кимоно",
            "nogi": "ноги-ги",
            "no-gi": "ноги-ги",
            "no gi": "ноги-ги",
            "rash guard": "рашгард",
            "rashguard": "рашгард",
            "spats": "спатсы",
            "mat": "мат",
            "mats": "маты",
            "mat time": "маттайм",
            "gym": "зал",
            "academy": "академия",
            "dojo": "додзё",
            "training partner": "тренировочный партнёр",
            "partner": "партнёр",
            "opponent": "оппонент",
            "white belt": "белый пояс",
            "blue belt": "синий пояс",
            "purple belt": "фиолетовый пояс",
            "brown belt": "коричневый пояс",
            "black belt": "чёрный пояс",
            "coral belt": "коралловый пояс",
            "stripe": "нашивка",
            "stripes": "нашивки",
            "instructor": "инструктор",
            "professor": "профессор",
            "coach": "тренер",
            "student": "ученик",
            "oss": "осс",
            "osu": "осс",
            "tournament": "турнир",
            "competition": "соревнование",
            "comp": "соревнование",
            "adcc": "ADCC",
            "ibjjf": "IBJJF",
            "open mat": "опен-мат",
            "spaz": "спаз",
            "spazzy": "спазовый",
            "spazzing out": "спазить",
            "gassed": "запыхался",
            "gas tank": "газ-танк",
            "cardio": "кардио",
            "shrimp out": "вышримпуй",
            "invert": "инверт",
            "inverted": "перевёрнутая позиция",
            "inversion": "инверсия",
            "stall": "затягивать",
            "stalling": "затягивание",

            # ─── Common instructor phrases (фразы тренера) ────────
            "be careful": "осторожно",
            "good job": "молодец",
            "nice work": "хорошая работа",
            "that's it": "вот именно",
            "keep going": "продолжай",
            "one more": "ещё раз",
            "switch": "меняемся",
            "switch partners": "меняем партнёров",
            "reset": "сброс",
            "start over": "начнём сначала",
            "let's go": "поехали",
            "watch out": "осторожно",
            "stay tight": "держись плотно",
            "stay heavy": "оставайся тяжёлым",
            "keep the pressure": "держи давление",
            "control the distance": "контролируй дистанцию",
            "hips down": "бёдра вниз",
            "hips up": "бёдра вверх",
            "get your hips in": "подключи бёдра",
            "use your hips": "используй бёдра",
        },
    ),
    (
        ("cooking", "recipe", "kitchen", "food", "culinary"),
        {
            # ─── Techniques ───────────────────────────────────────
            "whisk": "взбить венчиком",
            "sauté": "обжарить",
            "saute": "обжарить",
            "simmer": "томить на медленном огне",
            "boil": "кипятить",
            "poach": "припустить",
            "braise": "тушить",
            "blanch": "бланшировать",
            "sear": "обжарить до корочки",
            "roast": "запечь",
            "bake": "печь",
            "broil": "жарить на гриле",
            "grill": "гриль",
            "deep-fry": "жарить во фритюре",
            "stir-fry": "обжаривать на сильном огне",
            "steam": "на пару",
            "reduce": "уварить",
            "caramelize": "карамелизовать",
            "deglaze": "деглазировать",
            "fold in": "аккуратно вмешать",
            "fold": "аккуратно перемешать",
            "knead": "замешивать",
            "proof": "расстойка",
            "marinate": "мариновать",
            "render": "вытопить жир",
            "temper": "темперировать",

            # ─── Cuts ─────────────────────────────────────────────
            "dice": "нарезать кубиками",
            "diced": "кубиками",
            "mince": "мелко нарубить",
            "minced": "рубленый",
            "chop": "нарезать",
            "chopped": "нарезанный",
            "julienne": "нарезать соломкой",
            "brunoise": "брюнуаз",
            "slice": "нарезать ломтиками",
            "sliced": "ломтиками",
            "chiffonade": "шифонад",
            "cube": "кубик",
            "wedge": "долька",

            # ─── Equipment ────────────────────────────────────────
            "sheet pan": "противень",
            "dutch oven": "казан",
            "skillet": "сковорода",
            "saucepan": "сотейник",
            "cast iron": "чугун",
            "wok": "вок",
            "whisk (n)": "венчик",
            "spatula": "лопатка",
            "ladle": "половник",
            "tongs": "щипцы",
            "mortar and pestle": "ступка и пестик",
            "food processor": "кухонный комбайн",
            "stand mixer": "планетарный миксер",

            # ─── Ingredients & slang ──────────────────────────────
            "a dash": "щепотка",
            "a pinch": "щепотка",
            "to taste": "по вкусу",
            "room temperature": "комнатной температуры",
            "al dente": "аль денте",
            "umami": "умами",
        },
    ),
    (
        ("tech", "software", "programming", "coding", "developer",
         "devops", "engineering"),
        {
            "commit": "коммит",
            "commits": "коммиты",
            "pull request": "пул-реквест",
            "pr": "ПР",
            "merge": "смёржить",
            "merge conflict": "мёрдж-конфликт",
            "rebase": "ребейз",
            "rebasing": "ребейзить",
            "branch": "ветка",
            "checkout": "чекаут",
            "push": "запушить",
            "pull": "запуллить",
            "fork": "форк",
            "clone": "клонировать",
            "repo": "репа",
            "repository": "репозиторий",
            "deploy": "задеплоить",
            "deployment": "деплой",
            "rollback": "откатить",
            "debug": "дебажить",
            "debugging": "дебаг",
            "refactor": "отрефакторить",
            "refactoring": "рефакторинг",
            "build": "билд",
            "release": "релиз",
            "staging": "стейджинг",
            "production": "продакшн",
            "prod": "прод",
            "ci/cd": "CI/CD",
            "pipeline": "пайплайн",
            "api": "API",
            "endpoint": "эндпоинт",
            "payload": "пейлоад",
            "backend": "бэкенд",
            "frontend": "фронтенд",
            "stack": "стек",
            "framework": "фреймворк",
            "library": "библиотека",
            "dependency": "зависимость",
            "dependencies": "зависимости",
            "bug": "баг",
            "bugs": "баги",
            "fix": "фикс",
            "hotfix": "хотфикс",
            "feature": "фича",
            "issue": "ишью",
            "ticket": "тикет",
        },
    ),
    (
        ("trading", "crypto", "forex", "stocks", "finance", "investing"),
        {
            "bull": "быки",
            "bullish": "бычий",
            "bear": "медведи",
            "bearish": "медвежий",
            "long": "лонг",
            "go long": "открыть лонг",
            "short": "шорт",
            "go short": "открыть шорт",
            "pump": "памп",
            "pumping": "пампит",
            "dump": "дамп",
            "dumping": "дампит",
            "to the moon": "в космос",
            "moon": "в космос",
            "hodl": "ходлить",
            "hold": "холд",
            "bag holder": "бэг-холдер",
            "bagholder": "бэг-холдер",
            "rekt": "слит",
            "get rekt": "слиться",
            "liquidated": "ликвиднули",
            "liquidation": "ликвидация",
            "margin call": "маржин-колл",
            "stop loss": "стоп-лосс",
            "stop-loss": "стоп-лосс",
            "take profit": "тейк-профит",
            "tp": "ТП",
            "sl": "СЛ",
            "leverage": "плечо",
            "leveraged": "с плечом",
            "fomo": "FOMO",
            "fud": "FUD",
            "dyor": "DYOR",
            "whale": "кит",
            "whales": "киты",
            "paper hands": "бумажные руки",
            "diamond hands": "бриллиантовые руки",
            "rug pull": "раг-пулл",
            "rugged": "ругнули",
            "ath": "ATH",
            "atl": "ATL",
            "all-time high": "исторический максимум",
            "all time high": "исторический максимум",
            "dca": "DCA",
            "dollar cost average": "усреднение",
            "swing trade": "свинг-трейд",
            "day trade": "дейтрейд",
            "scalp": "скальп",
            "scalping": "скальпинг",
            "position": "позиция",
            "entry": "вход",
            "exit": "выход",
            "breakout": "пробой",
            "reversal": "разворот",
            "correction": "коррекция",
            "pullback": "откат",
            "support": "поддержка",
            "resistance": "сопротивление",
            "volume": "объём",
            "liquidity": "ликвидность",
            "order book": "стакан",
            "spread": "спред",
            "slippage": "проскальзывание",
            "altcoin": "альткоин",
            "altcoins": "альткоины",
            "alts": "альты",
            "shitcoin": "шиткоин",
            "memecoin": "мемкоин",
            "gem": "гема",
            "moonshot": "мунсhot",
            "bear market": "медвежий рынок",
            "bull market": "бычий рынок",
            "accumulation": "аккумуляция",
            "distribution": "дистрибуция",
        },
    ),
    (
        ("gaming", "esports", "game", "gameplay", "streaming"),
        {
            "frag": "фраг",
            "kill": "килл",
            "death": "смерть",
            "ko": "ко",
            "headshot": "хедшот",
            "respawn": "респ",
            "camper": "кэмпер",
            "camping": "кэмпить",
            "rush": "раш",
            "rushing": "рашить",
            "nerf": "нерф",
            "nerfed": "занерфили",
            "buff": "баф",
            "buffed": "забафали",
            "meta": "мета",
            "op": "имбовый",
            "overpowered": "имбовый",
            "noob": "нуб",
            "tryhard": "трайхард",
            "lag": "лаг",
            "lagging": "лагает",
            "carry": "кэрри",
            "carrying": "кэрри",
            "grind": "гринд",
            "grinding": "гриндить",
            "loot": "лут",
            "drop": "дроп",
            "raid": "рейд",
            "boss": "босс",
            "boss fight": "босс-файт",
            "dungeon": "данж",
            "pve": "PvE",
            "pvp": "PvP",
            "fps": "FPS",
            "rpg": "RPG",
            "moba": "MOBA",
            "nerf hammer": "нерф-молот",
            "patch": "патч",
            "patched": "запатчили",
            "dlc": "DLC",
            "gg": "GG",
            "ez": "EZ",
            "clutch": "клатч",
            "clutching": "клатчить",
        },
    ),
]


def _load_user_glossary() -> dict:
    """Load user-editable glossary overrides from disk if available.

    Format: JSON at <project_root>/presets/user_glossary.json
      {
        "domains": [
          {
            "name": "BJJ additions",
            "triggers": ["bjj", "jiu-jitsu"],
            "target_lang": "ru",
            "terms": { "leg drag": "лег-драг", ... }
          },
          ...
        ]
      }

    Precedence: user entries override built-in when the same term appears
    with matching triggers + target_lang.  Missing file = empty dict.
    Invalid JSON is logged but not fatal (server keeps running with
    built-in glossary only).
    """
    try:
        # Try project root relative to this file (tachidubb/pipeline/translator.py
        # → tachidubb/presets/user_glossary.json)
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        path = os.path.join(root, "presets", "user_glossary.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning(f"[glossary] Failed to load user_glossary.json: {e}")
        return {}


def _find_applicable_glossary(context_hint: str, target_lang: str) -> dict:
    """Scan context_hint for domain keywords; return merged glossary dict.

    Precedence (later overrides earlier):
      1. Built-in _DOMAIN_GLOSSARIES_RU (ships with code)
      2. User overrides from presets/user_glossary.json (editable via UI)
    """
    hint = (context_hint or "").lower()
    merged = {}

    # Layer 1: built-in glossaries (currently RU-only)
    if target_lang == "ru":
        for keywords, terms in _DOMAIN_GLOSSARIES_RU:
            if any(k in hint for k in keywords):
                merged.update(terms)

    # Layer 2: user overrides — works for any target language
    user_data = _load_user_glossary()
    for domain in user_data.get("domains", []):
        if domain.get("target_lang", "ru") != target_lang:
            continue
        triggers = [t.lower() for t in domain.get("triggers", [])]
        if triggers and any(t in hint for t in triggers):
            terms = domain.get("terms", {})
            if isinstance(terms, dict):
                merged.update(terms)

    return merged


def _build_batch_prompt(src: str, tgt: str, numbered: str,
                        context_hint: str = "",
                        preceding_context: list = None,
                        target_lang_code: str = "ru") -> str:
    context_block = ""
    if context_hint and context_hint.strip():
        context_block = (
            f"CONTEXT: {context_hint.strip()}\n"
            f"Use domain-appropriate vocabulary, slang, and technical terms for this context. "
            f"Translate specialized terminology naturally and idiomatically for {tgt} speakers.\n\n"
        )
    # Glossary: if context_hint mentions a known domain, inject preferred
    # term mappings. Gemma will follow these 80%+ of the time, vs ~50%
    # when left to its own judgement on loan-word terminology.
    glossary = _find_applicable_glossary(context_hint, target_lang_code)
    glossary_block = ""
    if glossary:
        # Only show terms that ACTUALLY appear in the current batch — keeps
        # prompt short and focuses the model's attention on relevant terms.
        numbered_lower = numbered.lower()
        relevant = {k: v for k, v in glossary.items() if k in numbered_lower}
        if relevant:
            pairs = "\n".join(f"  {k} → {v}" for k, v in sorted(relevant.items()))
            glossary_block = (
                f"DOMAIN GLOSSARY (use these exact {tgt} terms for matching {src} words):\n"
                f"{pairs}\n\n"
            )
    # Preceding context: last few EN→translated lines from earlier batches.
    # This helps Gemma preserve continuity for pronouns, tense, named
    # entities ("Nicky" stays "Nicky", not "Ники" in one batch and
    # "Никки" in the next), and mid-thought continuations. Critically
    # it's shown as REFERENCE ONLY — we don't want to the model to
    # re-emit these lines.
    preceding_block = ""
    if preceding_context:
        pairs = "\n".join(
            f"  {src}: {p['src']}\n  {tgt}: {p['tgt']}"
            for p in preceding_context[-4:]  # last 4 previous lines
        )
        preceding_block = (
            f"EARLIER DIALOGUE (for continuity reference — do NOT re-translate these):\n"
            f"{pairs}\n\n"
        )
    return (
        f"You are a professional {tgt} voice-dubbing translator working on a VIDEO DUB.\n"
        f"Every line must fit the same time slot as the original spoken line.\n\n"
        f"Output ONLY numbered translations. No reasoning, no preamble.\n\n"
        f"{context_block}"
        f"{glossary_block}"
        f"{preceding_block}"
        f"Each input line is formatted: [N] (DURATION_SECONDS) TEXT\n"
        f"The duration tells you how many seconds of screen time are available.\n"
        f"At natural {tgt} speaking pace (~2.5 syllables/second), your translation "
        f"MUST be deliverable within that duration. {tgt} often expands vs English — "
        f"you MUST compress when needed: drop fillers, use shorter synonyms, prefer "
        f"concise phrasing, avoid wordy constructions.\n\n"
        f"Translate every numbered line from {src} to {tgt}.\n\n"
        "STRICT RULES:\n"
        f"1. Every line MUST be FULLY translated into {tgt}. "
        f"Never leave any word, phrase, or fragment in {src} or any other language. "
        f"Even filler words, exclamations, names of moves/brands: render them in natural {tgt} "
        f"(transliterate proper nouns using {tgt} script, KEEPING THE SAME SPELLING as in EARLIER DIALOGUE if present).\n"
        "2. Output ONLY numbered translated lines. No explanations, no thinking, no commentary. "
        "Do NOT include the duration in output.\n"
        f"3. TIMING: Match the given duration. If a line is (1.5s), the {tgt} translation "
        f"must fit in ~1.5 seconds at normal speaking pace. If (5.0s), use the full time but don't pad.\n"
        f"4. Use natural spoken {tgt} that a dubbing actor would say out loud — "
        f"not written/literary style, but conversational speech.\n"
        "5. Preserve meaning, emotion, tone, and register of the original speaker.\n"
        "6. If an earlier line ended mid-thought, continue the natural flow (don't repeat content).\n"
        f"7. Return the same number of lines as the input, in the same order, numbered [1], [2], etc.\n\n"
        f"{src} lines (with target durations):\n{numbered}\n\n"
        f"{tgt} translations (numbered, same count, NO durations):"
    )


def _build_single_prompt(src: str, tgt: str, text: str, context_hint: str = "") -> str:
    context_block = ""
    if context_hint and context_hint.strip():
        context_block = f"Context: {context_hint.strip()}. Use appropriate terminology.\n"
    return (
        f"Translate this {src} sentence to {tgt}. "
        f"Output ONLY the {tgt} translation — no reasoning, no explanation, just the translation. "
        f"Every word must be in {tgt} (transliterate proper nouns).\n"
        f"{context_block}\n"
        f"{src}: {text}\n\n"
        f"{tgt}:"
    )


async def _warmup_ollama(url: str, model: str, timeout: float = 60.0) -> bool:
    """Send a trivial request to Ollama so it loads the model into RAM.
    Uses /api/chat with empty system so Gemma4 doesn't start thinking
    (which would make warmup take 30+ sec even for "hi").
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{url}/api/chat", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a translator. Reply with one word."},
                    {"role": "user", "content": "Say OK"},
                ],
                "stream": False,
                "options": {"num_predict": 3},
                "keep_alive": "10m",
            })
            return r.status_code == 200
    except Exception as e:
        log.warning(f"Ollama warmup failed ({type(e).__name__}: {e}); "
                    f"will rely on per-request timeout")
        return False


async def _call_ollama(url: str, model: str, prompt: str,
                      timeout: float = 240.0) -> str:
    """Call Ollama /api/chat with streaming enabled.

    Why /api/chat instead of /api/generate: Gemma4 has a thinking/reasoning
    mode controlled by the system prompt. Thinking is enabled by including
    a <|think|> token at system-prompt level. We leave it OUT to force
    direct-answer mode. /api/generate conflates system and user in one
    string and makes Gemma more likely to start reasoning on its own.
    With proper system role we get deterministic "answer only" behavior.

    Why streaming: thinking models can sit silent for 30-120s emitting
    internal chain-of-thought. Streaming lets us see tokens as they
    arrive, so we know the model is alive rather than hung.
    """
    # Adaptive timeout: overall is hard cap, per-chunk read is generous
    # because reasoning models can spend up to 2 min internally even when
    # thinking is off (they may still compute latent reasoning tokens).
    t_start = time.time()
    chunks = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=120.0)) as c:
        async with c.stream("POST", f"{url}/api/chat", json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    # No <|think|> token = direct answer mode for Gemma4.
                    # This is the ONLY reliable way per Gemma4 docs.
                    "content": (
                        "You are a professional translator for video dubbing. "
                        "Output ONLY the requested translation. "
                        "Do NOT explain, reason, or show any thinking. "
                        "Do NOT add preamble or commentary. "
                        "Just the translated lines, numbered as asked."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
                "top_p": 0.9,
            },
        }) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", errors="replace")[:500]
                if "model" in body.lower() and ("not found" in body.lower() or
                                                  "does not exist" in body.lower()):
                    raise RuntimeError(
                        f"Ollama model '{model}' not installed. "
                        f"Run: ollama pull {model}  (or pick another model in UI)"
                    )
                raise RuntimeError(
                    f"Ollama HTTP {r.status_code}: {body[:200]}"
                )
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # /api/chat delivers tokens inside message.content (not .response)
                msg = obj.get("message") or {}
                tok = msg.get("content", "")
                if tok:
                    chunks.append(tok)
                if obj.get("done"):
                    break
            if not chunks:
                raise RuntimeError(
                    f"Ollama returned empty response after "
                    f"{time.time() - t_start:.1f}s (possible model crash)"
                )
    return "".join(chunks)


async def _call_ollama_legacy(url: str, model: str, prompt: str,
                               timeout: float = 240.0) -> str:
    """Original non-streaming version (kept as fallback, not used by default)."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
                "top_p": 0.9,
            },
        })
        if r.status_code != 200:
            # Give a useful error instead of generic HTTPStatusError
            body = r.text[:500]
            if "model" in body.lower() and ("not found" in body.lower() or
                                              "does not exist" in body.lower()):
                raise RuntimeError(
                    f"Ollama model '{model}' not installed. "
                    f"Run: ollama pull {model}  (or pick another model in UI)"
                )
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {body}")
        return r.json().get("response", "")


async def ensure_model_available(url: str, model: str) -> None:
    """Quick preflight: verify the requested Ollama model is installed
    BEFORE starting the translation loop. Saves 4+ minute timeouts."""
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{url}/api/tags")
            if r.status_code != 200:
                raise RuntimeError(f"Ollama not reachable at {url} "
                                   f"(HTTP {r.status_code})")
            tags = r.json().get("models", [])
            names = {t.get("name", "") for t in tags}
            # Ollama model names may be like "gemma4:e4b" or "gemma4:e4b-instruct"
            # Match by prefix.
            if model in names:
                return
            if any(n.startswith(model) or n.split(":")[0] == model.split(":")[0]
                   for n in names):
                return
            installed = ", ".join(sorted(names)) or "(none)"
            raise RuntimeError(
                f"Ollama model '{model}' not installed.\n"
                f"Installed models: {installed}\n"
                f"Fix: run `ollama pull {model}` OR pick an installed model in UI."
            )
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama not reachable at {url}: {e}")


def _parse_numbered(text: str, count: int) -> dict:
    """Parse numbered lines [1] text / 1. text / 1) text etc."""
    out = {}
    # [N] format
    for m in re.finditer(r"\[(\d+)\]\s*(.+)", text):
        out[int(m.group(1))] = m.group(2).strip()
    if len(out) >= max(1, count // 2):
        return out
    # N. / N) / N] / N: / N-
    out = {}
    for line in text.split("\n"):
        m = re.match(r"^\s*(\d+)[\.\)\]\-:]\s*(.+)", line.strip())
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    if out:
        return out
    # Fallback: bare non-empty lines in order
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return {i + 1: lines[i] for i in range(min(count, len(lines)))}


def _strip_junk(s: str) -> str:
    s = s.strip().strip("\"'`")
    s = re.sub(r"^[\d\.\)\]\-:\s]+", "", s)
    return s.strip()


async def translate_segments(segments: list[dict],
                              source_lang: str,
                              target_lang: str,
                              model: str = "gemma4:e4b",
                              url: str = "",
                              context_hint: str = "",
                              progress_callback=None) -> list[dict]:
    """Translate all segments via Ollama with dubbing-optimized prompts + retry.

    context_hint: optional free-text hint describing the video's domain
        (e.g. "BJJ jiu-jitsu seminar", "cooking show", "tech podcast").
        Prepended to the prompt to improve terminology/slang translation.
    progress_callback: optional fn(done_batches, total_batches, eta_seconds)
        called after each batch. Used to surface translation progress to
        the UI so long runs don't look hung.
    """
    target_url = url or OLLAMA_URL
    src = lang_name(source_lang)
    tgt = lang_name(target_lang)

    # Preflight: fail fast if the model isn't installed (otherwise user
    # stares at "Translating..." for 4 minutes before a cryptic error).
    await ensure_model_available(target_url, model)

    # Warm up Ollama BEFORE starting the real translations. If the model
    # isn't already loaded in VRAM, the first /api/generate call has to
    # load 4+ GB from disk — which eats into our batch timeout and can
    # cause the first batch to falsely "hang". A tiny warmup call with
    # num_predict=1 takes ~15-30s on cold load but completes BEFORE we
    # start timing batches, so the user's "batch 1/N" starts from a warm
    # model and gets proper ETAs.
    log.info(f"Warming up Ollama model '{model}'...")
    t_warm = time.time()
    warmed = await _warmup_ollama(target_url, model, timeout=120.0)
    if warmed:
        log.info(f"Ollama warmed in {time.time() - t_warm:.1f}s")
    else:
        log.warning("Ollama warmup did not succeed; first batch may be slow")

    batch_size = 5
    translated: list[dict] = []
    total_batches = (len(segments) + batch_size - 1) // batch_size
    _t_start = time.time()

    for i in range(0, len(segments), batch_size):
        _t_batch_start = time.time()
        batch = segments[i:i + batch_size]
        # Include per-segment duration so Gemma aims for matching length
        def _fmt(j, s):
            dur = float(s.get("end", 0)) - float(s.get("start", 0))
            dur = max(0.3, dur)  # sanity floor
            return f"[{j+1}] ({dur:.1f}s) {s['text']}"
        numbered = "\n".join(_fmt(j, s) for j, s in enumerate(batch))

        # Preceding context: last few successfully-translated lines so
        # Gemma keeps tense, pronoun reference, and named entity spelling
        # consistent across batches. Critical for multi-speaker dialogue
        # where references like "he / him / the guy" cross batch bounds.
        # IMPORTANT: we cap by character budget (not just count) so long
        # segments don't blow through Gemma's context window and cause
        # silent timeouts. ~600 chars = roughly 150 tokens, safe.
        preceding = None
        if translated:
            CTX_CHAR_BUDGET = 600
            picks = []
            budget = CTX_CHAR_BUDGET
            # Walk backwards: prefer most recent, stop when budget exhausted
            for s in reversed(translated[-6:]):  # at most 6 candidates
                src_len = len(s.get("text", ""))
                tgt_len = len(s.get("translated_text", s["text"]))
                cost = src_len + tgt_len
                if cost > budget and picks:
                    break
                picks.append({
                    "src": s["text"],
                    "tgt": s.get("translated_text", s["text"]),
                })
                budget -= cost
            preceding = list(reversed(picks))  # chronological again

        prompt = _build_batch_prompt(src, tgt, numbered, context_hint,
                                     preceding_context=preceding,
                                     target_lang_code=target_lang)

        # Try batch call up to 2 times, then fallback to single-line mode.
        # Ollama sometimes times out under load (especially with large
        # context windows from preceding_context). A second attempt with
        # lower temperature + shorter context often succeeds.
        raw = None
        last_error = None
        for attempt in range(2):
            try:
                raw = await _call_ollama(target_url, model, prompt)
                break
            except Exception as e:
                # Serialize the exception robustly — some httpx exceptions
                # have empty str() and need repr() to be useful. Also include
                # the type so we can tell timeout vs connection vs HTTP error.
                err_type = type(e).__name__
                err_msg = str(e) or repr(e) or "<no message>"
                last_error = f"{err_type}: {err_msg}"
                if attempt == 0:
                    log.warning(
                        f"Translation batch {i // batch_size + 1}/{total_batches} "
                        f"attempt 1 failed ({last_error}); retrying without "
                        f"preceding context..."
                    )
                    # Rebuild prompt WITHOUT preceding context — it's the
                    # most likely culprit (context window overflow)
                    prompt = _build_batch_prompt(src, tgt, numbered, context_hint,
                                                 target_lang_code=target_lang)

        if raw is None:
            log.error(
                f"Translation batch {i // batch_size + 1}/{total_batches} "
                f"FAILED after retry: {last_error}. "
                f"Falling back to per-line translation..."
            )
            # Per-line fallback: one Ollama call per segment (slower but
            # much more robust — smaller context = less likely to timeout)
            for seg in batch:
                try:
                    single = await _call_ollama(
                        target_url, model,
                        _build_single_prompt(src, tgt, seg["text"], context_hint),
                        timeout=60.0,
                    )
                    single_text = _clean_response(single).strip()
                    # Take first non-empty line
                    for line in single_text.split("\n"):
                        cleaned = _strip_junk(line)
                        if cleaned:
                            translated.append({**seg, "translated_text": cleaned})
                            break
                    else:
                        translated.append({**seg, "translated_text": seg["text"]})
                except Exception as se:
                    log.error(
                        f"  per-line fallback failed for seg: "
                        f"{type(se).__name__}: {se or repr(se)}"
                    )
                    # Keep source text as last-resort "translation"
                    translated.append({**seg, "translated_text": seg["text"]})
            log.info(
                f"Translated batch {i // batch_size + 1}/{total_batches} "
                f"(per-line fallback)"
            )
            continue

        text = _clean_response(raw)
        line_map = _parse_numbered(text, len(batch))

        # Check each segment; retry any that look untranslated
        for j, seg in enumerate(batch):
            t = line_map.get(j + 1, "").strip()
            if not t or _looks_untranslated(t, target_lang):
                try:
                    single = await _call_ollama(
                        target_url, model,
                        _build_single_prompt(src, tgt, seg["text"], context_hint),
                        timeout=90.0,
                    )
                    single = _clean_response(single)
                    for line in single.split("\n"):
                        cleaned = _strip_junk(line)
                        if cleaned and not _looks_untranslated(cleaned, target_lang):
                            t = cleaned
                            break
                    else:
                        # Nothing better; keep whatever we had
                        pass
                except Exception as e:
                    log.warning(f"Retry failed for seg {i + j}: {e}")

            translated.append({**seg, "translated_text": t or seg["text"]})

        batch_idx = i // batch_size + 1
        batch_elapsed = time.time() - _t_batch_start
        total_elapsed = time.time() - _t_start
        # ETA = avg batch time × remaining batches
        remaining = total_batches - batch_idx
        avg_per_batch = total_elapsed / batch_idx
        eta_sec = int(remaining * avg_per_batch)
        eta_str = ""
        if remaining > 0:
            eta_str = f" ETA ~{eta_sec // 60}m {eta_sec % 60}s"
        log.info(
            f"Translated batch {batch_idx}/{total_batches} "
            f"({batch_elapsed:.0f}s){eta_str}"
        )
        if progress_callback:
            try:
                progress_callback(batch_idx, total_batches, eta_sec)
            except Exception as e:
                log.debug(f"progress_callback raised: {e}")

    return translated
