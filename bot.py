import argparse
import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
IMAGE_DIR = OUTPUTS / "images"
QUOTE_DIR = OUTPUTS / "quotes"
PROMPT_DIR = OUTPUTS / "prompts"
STYLE_FILE = ROOT / "style_config.json"


def load_style() -> Dict[str, Any]:
    """Load reusable brand/style settings."""
    if not STYLE_FILE.exists():
        raise FileNotFoundError("Missing style_config.json")
    return json.loads(STYLE_FILE.read_text(encoding="utf-8"))


def slugify(text: str, max_len: int = 60) -> str:
    """Create clean filenames from quote text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len].strip("-") or "quote"


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    QUOTE_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)


def generate_quotes(
    client: OpenAI,
    theme: str,
    count: int,
    quote_style: str,
    phrase_format: str | None = None,
) -> List[str]:
    """
    Generate motivational quote ideas.

    Example phrase_format:
    - "Grace is..."
    - "Strength is..."
    - "Discipline is..."
    """
    format_instruction = ""
    if phrase_format:
        format_instruction = (
            f'\nEvery quote must begin with this exact phrase style: "{phrase_format}". '
            "Keep the quote as one sentence."
        )

    prompt = f"""
Generate {count} original motivational quotes for social media.

Theme: {theme}
Tone/style: {quote_style}
Rules:
- Keep each quote short, strong, and clear.
- No author names.
- No numbering in the quote itself.
- Avoid clichés when possible.
- Make each quote image-worthy.
{format_instruction}

Return only a JSON array of strings.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    text = response.output_text.strip()

    try:
        quotes = json.loads(text)
    except json.JSONDecodeError:
        # Fallback parser in case the model returns plain lines.
        quotes = [
            line.strip("-•0123456789. ").strip()
            for line in text.splitlines()
            if line.strip()
        ]

    quotes = [q.strip().strip('"') for q in quotes if q.strip()]
    return quotes[:count]


def build_image_prompt(quote: str, style: Dict[str, Any]) -> str:
    """Turn a quote into a reusable image prompt."""
    return f"""
Create a polished cinematic motivational poster for social media.

Main quote text to display clearly:
"{quote}"

Visual style:
{style["visual_style"]}

Character:
{style["character"]}

Scene:
{style["scene"]}

Typography:
{style["typography"]}

Composition:
{style["composition"]}

Quality rules:
{style["quality_rules"]}

Important:
- Make the quote large, readable, and correctly spelled.
- Do not add extra unrelated words unless they are small supporting design elements.
- Keep the finished image clean, premium, and suitable for Facebook, Instagram, and Threads.
""".strip()


def generate_image(
    client: OpenAI,
    prompt: str,
    output_path: Path,
    model: str = "gpt-image-1",
    size: str = "1024x1536",
) -> None:
    """
    Generate an image and save it to output_path.

    Note:
    OpenAI image model availability can change. If your account supports a newer image model,
    update the --image-model argument when running the bot.
    """
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_path.write_bytes(image_bytes)


def save_batch_metadata(
    batch_id: str,
    theme: str,
    quotes: List[str],
    prompts: List[Dict[str, str]],
) -> None:
    quote_file = QUOTE_DIR / f"{batch_id}_quotes.json"
    prompt_file = PROMPT_DIR / f"{batch_id}_prompts.json"

    quote_file.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "theme": theme,
                "quotes": quotes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prompt_file.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "prompts": prompts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_bot(args: argparse.Namespace) -> None:
    load_dotenv()
    ensure_dirs()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and add your API key."
        )

    client = OpenAI(api_key=api_key)
    style = load_style()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generating {args.count} quotes for theme: {args.theme}")

    quotes = generate_quotes(
        client=client,
        theme=args.theme,
        count=args.count,
        quote_style=args.quote_style,
        phrase_format=args.phrase_format,
    )

    prompts = []

    if args.quotes_only:
        save_batch_metadata(batch_id, args.theme, quotes, prompts)
        print(f"Saved quotes only: outputs/quotes/{batch_id}_quotes.json")
        return

    for index, quote in enumerate(quotes, start=1):
        prompt = build_image_prompt(quote, style)
        prompts.append({"quote": quote, "prompt": prompt})

        filename = f"{batch_id}_{index:02d}_{slugify(quote)}.png"
        output_path = IMAGE_DIR / filename

        print(f"[{index}/{len(quotes)}] Generating image: {quote}")
        generate_image(
            client=client,
            prompt=prompt,
            output_path=output_path,
            model=args.image_model,
            size=args.size,
        )
        print(f"Saved: {output_path.relative_to(ROOT)}")

    save_batch_metadata(batch_id, args.theme, quotes, prompts)
    print("Batch complete.")
    print(f"Quotes saved to: outputs/quotes/{batch_id}_quotes.json")
    print(f"Prompts saved to: outputs/prompts/{batch_id}_prompts.json")
    print(f"Images saved to: outputs/images/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate motivational quotes and images using the OpenAI API."
    )

    parser.add_argument(
        "--theme",
        default="grace, discipline, emotional control, and self-improvement",
        help="Theme for the quote batch.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of quotes/images to generate.",
    )
    parser.add_argument(
        "--quote-style",
        default="bold, reflective, strong, concise, social-media ready",
        help="Tone/style for the generated quotes.",
    )
    parser.add_argument(
        "--phrase-format",
        default=None,
        help='Optional phrase pattern, such as "Grace is..." or "Strength is..."',
    )
    parser.add_argument(
        "--image-model",
        default="gpt-image-1",
        help="Image model to use. Example: gpt-image-1",
    )
    parser.add_argument(
        "--size",
        default="1024x1536",
        help="Image size. Common portrait size: 1024x1536",
    )
    parser.add_argument(
        "--quotes-only",
        action="store_true",
        help="Only generate quotes, not images.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_bot(parse_args())
