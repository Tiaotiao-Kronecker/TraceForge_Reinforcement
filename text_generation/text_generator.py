import os
import re
import json
import base64
import argparse
import mimetypes
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import Content, Part, Schema
from openai import OpenAI

load_dotenv()

# Roles considered "task-related" for keywords
_ALLOWED_ROLES_FOR_KEYWORDS = {
    "tool", "primary_target", "secondary_target", "destination", "obstacle"
}

# Very light action removal for fallback only
_REMOVE_ACTIONS_LIGHT = {"reach", "grasp", "pre_grasp", "manipulation"}

# Broader filters used by _is_objectish (kept for robustness in fallback)
_ACTIONY_TOKENS = {
    "reach","reaching","grasp","grasping","pick","pickup","pick_up",
    "place","placing","put","put_down","putdown","move","moving",
    "manipulation","pre_grasp","pregrasp","hold","holding","rotate","rotating",
    "open","close","closing","opening","align","aligning","push","pull",
    "transfer","stack","unstack","lift","lower"
}
_RELATION_TOKENS = {
    "left_of","right_of","on_top_of","under","in_front_of","behind",
    "near","next_to","into","onto","from","to","in","out"
}
_STATE_TOKENS = {"open","closed","empty","full","on","off"}

def _is_objectish(token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    if t in _ACTIONY_TOKENS or t in _RELATION_TOKENS or t in _STATE_TOKENS:
        return False
    if t.startswith(("reach","grasp","pick","place","put","move","manipul","hold","rotate",
                     "open","close","align","push","pull","transfer","stack","unstack","lift","lower")):
        return False
    return any(ch.isalpha() for ch in t)

def _to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _get_google_api_key(explicit_api_key: str = None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")

def _get_openai_api_key(explicit_api_key: str = None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("OPENAI_API_KEY")

def _encode_image(path: str):
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime or "image/jpeg", b64

def _build_schema() -> Schema:
    return Schema(
        type="object",
        properties={
            "task_instructions": Schema(
                type="object",
                properties={
                    "general": Schema(type="string"),
                    "detailed": Schema(type="string"),
                    "multistep": Schema(type="array", items=Schema(type="string")),
                },
                required=["general", "detailed", "multistep"],
            ),
            "keywords": Schema(
                type="array",
                items=Schema(type="string"),
                description="3–7 lower_snake_case tokens of task-related OBJECTS (no actions/relations/states)"
            ),
            "focus_objects": Schema(
                type="array",
                items=Schema(
                    type="object",
                    properties={
                        "name": Schema(type="string"),
                        "role": Schema(type="string", enum=[
                            "primary_target", "secondary_target", "tool", "destination", "obstacle", "context"
                        ]),
                        "visual_cues": Schema(type="string"),
                        "reasoning": Schema(type="string"),
                    },
                    required=["name", "role", "visual_cues", "reasoning"]
                ),
                description="2–6 objects relevant to the task"
            ),
        },
        required=["task_instructions", "keywords", "focus_objects"],
    )

def _build_openai_schema():
    """
    Define a simplified function-calling schema for OpenAI models.
    Only require 3 short instructions (instruction_1, instruction_2, instruction_3).
    """
    return {
        "type": "function",
        "function": {
            "name": "generate_task_description",
            "description": "Generate exactly three imperative instructions from the given frames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction_1": {
                        "type": "string",
                        "description": "Concise imperative command (≤ 10 words)."
                    },
                    "instruction_2": {
                        "type": "string",
                        "description": "Explicit stepwise command (≤ 20 words)."
                    },
                    "instruction_3": {
                        "type": "string",
                        "description": "Natural request form (≤ 15 words)."
                    }
                },
                "required": ["instruction_1", "instruction_2", "instruction_3"]
            },
        },
    }

def _build_prompt() -> str:
    with open(Path(__file__).parent / "text_generator_prompt.txt", "r") as f:
        return f.read()

def _generate_with_google(image_paths: list, *, api_key: str = None, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """Generate task description using Google Gemini."""
    client = Client(api_key=_get_google_api_key(api_key))

    image_parts = []
    for path in image_paths:
        mime, data = _encode_image(path)
        image_parts.append(Part(inline_data={"mime_type": mime, "data": data}))

    prompt_text = _build_prompt()
    content_parts = [Part(text=prompt_text)] + image_parts

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            Content(role="user", parts=content_parts)
        ],
    )

    result_text = resp.text
    if not result_text or not result_text.strip():
        raise ValueError("API returned empty or null response")
    return _parse_json_response(result_text)

def _generate_with_openai(image_paths: list, *, api_key: str = None, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Generate task description using OpenAI GPT."""
    client = OpenAI(api_key=_get_openai_api_key(api_key))
    prompt_text = _build_prompt()

    content = [{"type": "text", "text": prompt_text}]
    for path in image_paths:
        mime, data = _encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"}
        })

    messages = [{"role": "user", "content": content}]

    try:
        params = {
            "model": model_name,
            "messages": messages,
            "tools": [_build_openai_schema()],
            "tool_choice": "auto",
            "max_completion_tokens": 4000
        }
        response = client.chat.completions.create(**params)
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            return json.loads(function_args)
    except Exception:
        pass

    params = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": 4000
    }
    response = client.chat.completions.create(**params)
    result_text = response.choices[0].message.content
    if not result_text or not result_text.strip():
        raise ValueError("OpenAI returned empty or null response")
    return _parse_json_response(result_text)

def _parse_json_response(result_text: str) -> Dict[str, Any]:
    """Parse JSON response from either provider."""
    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        fence = re.search(r"```json\s*([\s\S]*?)\s*```", result_text, re.IGNORECASE)
        if fence:
            parsed = json.loads(fence.group(1))
        else:
            start = result_text.find("{")
            end = result_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(result_text[start:end+1])
            else:
                raise ValueError("Could not find valid JSON in response")
    return _post_process_response(parsed)

def _post_process_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process the parsed response to normalize format."""
    instr_keys = {"instruction_1", "instruction_2", "instruction_3"}
    if instr_keys.issubset(parsed.keys()):
        return {k: parsed[k] for k in ("instruction_1", "instruction_2", "instruction_3")}

    # Legacy schema path (task_instructions/keywords/focus_objects)
    td = parsed.get("task_instructions", {})
    steps = td.get("multistep")
    if isinstance(steps, str):
        items = [s.strip(" -\t") for s in re.split(r"(?:\n+)|(?:^\s*\d+\.\s*)|(?:\s*\d+\.\s*)", steps) if s and s.strip()]
        parsed["task_instructions"]["multistep"] = items[:6] if items else []

    task_kw = []
    if isinstance(parsed.get("focus_objects"), list):
        for fo in parsed["focus_objects"]:
            role = (fo or {}).get("role", "")
            name = (fo or {}).get("name", "")
            if role in _ALLOWED_ROLES_FOR_KEYWORDS and name:
                nrm = _to_snake(name)
                if nrm and _is_objectish(nrm):
                    task_kw.append(nrm)

    raw_kw = parsed.get("keywords", []) or []
    if len(task_kw) < 3 and raw_kw:
        fb = []
        for k in raw_kw:
            nrm = _to_snake(k)
            if nrm and (nrm not in _REMOVE_ACTIONS_LIGHT) and _is_objectish(nrm):
                fb.append(nrm)
        task_kw = _dedupe_keep_order(task_kw + fb)

    task_kw = _dedupe_keep_order(task_kw)[:7]
    parsed["keywords"] = task_kw

    return parsed

def generate_task_description(
    image_paths: list,
    *,
    api_key: str = None,
    model_name: str = "gemini-2.5-flash",
    provider: str = "google"
) -> Dict[str, Any]:
    """
    Generate task description from images using specified provider.

    Args:
        image_paths: List of image file paths
        api_key: API key for the provider
        model_name: Model name to use
        provider: Either "google" or "openai"

    Returns:
        Dictionary containing task descriptions, keywords, and focus objects
    """
    if not image_paths:
        raise ValueError("At least one image path must be provided")

    for i, path in enumerate(image_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image {i} not found: {path}")

    provider = provider.lower()

    if provider == "google":
        return _generate_with_google(image_paths, api_key=api_key, model_name=model_name)
    elif provider == "openai":
        return _generate_with_openai(image_paths, api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'google' or 'openai'")

def _main_cli():
    parser = argparse.ArgumentParser(description="Generate task description JSON from images")
    parser.add_argument("--images", required=True, nargs='+', type=str, help="Paths to frame images")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--provider", type=str, default="google", choices=["google", "openai"],
                       help="Provider to use: google (Gemini) or openai (GPT)")
    args = parser.parse_args()


    result = generate_task_description(
        args.images,
        api_key=args.api_key,
        model_name=args.model,
        provider=args.provider
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"INFO: Saved result to {args.out}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    _main_cli()
