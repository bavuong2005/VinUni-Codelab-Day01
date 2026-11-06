"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from pathlib import Path
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Estimated costs per 1M input and output tokens (USD).
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    
    return text, latency, usage



# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    usage_metadata = response.usage_metadata
    usage = {
        "input_tokens": usage_metadata.prompt_token_count if usage_metadata else 0,
        "output_tokens": usage_metadata.candidates_token_count if usage_metadata else 0,
    }
    return response.text or "", latency, usage

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time
    
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }
    
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 3.8 Flash (gemini-3.8-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    def call_and_price(call, model: str, price_key: str) -> dict:
        try:
            response, latency, usage = call(prompt, model=model)
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            rates = PRICING_1M_TOKENS[price_key]
            cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
            return {
                "response": response,
                "latency": latency,
                "cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as exc:
            return {
                "response": f"API call failed: {exc}",
                "latency": 0.0,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    # Keep each provider independent so one account's billing error cannot skip Gemini.
    return {
        "gpt4o": call_and_price(call_openai, OPENAI_MODEL, OPENAI_MODEL),
        "gpt4o_mini": call_and_price(call_openai, OPENAI_MINI_MODEL, OPENAI_MINI_MODEL),
        "gemini_flash": call_and_price(call_gemini, GEMINI_MODEL, GEMINI_MODEL),
    }

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 3.8 Flash.
    Maintains the last 3 turns of conversation history for context.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Error] Set GEMINI_API_KEY in the project-root .env file.\033[0m")
        return
        
    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")
    
    # Store history as a list of dictionaries with 'role' and 'text'
    # Gemini 3.8 Flash structure generally expects 'user' and 'model' roles.
    history = []
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\033[93mGoodbye!\033[0m")
                break
                
            # Compile messages from history (trimmed to last 3 turns / 6 messages)
            messages_to_send = []
            for h in history[-6:]:
                messages_to_send.append(h)
            messages_to_send.append({"role": "user", "text": user_input})
            
            print("\033[92mGemini:\033[0m ", end="", flush=True)
            
            full_reply = ""
            try:
                # Option A: New GenAI client streaming
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key)
                
                # Format messages for the new SDK
                # For chat history, new SDK prefers client.chats.create() or genai.types.Content structure
                formatted_contents = []
                for msg in messages_to_send:
                    formatted_contents.append(
                        types.Content(
                            role=msg["role"],
                            parts=[types.Part.from_text(text=msg["text"])]
                        )
                    )
                    
                response_stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=formatted_contents
                )
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text
                    
            except ImportError:
                # Option B: Fallback to legacy google-generativeai stream
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_inst = genai.GenerativeModel(GEMINI_MODEL)
                
                # Format legacy messages: role is 'user' or 'model'
                legacy_contents = []
                for msg in messages_to_send:
                    legacy_contents.append({
                        "role": msg["role"] if msg["role"] == "user" else "model",
                        "parts": [msg["text"]]
                    })
                    
                response_stream = model_inst.generate_content(legacy_contents, stream=True)
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text
                    
            print("\n")
            
            # Record the turn in history
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})
            
        except KeyboardInterrupt:
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error Calling API]: {e}\033[0m\n")

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    if last_exception:
        raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for prompt in prompts:
        try:
            comp = compare_models(prompt)
            comp["prompt"] = prompt
            results.append(comp)
        except Exception as e:
            # Return partial mock results for error resistance during testing
            results.append({
                "prompt": prompt,
                "gpt4o": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gpt4o_mini": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gemini_flash": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
            })
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | ---: | :---: | ---: |",
    ]

    def clean_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    for result in results:
        prompt = clean_cell(result.get("prompt", ""))
        models = (
            ("GPT-4o", "gpt4o"),
            ("GPT-4o-Mini", "gpt4o_mini"),
            ("Gemini-Flash", "gemini_flash"),
        )

        for model_name, result_key in models:
            stats = result.get(result_key)
            if not isinstance(stats, dict):
                continue

            response = clean_cell(stats.get("response", ""))
            if len(response) > 50:
                response = response[:47] + "..."

            latency = float(stats.get("latency", 0.0))
            input_tokens = stats.get("input_tokens", 0)
            output_tokens = stats.get("output_tokens", 0)
            cost = float(stats.get("cost", 0.0))
            table_lines.append(
                f"| {prompt} | {model_name} | {response} | {latency:.2f}s | "
                f"{input_tokens}/{output_tokens} | ${cost:.6f} |"
            )

    return "\n".join(table_lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
