# =============================================================================
# utils/api_client.py — Async OpenAI-compatible API client
# =============================================================================
import asyncio
from openai import AsyncOpenAI
from config import DMX_API_KEY, DMX_BASE_URL


async def create_async_client():
    """Create an AsyncOpenAI client using config values."""
    print("Initializing API async client...")
    client = AsyncOpenAI(
        api_key=DMX_API_KEY,
        base_url=DMX_BASE_URL,
        timeout=60.0,
    )
    print("API client initialized successfully!")
    return client


async def call_api_with_retry(client, model_name, messages, max_tokens=1000,
                              temperature=0, top_p=1.0, max_retries=10):
    """Call the chat API with exponential backoff retry."""
    retry_count = 0
    while True:
        retry_count += 1
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                raise Exception("API returned empty content")
            if retry_count > 1:
                print(f"  Retry {retry_count} succeeded")
            return content.strip()
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  API call failed (attempt {retry_count}): {error_msg}")
            wait_time = min(2 ** retry_count, 60)
            print(f"  Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)