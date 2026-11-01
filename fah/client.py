"""Async client for an OpenAI-compatible /v1/chat/completions endpoint (furiosa-llm, vLLM, the public RNGD endpoint)."""
import asyncio
import os
import random
import time

import httpx

DEFAULT_BASE_URL = "https://endpoint.access.furiosa.dev/v1"


def load_dotenv(path):
    """KEY=VALUE lines -> os.environ (existing variables win). No dependency on python-dotenv."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Client:
    def __init__(self, model, base_url=None, api_key=None, concurrency=16, timeout=3600.0, max_retries=6, params=None):
        self.model = model
        self.base_url = (base_url or os.environ.get("FAH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("FAH_API_KEY")
        if not self.api_key:
            raise SystemExit("FAH_API_KEY is not set (put it in .env or the environment)")
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.params = dict(params or {})          # sampling + extra body, merged into every request
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0),
                                      limits=httpx.Limits(max_connections=concurrency + 4),
                                      headers={"Authorization": f"Bearer {self.api_key}"})

    async def close(self):
        await self.http.aclose()

    async def chat(self, messages, **override):
        body = {"model": self.model, "messages": messages, **self.params, **override}
        body = {k: v for k, v in body.items() if v is not None}
        last = None
        async with self.sem:
            for attempt in range(self.max_retries):
                t0 = time.time()
                try:
                    r = await self.http.post(f"{self.base_url}/chat/completions", json=body)
                    if r.status_code in (429, 500, 502, 503, 504):
                        last = f"HTTP {r.status_code}: {r.text[:300]}"
                    elif r.status_code != 200:
                        # 4xx other than 429 will not get better on retry (e.g. prompt longer than the context)
                        return dict(error=f"HTTP {r.status_code}: {r.text[:500]}", latency=time.time() - t0)
                    else:
                        j = r.json()
                        ch = j["choices"][0]
                        msg = ch.get("message") or {}
                        return dict(
                            content=msg.get("content") or "",
                            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                            finish_reason=ch.get("finish_reason"),
                            usage=j.get("usage") or {},
                            latency=time.time() - t0,
                        )
                except (httpx.TransportError, httpx.TimeoutException, ValueError, KeyError) as e:
                    last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        return dict(error=f"gave up after {self.max_retries} attempts: {last}")
