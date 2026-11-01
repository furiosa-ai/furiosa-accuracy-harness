"""Async client for an OpenAI-compatible /v1/chat/completions endpoint (furiosa-llm, vLLM, the public RNGD endpoint)."""
import asyncio
import json
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
    def __init__(self, model, base_url=None, api_key=None, concurrency=16, timeout=3600.0, max_retries=6, params=None,
                 stream=True):
        self.model = model
        self.stream = stream
        self.base_url = (base_url or os.environ.get("FAH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("FAH_API_KEY")
        if not self.api_key:
            raise SystemExit("FAH_API_KEY is not set (put it in .env or the environment)")
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.params = dict(params or {})          # sampling + extra body, merged into every request
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0, read=600.0),
                                      limits=httpx.Limits(max_connections=concurrency + 4),
                                      headers={"Authorization": f"Bearer {self.api_key}"})

    async def close(self):
        await self.http.aclose()

    async def chat(self, messages, **override):
        """Streams by default: a long reasoning reply (minutes) over a non-streamed request is cut by HTTP gateways
        (HTTP 502/504 after ~5 min on the public endpoint); a stream keeps bytes flowing."""
        body = {"model": self.model, "messages": messages, **self.params, **override}
        body = {k: v for k, v in body.items() if v is not None}
        if self.stream:
            body.update(stream=True, stream_options={"include_usage": True})
        last, long_fails = None, 0
        async with self.sem:
            for attempt in range(self.max_retries):
                if long_fails >= 2:
                    # the same long reply was cut twice after minutes: a proxy/gateway request timeout, not a blip.
                    return dict(error=f"cut twice after a long generation (gateway request timeout?): {last}")
                t0 = time.time()
                try:
                    if self.stream:
                        res = await self._stream(body, t0)
                    else:
                        res = await self._plain(body, t0)
                    if "retry" not in res:
                        return res
                    last = res["retry"]
                except (httpx.TransportError, httpx.TimeoutException, ValueError, KeyError) as e:
                    last = f"{type(e).__name__}: {e} after {time.time() - t0:.0f} s"
                long_fails += time.time() - t0 > 120
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        return dict(error=f"gave up after {self.max_retries} attempts: {last}")

    @staticmethod
    def _status(code, text, t0):
        if code in (429, 500, 502, 503, 504):
            return dict(retry=f"HTTP {code}: {text[:300]} after {time.time() - t0:.0f} s")
        # other 4xx will not get better on retry (e.g. prompt longer than the context)
        return dict(error=f"HTTP {code}: {text[:500]}", latency=time.time() - t0)

    async def _plain(self, body, t0):
        r = await self.http.post(f"{self.base_url}/chat/completions", json=body)
        if r.status_code != 200:
            return self._status(r.status_code, r.text, t0)
        j = r.json()
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        return dict(content=msg.get("content") or "",
                    reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                    finish_reason=ch.get("finish_reason"), usage=j.get("usage") or {}, latency=time.time() - t0)

    async def _stream(self, body, t0):
        content, reasoning, finish, usage, ttft = [], [], None, {}, None
        async with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            if r.status_code != 200:
                return self._status(r.status_code, (await r.aread()).decode(errors="replace"), t0)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                j = json.loads(data)
                if j.get("error"):
                    return dict(retry=f"stream error: {str(j['error'])[:300]}")
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if (c or rc) and ttft is None:
                        ttft = time.time() - t0
                    if c:
                        content.append(c)
                    if rc:
                        reasoning.append(rc)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        if finish is None:
            return dict(retry=f"stream ended without finish_reason after {time.time() - t0:.0f} s")
        return dict(content="".join(content), reasoning="".join(reasoning), finish_reason=finish, usage=usage,
                    latency=time.time() - t0, ttft=ttft)
