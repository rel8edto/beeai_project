# shareholders.py  – Octagon holdings agent
import os, json, re, textwrap, traceback
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume
from beeai_framework.backend.message import UserMessage

from ..utils.utils import server, chat_model
from dotenv import load_dotenv, find_dotenv

# ─── ENV / CLIENTS ───────────────────────────────────────────────────────
load_dotenv(find_dotenv())
OCTAGON_API_KEY = os.getenv("OCTAGON_API_KEY")

octagon_client = AsyncOpenAI(
    api_key=OCTAGON_API_KEY,
    base_url="https://api-gateway.octagonagents.com/v1",
)

# quick alias map
NAME_TO_TICKER = {
    "google": "GOOG", "alphabet": "GOOG",
    "microsoft": "MSFT", "tesla": "TSLA",
    "apple": "AAPL",   "facebook": "META",
}

# ─── HELPERS ─────────────────────────────────────────────────────────────
def fmt(n: int | float | None) -> str:
    if n is None: return "n/a"
    if abs(n) >= 1_000_000_000: return f"{n/1_000_000_000:.1f} B"
    if abs(n) >= 1_000_000:     return f"{n/1_000_000:.1f} M"
    return f"{n:,}"

async def lookup_ticker(company: str) -> str | None:
    """
    Resolve `company` to its primary stock symbol, or return None for
    private / unknown firms.
    """
    # 1️⃣  hard-coded aliases
    if company.lower() in NAME_TO_TICKER:
        return NAME_TO_TICKER[company.lower()]

    # 2️⃣  ask Octagon’s symbol-lookup agent
    query = (f"Return ONLY the primary stock-ticker symbol for the company "
             f"named '{company}'. If it is not publicly traded, reply 'PRIVATE'.")
    try:
        resp = await octagon_client.responses.create(
            model="octagon-stock-data-agent",
            input=query,
        )
        symbol = "".join(p.text for p in resp.output[0].content).strip().upper()
        if symbol and symbol not in {"PRIVATE", "N/A"}:
            return symbol
    except Exception:
        pass  # network / quota / etc.

    return None

# ─── AGENT ───────────────────────────────────────────────────────────────
@server.agent(name="octagon_holdings", metadata=Metadata(ui={"type": "hands-off"}))
async def octagon_holdings(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    · Resolves a ticker, fetches Octagon 13-F holdings, and streams a
      **Key Shareholders** paragraph. Gracefully exits for private firms.
    """
    company = str(input[-1]).strip()
    if not company:
        yield MessagePart(content="Please provide a company name or ticker symbol.")
        return

    # ── 1. ticker resolution ──────────────────────────────────────────
    ticker = await lookup_ticker(company)
    if ticker is None:
        yield MessagePart(content=f"🔎 {company}: company isn’t publicly listed.")
        return

    # ── 2. query holdings agent ───────────────────────────────────────
    oct_query = (f"Get a summary of institutional positions for {ticker} "
                 f"for Q4 2024 (current) and Q3 2024 (previous). Respond in JSON.")
    try:
        resp = await octagon_client.responses.create(
            model="octagon-holdings-agent",
            input=oct_query,
        )
        raw = "".join(p.text for p in resp.output[0].content).strip()
        rows = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        yield MessagePart(
            content=(f"⚠️  No 13-F data available for **{ticker}** "
                     "(possible IPO or thin coverage).")
        )
        return

    # ── 3. pick current / previous rows ───────────────────────────────
    current, *_ = rows
    previous = next((r for r in rows if r["date"] == "2024-09-30"), None)

    # ── 4. bulletise for the LLM ──────────────────────────────────────
    bullets = [
        f"Quarter end date: {current['date']}",
        f"Reporting institutions: {fmt(current['investorsHolding'])}"
        + (f" (was {fmt(previous['investorsHolding'])} last quarter)" if previous else ""),
        f"13-F shares reported: {fmt(current['numberOf13Fshares'])}",
        f"Market value of holdings: ${fmt(current['totalInvested'])}",
        f"Ownership percentage: {current['ownershipPercent']:.2f} %",
        f"New positions: {fmt(current['newPositions'])}",
        f"Positions increased: {fmt(current['increasedPositions'])}",
        f"Positions reduced: {fmt(current['reducedPositions'])}",
        f"Positions closed: {fmt(current['closedPositions'])}",
        f"Put / call ratio: {current['putCallRatio']:.2f}",
    ]

    system_guard = textwrap.dedent("""
        You are writing the **Key Shareholders** paragraph for an
        equity-research report.

        • Begin with the bold heading **Key Shareholders**.
        • Summarise the figures below in 3-5 sentences.
        • Mention quarter-on-quarter changes only when the change is > 5 %.
        • Base every statement solely on the bullets. Avoid hype.
    """).strip()

    prompt = system_guard + "\n\n### Data:\n" + "\n".join(f"• {b}" for b in bullets)
    llm_resp = await chat_model.create(messages=[UserMessage(prompt)])
    summary  = llm_resp.get_text_content()

    yield MessagePart(content=summary)
