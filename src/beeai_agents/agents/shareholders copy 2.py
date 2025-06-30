# --- Required imports ---
import os, json
from collections.abc import AsyncGenerator
import textwrap
# Use the ASYNC version of the OpenAI client for compatibility with the async framework
from openai import AsyncOpenAI
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume

from beeai_framework.backend.message import UserMessage

# Assuming 'server' is defined in your utils, as in your original code
from ..utils.utils import server, chat_model
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import Any 

ROOT = Path(__file__).resolve().parents[3] 
print("ROOT 3====>", ROOT)
# load_dotenv(ROOT / "./env")
load_dotenv(find_dotenv())

OCTAGON_API_KEY = os.getenv("OCTAGON_API_KEY")

# Initialize the ASYNC client to work with your 'async def' agent.
octagon_client = AsyncOpenAI(
    api_key=OCTAGON_API_KEY,
    base_url="https://api-gateway.octagonagents.com/v1")

NAME_TO_TICKER: dict[str, str] = {
    "google": "GOOG",
    "alphabet": "GOOG",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "apple": "AAPL",
    # add more mappings here ↓
    "facebook": "META",
}
# ----------------------------------------------------------------------------- 
# helper: make numeric values pretty
# -----------------------------------------------------------------------------
def fmt(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f} B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    return f"{n:,}"

# --- The New Agent Implementation ---
@server.agent(name="octagon_holdings", metadata=Metadata(ui={"type": "hands-off"}))
async def octagon_holdings(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    • Pulls institutional holdings data for a company using the Octagon Holdings Agent.
    • Formats the response, including sources, and streams it back to the user.
    """
    # --- 1. resolve user input to a ticker -----------------------------------
    raw_query: str = str(input[-1]).strip()
    company_name = raw_query
    if not raw_query:
        yield MessagePart("Please provide a company name or ticker symbol.")
        return

    # normalise for dictionary lookup
    ticker = NAME_TO_TICKER.get(raw_query.lower(), raw_query.upper())

    # from here on we ALWAYS use `ticker`
    # -------------------------------------------------------------------------

    try:
        # 2. Build Octagon query exactly as before (use `ticker`)
        hist_q = "2024-09-30"
        curr_q = "2024-12-31"
        oct_query = (
            f"Get institutional holdings for {ticker} "
            f"for {curr_q} (current) and {hist_q} (previous)."
        )

        resp = await octagon_client.responses.create(
            model="octagon-holdings-agent",
            input=oct_query,
        )

        # Octagon’s answer is “assistant” content containing JSON
        txt = "".join(part.text for part in resp.output[0].content).strip()
        raw_json = json.loads(txt)

        # Expect most-recent row first
        current, *_ = raw_json
        # Try to grab the matching previous row
        previous = next((r for r in raw_json if r["date"] == hist_q), None)

        # ── 3. convert to bullet list for LLM ────────────────────────────────
        bullets = [
            f"Quarter end date: {current['date']}",
            f"Number of reporting institutions: {fmt(current['investorsHolding'])}"
            + (
                f" (was {fmt(previous['investorsHolding'])} last quarter)"
                if previous else ""
            ),
            f"Total 13-F shares reported: {fmt(current['numberOf13Fshares'])}",
            f"Total market value reported: ${fmt(current['totalInvested'])}",
            f"Ownership percentage: {current['ownershipPercent']:.2f} %",
            f"New positions opened: {fmt(current['newPositions'])}",
            f"Positions increased: {fmt(current['increasedPositions'])}",
            f"Positions reduced: {fmt(current['reducedPositions'])}",
            f"Positions closed: {fmt(current['closedPositions'])}",
            f"Put / call ratio: {current['putCallRatio']:.2f}",
        ]

        # ── 4. prompt our chat model to narrate those bullets ────────────────
        system_guard = textwrap.dedent(
            f"""
            You are writing the **Key Shareholders – {ticker}** paragraph for an
            equity-research report.

            • Begin with the bold heading **Key Shareholders – {ticker}**.      # ← FIX
            • Summarise the figures below in 3-5 sentences. Mention quarter-on-
              quarter changes only when > 5 %.
            • Base every statement solely on the bullets. Avoid hype.
            """
        )

        prompt = system_guard + "\n\n### Data:\n" + "\n".join(f"• {b}" for b in bullets)
        llm_resp   = await chat_model.create(messages=[UserMessage(prompt)])
        paragraph  = llm_resp.get_text_content()
        yield MessagePart(paragraph)

    except Exception as exc:
        # 6. Handle potential API errors gracefully.
        yield MessagePart(
            content=f"Sorry, I couldn’t fetch holdings data for “{company_name}” from Octagon: {exc}"
        )
