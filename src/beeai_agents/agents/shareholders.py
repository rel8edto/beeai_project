# --- Required imports ---
import os, json, textwrap
from collections.abc import AsyncGenerator
from typing import Any

import traceback       
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

ROOT = Path(__file__).resolve().parents[3] 
print("ROOT 3====>", ROOT)
# load_dotenv(ROOT / "./env")
load_dotenv(find_dotenv())

OCTAGON_API_KEY = os.getenv("OCTAGON_API_KEY")

# Initialize the ASYNC client to work with your 'async def' agent.
octagon_client = AsyncOpenAI(
    api_key=OCTAGON_API_KEY,
    base_url="https://api-gateway.octagonagents.com/v1")

print("octagon_client ===>", octagon_client)

NAME_TO_TICKER = {
    "google": "GOOG", "alphabet": "GOOG",
    "microsoft": "MSFT", "tesla": "TSLA",
    "apple": "AAPL",   "facebook": "META",
}

def fmt(n: int | float | None) -> str:
    """Pretty-print large numbers."""
    if n is None: return "n/a"
    if abs(n) >= 1_000_000_000: return f"{n/1_000_000_000:.1f} B"
    if abs(n) >= 1_000_000:     return f"{n/1_000_000:.1f} M"
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
    # 1. Get the company name or ticker from the user's last message.
   
    raw = str(input[-1]).strip()
    company_name= raw
    if not raw:
        yield MessagePart("Please provide a company name or ticker symbol.")
        return

    ticker = NAME_TO_TICKER.get(raw.lower(), raw.upper())

    hist_q, curr_q = "2024-09-30", "2024-12-31"
    oct_query = (
        f"Get a summary of institutional positions for {ticker} "
        f"for Q4 2024 (current) and Q3 2024 (previous)."
    )
    
    try:
        # 3. Call the Octagon API using the async client.
        response = await octagon_client.responses.create(
            model="octagon-holdings-agent",
            input=oct_query
        )
        
       # assistant-content → plain JSON string
        txt = "".join(part.text for part in response.output[0].content).strip()
        rows = json.loads(txt)
        

    except Exception as exc:
        tb = traceback.format_exc()
        yield MessagePart(
            content=(
                f"Sorry, I couldn’t fetch data for “{company_name}”.\n\n"
                f"**Error:** {exc}\n\n"
                f"```traceback\n{tb}\n```"
            )
        )
   

     # ── 2. pick current / previous rows --------------------------------------
    current, *_ = rows
    previous = next((r for r in rows if r["date"] == hist_q), None)

    # ── 3. build bullet list for the LLM -------------------------------------
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
    
    # ── 4. craft prompt & ask your chat model --------------------------------
    system_guard = textwrap.dedent(f"""
        You are writing the **Key Shareholders** paragraph for an
        equity-research report.

        • Begin with the bold heading **Key Shareholders**.
        • Summarise the figures below in 3-5 sentences.
        • Mention quarter-on-quarter changes only when the change is > 5 %.
        • Base every statement solely on the bullets. Avoid hype.
    """).strip()

    prompt = system_guard + "\n\n### Data:\n" + "\n".join(f"• {b}" for b in bullets)
    
    
    llm_resp = await chat_model.create(messages=[UserMessage(prompt)])   # ← as requested
    summary  = llm_resp.get_text_content()

    yield MessagePart(content = summary)
