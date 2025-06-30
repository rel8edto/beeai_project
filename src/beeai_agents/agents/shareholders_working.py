# --- Required imports ---
import os
from collections.abc import AsyncGenerator

# Use the ASYNC version of the OpenAI client for compatibility with the async framework
from openai import AsyncOpenAI
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume

# Assuming 'server' is defined in your utils, as in your original code
from ..utils.utils import server
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

ROOT = Path(__file__).resolve().parents[3] 
# load_dotenv(ROOT / "./env")
load_dotenv(find_dotenv())

OCTAGON_API_KEY = os.getenv("OCTAGON_API_KEY")

# Initialize the ASYNC client to work with your 'async def' agent.
octagon_client = AsyncOpenAI(
    api_key=OCTAGON_API_KEY,
    base_url="https://api-gateway.octagonagents.com/v1")


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
    
    company_name = str(input[-1]).strip()
    if company_name=="google":
        company_name="GOOG"
    elif company_name=="Microsoft":
        company_name="MSFT"
    elif company_name=="Tesla":
        company_name="TSLA"
    elif company_name=="Apple":
        company_name="AAPL"
    if not company_name:
        yield MessagePart(content="Please provide a company name or stock ticker.")
        return

    try:
        # 2. Construct the query for the Octagon Holdings Agent.
        # This can be customized (e.g., asking for specific quarters or holders).
        octagon_query = f"Get a summary of institutional positions for {company_name} for Q4 of 2024."

        # 3. Call the Octagon API using the async client.
        response = await octagon_client.responses.create(
            model="octagon-holdings-agent",
            input=octagon_query
        )

        # 4. Extract the content and format the response.
        # print("response from Octagon API :", response)
        msg = response.output[0]
        chunks         = msg.content 
        summary_text   = "".join(c.text for c in chunks)
        annotations    = [a for c in chunks for a in c.annotations]


        # Build a comprehensive response string with sources for better context.
        full_response = summary_text

        if annotations:
            full_response += "\n\n**Sources:**"
            for annotation in annotations:
                # The annotation object provides details about the source filing or URL.
                full_response += f"\n- {annotation.order}. {annotation.name}: {annotation.url}"

        # 5. Yield the final, formatted result back to the user.
        yield MessagePart(content=full_response)

    except Exception as exc:
        # 6. Handle potential API errors gracefully.
        yield MessagePart(
            content=f"Sorry, I couldn’t fetch holdings data for “{company_name}” from Octagon: {exc}"
        )
