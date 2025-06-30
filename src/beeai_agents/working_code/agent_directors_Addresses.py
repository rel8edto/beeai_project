import os
import json
from collections.abc import AsyncGenerator

import httpx
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from beeai_framework.backend.chat import ChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel

from beeai_framework.agents.react import ReActAgent

from beeai_framework.backend.message import UserMessage

from dotenv import load_dotenv
load_dotenv()

WATSONX_URL=os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID=os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY=os.getenv("WATSONX_APIKEY")

server = Server()

# LLM_NAME = os.getenv("ollama:ibm-granite/granite3.3")
# chat_model = ChatModel.from_name("ollama:ibm-granite/granite3.3")  
chat_model = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct",
                                 {
            "project_id": WATSONX_PROJECT_ID,
             "api_key": WATSONX_API_KEY,
             "base_url": WATSONX_URL,
             })
        

# ──────────────────────────────────────────────────────────────────────────────
# Helper – fetch from PDS
# ──────────────────────────────────────────────────────────────────────────────
async def fetch_company_data_from_pds(
    company_name: str,
    state: str = "NY",
) -> dict:
    token = os.getenv(
        "PDS_TOKEN",
        "Bearer xeWiXeVqMwAB39wrg/HG4fFFA6bZtkf0vIT8kczVRAbyHqqXHkqTub481r/HvtLqC4",
    )
    url = (
        "http://10.8.0.70:8453/companies/search?"
        f"searchType=graphOnly&companyName={company_name}&stateProvince={state}"
    )
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers)
        resp.raise_for_status()
        return resp.json()



@server.agent(name="company_summary", metadata=Metadata(ui={"type": "hands-off"}))
async def company_summary(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    company = str(input[-1]).strip()
    rec     = next((r for r in (await fetch_company_data_from_pds(company)).get("result", []) if r.get("kind") == "Company"), {})
    addr    = rec.get("addresses", [{}])[0]
    line    = ", ".join(filter(None, [addr.get("addressLine"), addr.get("city"), addr.get("region"), addr.get("postal")]))
    with open(f"rec_dump.json", "w") as f:
        json.dump(rec, f, indent=2)
    print("===== directors:", rec.get("directors"))
    directors = ", ".join(d.get("name") for d in rec.get("directors", []) if d.get("name")) or "N/A"

    prompt = (
        "Summarize in one sentence:\n"
        f"Company: {rec.get('name', company)}\n"
        f"Address: {line}\n"
        f"Directors: {directors}"
    )
    resp = await chat_model.create(messages=[UserMessage(prompt)])
    yield MessagePart(content=resp.get_text_content())


def run() -> None:  # local dev helper
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
    
    
    

