# src/api.py  (adjust import paths if your package name differs)
from fastapi import FastAPI, Request, HTTPException
from acp_sdk import MessagePart, Message
from typing import List
import asyncio

from src.beeai_agents.agent import company_profile          # <- your existing file

app = FastAPI()

@app.post("/query")
async def query_endpoint(req: Request):
    body = await req.json()
    company = body.get("company")
    if not company:
        raise HTTPException(400, detail="Field 'company' is required")

    # Wrap text into ACP Message format expected by company_profile
    msg_in: List[Message] = [Message(parts=[MessagePart(content=company)])]

    chunks: List[str] = []
    async for part in company_profile(msg_in, context=None):
        if isinstance(part, MessagePart):
            chunks.append(part.content)

    return {"answer": "".join(chunks)}
