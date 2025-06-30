import os
import json
from collections.abc import AsyncGenerator

from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume


from .utils.utils import  server


from .agents.addresses_agent import key_addresses
from .agents.key_officers_agent import key_officers
from .agents.executive_summary_agent import executive_summary
from .agents.shareholders import octagon_holdings



@server.agent(name="company_profile", metadata=Metadata(ui={"type": "hands-off"}))
async def company_profile(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:

    company_name = str(input[-1]).strip()

    # --- helper to build the “user” message ----------------------------------
    def make_user_msg(text: str) -> Message:
        """Return an ACP Message equivalent to: {'role':'user', 'content': text}"""
        return Message(parts=[MessagePart(content=text)])
    
    # 1) run executive_summary ----------------------------------------------------
    executive_summary_chunks: list[str] = []
    async for part in executive_summary([make_user_msg(company_name)], context):
        if isinstance(part, MessagePart):
            executive_summary_chunks.append(part.content)           # MessagePart → grab its .content
            
    # 2) run key_addresses ----------------------------------------------------
    addr_chunks: list[str] = []
    async for part in key_addresses([make_user_msg(company_name)], context):
        if isinstance(part, MessagePart):
            addr_chunks.append(part.content)           # MessagePart → grab its .content

    # 3) run key_officers -----------------------------------------------------
    officer_chunks: list[str] = []
    async for part in key_officers([make_user_msg(company_name)], context):
        if isinstance(part, MessagePart):
            officer_chunks.append(part.content)

    # 3) run Octagon Holdings Agent -----------------------------------------------------
    holdings_chunks: list[str] = []
    async for part in octagon_holdings([make_user_msg(company_name)], context):
        if isinstance(part, MessagePart):
            holdings_chunks.append(part.content)
            
    # 4) stream combined result back to caller -------------------------------
    combined = "\n\n".join(["".join(executive_summary_chunks),
                            "".join(addr_chunks), 
                            "".join(officer_chunks),
                            "".join(holdings_chunks)
                            ])
    yield MessagePart(content=combined)


def run() -> None:  # local dev helper
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8080)))


if __name__ == "__main__":
    run()
    
    
    

