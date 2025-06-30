import json
from collections.abc import AsyncGenerator

import httpx
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from beeai_framework.backend.message import UserMessage


from ..utils.utils import format_addr, server, chat_model, fetch_company_data_from_pds


# ──────────────────────────────────────────────────────────────────────────────
# Bee agent
# ──────────────────────────────────────────────────────────────────────────────
@server.agent(name= "key_addresses",metadata=Metadata(ui={"type": "hands-off"}))
async def key_addresses(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    • Pull every usable address from PDS
    • Ask the LLM to craft the “Key Addresses” paragraph
    • Stream that paragraph back to the user
    """
    company_name = str(input[-1]).strip()
    print("===>key_addresses started for===> :", company_name)

    try:
        data = await fetch_company_data_from_pds(company_name)
        
        rec = next(
            (r for r in data.get("result", []) if r.get("kind") == "Company"),{},)
        
        # print("rec :", rec)
        raw_addrs = rec.get("addresses", [{}])
        # print("raw_addrs :", raw_addrs)
        addrs = {
            format_addr(addr)
            for addr in raw_addrs
            if not addr.get("suppress") and (addr.get("addressLine") or addr.get("Address_Line"))
        }
        # print("Final addresses :", addrs)
        if not addrs:
            yield MessagePart(content="No addresses found.")
            return

        
        name = rec.get("name", company_name)

        # pick the first string as HQ, keep order stable for LLM prompt
        addr_list = list(addrs)
        hq = addr_list[0]
        others = "; ".join(addr_list[1:]) if len(addr_list) > 1 else ""
        
        # ── craft LLM prompt ──────────────────────────────────────────────────────
        example = """
                **Key Addresses **
                ONEREP LLC’s key registered address in the US is 1750 Tysons Blvd, Suite 1500, McLean, VA 22102. This location is listed as their headquarters and is in a multi-unit commercial building with numerous other businesses.
                ONEREP LLC is also legally registered at 7288 Hanover Green Dr, Mechanicsville, VA 23111; 2108 N St #C, Sacramento, CA, 95816; and 5441 S Macadam Ave Suite R, Portland, OR 97239. Nuwber, Inc. is also registered at 5441 S Macadam Ave Suite R, Portland, OR 97239 address. They do not appear to have any physical office space in California or Oregon.
                """.strip()
        

        prompt = (
            f"Here is an example:\n{example}\n\n"
            "Now, write a similar paragraph titled **Key Addresses** for this company using the addresses below.\n"
            f"• Headquarters: {hq}\n"
            f"• Other addresses: {others or 'None'}"
        )

        
        # chat_input = {"messages": [UserMessage(prompt)]
        #             #   , "stream": False
        #               }

        # response   = await chat_model.create(chat_input) 
        response = await chat_model.create(messages=[UserMessage(prompt)])
        summary = response.get_text_content()  # plain-text result
        yield MessagePart(content=summary)


    except Exception as exc:
        yield MessagePart(
            content=f"Sorry, I couldn’t fetch data for “{company_name}”: {exc}"
        )
        


