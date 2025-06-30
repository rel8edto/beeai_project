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
            if not addr.get("suppress") 
            and (addr.get("addressLine") or addr.get("Address_Line"))
            and "no address line" not in (addr.get("addressLine") 
                                    or addr.get("Address_Line")).lower()
        }
        if not addrs:
            yield MessagePart(content="No addresses found.")
            return

        
        name = rec.get("name", company_name)

        # pick the first string as HQ, keep order stable for LLM prompt
        addr_list = list(addrs)
        print("Number of addresses===> :", len(addr_list))
        hq = addr_list[0]
        us_addrs  = [a for a in addr_list[1:] if ", " in a and " US" in a][:3]
        intl_addrs = [a for a in addr_list[1:] if a not in us_addrs]
        us_block  = "; ".join(us_addrs)  or "None"
        intl_block = "; ".join(intl_addrs) or "None"
        
        # ── craft LLM prompt ──────────────────────────────────────────────────────
        example = """
                **Key Addresses **
                ONEREP LLC’s key registered address in the US is 1750 Tysons Blvd, Suite 1500, McLean, VA 22102. This location is listed as their headquarters and is in a multi-unit commercial building with numerous other businesses.
                ONEREP LLC is also legally registered at 7288 Hanover Green Dr, Mechanicsville, VA 23111; 2108 N St #C, Sacramento, CA, 95816; 
                and 5441 S Macadam Ave Suite R, Portland, OR 97239. Nuwber, Inc. is also registered at 5441 S Macadam Ave Suite R, Portland, 
                OR 97239 address. They do not appear to have any physical office space in California or Oregon.
                """.strip()
        
        system_guard = (
            "You are writing the **Key Addresses – {name}** paragraph for an analyst report.\n"
            "• Mention the headquarters first.\n"
            "• Then highlight up to **three** notable U.S. addresses (if any).\n"
            "• Conclude with 1-2 sentences that state which continents the remaining "
            "addresses cover (infer the continents yourself).\n"
            "• Base every statement **only** on the addresses supplied.\n"
            "• Do not copy wording from the sample.\n"
        ).format(name=name)       

        prompt = (
                f"{system_guard}\n"
                "### Format sample (style only):\n"
                f"{example}\n\n"
                "### Data for the company:\n"
                f"• Headquarters: {hq}\n"
                f"• Key U.S. addresses: {us_block}\n"
                f"• Other global addresses: {intl_block}"
            )
        
        response = await chat_model.create(messages=[UserMessage(prompt)])
        summary = response.get_text_content()
        yield MessagePart(content=summary)


    except Exception as exc:
        yield MessagePart(
            content=f"Sorry, I couldn’t fetch data for “{company_name}”: {exc}"
        )
        


