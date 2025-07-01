import json
from collections.abc import AsyncGenerator
import traceback         
import httpx
from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from beeai_framework.backend.message import UserMessage


from ..utils.utils import format_addr, is_us, server, chat_model, fetch_company_data_from_pds


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
        print("===>fetching company data from PDS===>")
        data = await fetch_company_data_from_pds(company_name)
        print("===>fetched company data from PDS===>")
        rec = next(
            (r for r in data.get("result", []) if r.get("kind") == "Company"),{},)
        
        print("===>rec length===>:", len(rec))
        # print("rec :", rec)
        raw_addrs = rec.get("addresses", [{}])
        print("===>raw_addrs length===>:", len(raw_addrs))
        # ── 1. Harvest usable addresses and deduplicate  ───────────────────────
        clean_pairs: list[tuple[str, dict]] = []          # (pretty_addr , original_dict)
        seen: set[str] = set()                            # case-insensitive key

        for a in raw_addrs:
            # ignore suppressed / blank / “No Address Line Given” rows
            if a.get("suppress"):
                continue
            line = (a.get("addressLine") or a.get("Address_Line") or "")
            if not line or "no address line" in line.lower():
                continue

            pretty = format_addr(a)                       # normalised printable string
            key = pretty.lower()                         # de-dupe key (case-insensitive)
            if key not in seen:
                clean_pairs.append((pretty, a))
                seen.add(key)
        print("===>number of clean_pairs===>:", len(clean_pairs))
        if not clean_pairs:
            yield MessagePart(content =f"No addresses found.")
            return

        
        # ── 2. Build the U.S. and international buckets (now distinct) ─────────
        us_addrs   = [p for p, meta in clean_pairs if is_us(meta)][:3]         # ≤ 3 distinct
        intl_codes = {
            (meta.get("country") or meta.get("Country") or "").upper()
            for p, meta in clean_pairs if not is_us(meta)
        }

        us_block   = "; ".join(us_addrs) if us_addrs else "None"
        intl_block = ", ".join(sorted(intl_codes)) or "None"

        
        
        print("===>us_block length===>:", len(us_block))
        print("===>intl_block length===>:", len(intl_block))
        
        # ── craft LLM prompt ──────────────────────────────────────────────────────
        sample = """
        **Key Addresses**
        ONEREP LLC maintains several U.S. locations including 1750 Tysons Blvd, McLean, VA; …
        Outside the U.S., additional offices span Europe and Asia-Pacific.
        """.strip()

        # system_guard = f"""
        # You are writing the **Key Addresses – {company_name}** paragraph for an analyst report.

        # • Highlight up to three notable **U.S.** addresses (if any).
        # • Conclude with 1–2 sentences that describe the geographic spread
        #   of the remaining addresses (e.g. “additional offices across Europe and Asia-Pacific”).
        # • Base every statement **only** on the addresses supplied – no inventions, no copy-pasting
        #   from the sample.
        # """.strip()
        # system_guard = f"""
        # You are writing the **Key Addresses** – {company_name}** paragraph for an analyst report.

        # • Quote up to **three** illustrative U.S. addresses verbatim.
        # • Then add 1-2 short sentences describing the *international footprint*,
        # **grouping** the non-U.S. countries I provide into their continents
        # (e.g. “Europe, Asia-Pacific and Africa”). Do **not** list every country.
        # • Base every statement only on the data supplied – no invented sites.
        # """.strip()

        system_guard = f"""
            Begin the paragraph with the bold heading of following text **Key Addresses**.

            • Quote up to **three** illustrative U.S. addresses verbatim.  
            • Add 1-2 concise sentences that describe the international footprint,
            **grouping** the non-U.S. countries I supply into their continents
            (e.g. “Europe, Asia-Pacific and Africa”). Do **not** list every country.  
            • Base every statement strictly on the data provided – no invented sites.
            """.strip()

        prompt = (
            system_guard
            + "\n\n### Data\n"
            + f"• U.S. addresses: {us_block}\n\n\n"
            + f"• Non-U.S. country codes: {intl_block}"
        )
        # prompt = (
        #     system_guard
        #     + "\n\n### Format sample (style only):\n"
        #     + sample
        #     + "\n\n### Data:\n"
        #     + f"• Key U.S. addresses: {us_block}\n"
        #     + f"• Other global addresses: {intl_block}"
        # )

        
        response = await chat_model.create(messages=[UserMessage(prompt)])
        summary = response.get_text_content()
        yield MessagePart(content=summary)


    except Exception as exc:
        tb = traceback.format_exc()
        # Log internally (prints to container logs / Cloud logging)
        print("key_addresses error:", tb, flush=True)
        
        yield MessagePart(
            content=(
                f"Sorry, I couldn’t fetch data for “{company_name}”.\n\n"
                f"**Error:** {exc}\n\n"
                f"```traceback\n{tb}\n```"
        )
        )
        


