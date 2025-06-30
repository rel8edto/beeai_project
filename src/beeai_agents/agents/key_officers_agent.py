import traceback       
from collections.abc import AsyncGenerator


from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume
from beeai_framework.backend.message import UserMessage

from ..utils.utils import  format_officer, fetch_company_data_from_pds, server, chat_model


@server.agent(name="key_officers", metadata=Metadata(ui={"type": "hands-off"}))
async def key_officers(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    • Pull every director / officer object from PDS.
    • If none are found, fall back to the model’s own knowledge.
    • Stream the **Key Officers** paragraph back to the user.
    """
    company_name = str(input[-1]).strip()
    try:
        data = await fetch_company_data_from_pds(company_name)

        rec = next((r for r in data.get("result", []) if r.get("kind") == "Company"), {})

        raw_directors = rec.get("directors", [])  # <- every officer record
        officers = {format_officer(director) for director in raw_directors if director}  # dedupe

        # ------------------------------------------------------------------ #
        # 1️⃣  NO DATA → ask the LLM to rely on its own knowledge            #
        # ------------------------------------------------------------------ #
        if not officers:
            # prompt = (
            #     f"You are writing the **Key Officers** – {company_name}** section of an "
            #     "equity-research report.\n"
                # "• Use your own knowledge to identify the current CEO and other C-suite "
                # f"leaders of {company_name}.\n"
                # "• In 3-4 sentences, state each person’s role and—briefly—their background.\n"
                # "• If you are unsure who the executives are, say so clearly instead of guessing."
            # )
            prompt = f"""
                You are writing the **Key Officers** section of an "
                "equity-research report.\n"
                
                **Write exactly in this order:**
                1. A heading line: **Key Officers**
                2. Use your own knowledge to identify the current CEO and other C-suite "
                f"leaders of {company_name}.\n"
                "• In 3-4 sentences, state each person’s role and—briefly—their background.\n"
                "• If you are unsure who the executives are, say so clearly instead of guessing."
                """

            resp = await chat_model.create(messages=[UserMessage(prompt)])
            yield MessagePart(content=resp.get_text_content())
            return

        # keep a stable order for the LLM
        officer_list = list(officers)

        # ── craft LLM prompt ──────────────────────────────────────────────────────
        
                # ── craft LLM prompt ──────────────────────────────────────────────────────
        example = """
        **Key Officers**
        ONEREP LLC was first established in Eastern Europe in 2015 and incorporated in
        the US in October 2018. The Founder & CEO, Dzmitry Shelest …  The current CTO,
        Mikalai Shershan …  The SVP of Strategic Partnerships, Mark Kapczynski …
        """.strip()

        system_guard = f"""
        **Key Officers-{company_name}
        Begin with the bold heading **Key Officers-{company_name}**.

        You are writing the **Key Officers – {company_name}** paragraph for an analyst
        report.

        • Base every statement **only** on the officers supplied.  
        • Do **not** copy wording from the sample – it is illustrative only.  
        • Do **not** invent biographies or commentary that is not present in the data.
        """.strip()


        prompt = (
            f"{system_guard}\n"
            "### Sample (format ONLY – do not repeat wording):\n"
            f"{example}\n\n"
            "### Write the paragraph for the following officers:\n"
            + "\n".join(f"• {o}" for o in officer_list)
        )

        response = await chat_model.create(messages=[UserMessage(prompt)])
        summary = response.get_text_content()
        yield MessagePart(content=summary)

    except Exception as exc:
        tb = traceback.format_exc()
        yield MessagePart(
            content=(
                f"Sorry, I couldn’t fetch data for “{company_name}”.\n\n"
                f"**Error:** {exc}\n\n"
                f"```traceback\n{tb}```"
        )
)
   