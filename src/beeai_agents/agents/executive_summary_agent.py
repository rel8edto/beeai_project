
from collections.abc import AsyncGenerator


from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume
from beeai_framework.backend.message import UserMessage

from ..utils.utils import  server



@server.agent(name="executive_summary", metadata=Metadata(ui={"type": "hands-off"}))
async def executive_summary(
    input: list[Message],
    context: Context,
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Return a one-paragraph executive summary in the exact wording you supplied."""
    company = str(input[-1]).strip().upper()

    summary = (
        f"**Executive Summary**\n"
        f"A deep dive was completed for {company} to review its relationships and "
        f"identify foreign linkages. Using several data sources and open-source "
        f"intelligence, a relationship summary has been compiled for {company}’s "
        f"key officers, shareholders, addresses, controversies and financial "
        f"indicators, with the following key takeaways."
    )

    yield MessagePart(content=summary)
