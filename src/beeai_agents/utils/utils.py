import os, httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from beeai_framework.backend.chat import ChatModel
from acp_sdk.server import Server

from dotenv import load_dotenv
load_dotenv()

WATSONX_URL=os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID=os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY=os.getenv("WATSONX_APIKEY")

server = Server()

chat_model = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct",
                                 {
            "project_id": WATSONX_PROJECT_ID,
             "api_key": WATSONX_API_KEY,
             "base_url": WATSONX_URL,
             })
    

# ──────────────────────────────────────────────────────────────────────────────
# Helper – prettify one address dict → single-line string
# ──────────────────────────────────────────────────────────────────────────────
def format_addr(addr: dict) -> str:
    """Return “Street, City, ST, ZIP” (skip blanks)."""
    # print("address inside format_addr func======>: ", addr)
    parts = [
        addr.get("addressLine") or addr.get("Address_Line"),
        addr.get("city") or addr.get("City"),
        addr.get("region") or addr.get("State"),
        addr.get("postal") or addr.get("Zip"),
    ]
    return ", ".join(filter(None, parts))

# ──────────────────────────────────────────────────────────────────────────────
# NEW agent – returns a paragraph titled **Key Officers**
# ──────────────────────────────────────────────────────────────────────────────
def format_officer(director: dict) -> str:
    """
    Produce a single-line representation:
    “<full name> (first seen 2020, relType DIRECTED_BY)”   — tweak as you like
    """
    name = (
        director.get("name")
        or director.get("primaryName", {}).get("fullName")
        or "Unnamed individual"
    )
    dob = director.get("dateOfBirth")
    dob_str = f", born {dob}" if dob else ""
    rel = director.get("relType") or ""
    return f"{name}{dob_str} – {rel.replace('_', ' ').title()}"

    
# ──────────────────────────────────────────────────────────────────────────────
# Helper – fetch from PDS
# ──────────────────────────────────────────────────────────────────────────────
# 20 s read timeout; 10 s connect timeout
PDS_TIMEOUT = httpx.Timeout(40.0)

RETRY_POLICY  = dict(
    wait      = wait_exponential(multiplier=0.5, max=8),
    stop      = stop_after_attempt(3),
    retry     = retry_if_exception_type((httpx.ReadTimeout, httpx.ReadError)),
    reraise   = True,
)
@retry(**RETRY_POLICY)
async def fetch_company_data_from_pds(
    company_name: str,
    state: str = "NY",
) -> dict:
    token = os.getenv(
        "PDS_TOKEN",
        "Bearer xeWiXeVqMwAB39wrg/HG4fFFA6bZtkf0vIT8kczVRAbyHqqXHkqTub481r/HvtLqC4",
    )
    url = (
        "https://api.rel8ed.to/companies/search?"
        f"searchType=graphOnly&companyName={company_name}&stateProvince={state}"
    )
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=PDS_TIMEOUT) as client:
        resp = await client.post(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

def is_us(addr: dict) -> bool:
    """
    Return True if the raw PDS address is in the United States.
    PDS stores country codes in `country` (alpha-2) or sometimes `Country`.
    """
    return (addr.get("country") or addr.get("Country")) == "US"
