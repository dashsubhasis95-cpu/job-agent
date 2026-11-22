import json
import logging
import re

from groq import BadRequestError, Groq
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from .config import GROQ_API_KEY, LLM_MODEL
from .db import get_conn
from .search import search_jobs

log = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

SYSTEM = (
    "You are a job-search assistant. Use search_jobs to find roles; never invent jobs. "
    "Use save_job when the user wants to save one and list_saved_jobs to show saved ones. "
    "When you have what you need, reply in plain text: a short answer that mentions each "
    "recommended job's title, company, location and its id number."
)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _tool(
        "search_jobs",
        "Semantic search over the jobs index. Supports optional location and remote filters.",
        {
            "query": {"type": "string"},
            "location": {"type": "string"},
            "remote_only": {"type": "boolean"},
            "k": {"type": "integer"},
        },
        ["query"],
    ),
    _tool("save_job", "Save a job for the current user.", {"job_id": {"type": "integer"}}, ["job_id"]),
    _tool("list_saved_jobs", "List the current user's saved jobs.", {}, []),
]


class SearchArgs(BaseModel):
    query: str
    location: str | None = None
    remote_only: bool | None = False
    k: int | None = Field(5, ge=1, le=10)


class SaveArgs(BaseModel):
    job_id: int


class AgentReply(BaseModel):
    message: str
    job_ids: list[int] = []


SESSIONS: dict[str, list[dict]] = {}  # per-session memory (plain text turns)
MAX_STEPS = 6


def _run_tool(name: str, args: dict, session_id: str):
    if name == "search_jobs":
        a = SearchArgs(**args)  # validation errors are returned to the model
        return search_jobs(a.query, a.location, bool(a.remote_only), a.k or 5)
    if name == "save_job":
        a = SaveArgs(**args)
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO saved_jobs (user_ref, job_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (session_id, a.job_id),
            )
        return {"saved": a.job_id}
    if name == "list_saved_jobs":
        with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT j.id, j.title, j.company, j.location FROM saved_jobs s "
                "JOIN jobs j ON j.id = s.job_id WHERE s.user_ref = %s ORDER BY s.saved_at DESC",
                (session_id,),
            )
            return cur.fetchall()
    return {"error": f"unknown tool {name}"}


def _call_llm(msgs: list[dict]):
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=msgs,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2500,
            )
        except BadRequestError as e:
            # open models sometimes emit a malformed tool call: just try again
            if "tool_use_failed" in str(e) and attempt < 2:
                log.warning("tool_use_failed, retrying (%d)", attempt + 1)
                continue
            raise


def chat(session_id: str, text: str) -> AgentReply:
    history = SESSIONS.setdefault(session_id, [])
    msgs = [{"role": "system", "content": SYSTEM}] + history[-20:] + [{"role": "user", "content": text}]
    seen: dict[int, dict] = {}  # jobs returned by search during this turn
    final: AgentReply | None = None

    try:
        for _ in range(MAX_STEPS):
            msg = _call_llm(msgs).choices[0].message

            if not msg.tool_calls:  # plain-text answer = done
                final = AgentReply(message=(msg.content or "").strip() or "Sorry, I have no answer.")
                break

            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = _run_tool(tc.function.name, args, session_id)
                    if tc.function.name == "search_jobs":
                        seen.update({r["id"]: r for r in result})
                    out = json.dumps(result, default=str)
                except Exception as e:  # return errors to the model so it can recover
                    out = json.dumps({"error": str(e)})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    except Exception:
        log.exception("agent failed")
        final = AgentReply(message="Sorry, something went wrong. Please try again.")

    if final is None:
        final = AgentReply(message="Sorry, I couldn't finish that. Try rephrasing.")
    elif seen:  # attach the jobs the answer actually refers to
        mentioned = [jid for jid in seen if re.search(rf"\b{jid}\b", final.message)]
        final.job_ids = mentioned or list(seen)[:5]

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final.message})
    return final