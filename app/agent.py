import json

import anthropic
from psycopg.rows import dict_row
from pydantic import BaseModel, ValidationError

from .config import ANTHROPIC_API_KEY, LLM_MODEL
from .db import get_conn
from .search import search_jobs

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM = (
    "You are a job-search assistant. Use search_jobs to find roles; never invent jobs. "
    "Use save_job when the user wants to save one and list_saved_jobs to show saved ones. "
    "When you are done, call respond with a short message and the ids of the jobs you recommend."
)

TOOLS = [
    {
        "name": "search_jobs",
        "description": "Semantic search over the jobs index. Supports optional location and remote filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "location": {"type": "string"},
                "remote_only": {"type": "boolean"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_job",
        "description": "Save a job for the current user.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "list_saved_jobs",
        "description": "List the current user's saved jobs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "respond",
        "description": "Final answer to the user. Always finish with this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "job_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["message"],
        },
    },
]


class AgentReply(BaseModel):
    message: str
    job_ids: list[int] = []


SESSIONS: dict[str, list[dict]] = {}  # per-session memory (plain text turns)
MAX_STEPS = 6


def _run_tool(name: str, args: dict, session_id: str):
    if name == "search_jobs":
        return search_jobs(
            args["query"], args.get("location"), args.get("remote_only", False), args.get("k", 5)
        )
    if name == "save_job":
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO saved_jobs (user_ref, job_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (session_id, args["job_id"]),
            )
        return {"saved": args["job_id"]}
    if name == "list_saved_jobs":
        with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT j.id, j.title, j.company, j.location FROM saved_jobs s "
                "JOIN jobs j ON j.id = s.job_id WHERE s.user_ref = %s ORDER BY s.saved_at DESC",
                (session_id,),
            )
            return cur.fetchall()
    return {"error": f"unknown tool {name}"}


def chat(session_id: str, text: str) -> AgentReply:
    history = SESSIONS.setdefault(session_id, [])
    msgs = history[-20:] + [{"role": "user", "content": text}]
    final: AgentReply | None = None

    for _ in range(MAX_STEPS):
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1000,
            system=SYSTEM,
            tools=TOOLS,
            tool_choice={"type": "any"},  # always call a tool; "respond" ends the loop
            messages=msgs,
        )
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "respond":
                try:
                    final = AgentReply(**block.input)
                except ValidationError:
                    final = AgentReply(message="Sorry, I couldn't format that answer.")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": "ok"})
            else:
                try:
                    out = _run_tool(block.name, block.input, session_id)
                except Exception as e:  # return errors to the model so it can recover
                    out = {"error": str(e)}
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out, default=str),
                    }
                )
        if final:
            break
        msgs.append({"role": "user", "content": results})

    if final is None:
        final = AgentReply(message="Sorry, I couldn't finish that. Try rephrasing.")

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final.message})
    return final
