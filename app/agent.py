import json

from groq import Groq
from psycopg.rows import dict_row
from pydantic import BaseModel, ValidationError

from .config import GROQ_API_KEY, LLM_MODEL
from .db import get_conn
from .search import search_jobs

client = Groq(api_key=GROQ_API_KEY)

SYSTEM = (
    "You are a job-search assistant. Use search_jobs to find roles; never invent jobs. "
    "Use save_job when the user wants to save one and list_saved_jobs to show saved ones. "
    "When you are done, call respond with a short message and the ids of the jobs you recommend."
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
    _tool(
        "respond",
        "Final answer to the user. Always finish with this.",
        {"message": {"type": "string"}, "job_ids": {"type": "array", "items": {"type": "integer"}}},
        ["message"],
    ),
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
    msgs = [{"role": "system", "content": SYSTEM}] + history[-20:] + [{"role": "user", "content": text}]
    final: AgentReply | None = None

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:  # model answered in plain text: use it as the final answer
            final = AgentReply(message=msg.content or "Sorry, I have no answer.")
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

            if tc.function.name == "respond":
                try:
                    final = AgentReply(**args)
                except ValidationError:
                    final = AgentReply(message="Sorry, I couldn't format that answer.")
                out = "ok"
            else:
                try:
                    out = json.dumps(_run_tool(tc.function.name, args, session_id), default=str)
                except Exception as e:  # return errors to the model so it can recover
                    out = json.dumps({"error": str(e)})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})

        if final:
            break

    if final is None:
        final = AgentReply(message="Sorry, I couldn't finish that. Try rephrasing.")

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final.message})
    return final