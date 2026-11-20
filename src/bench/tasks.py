"""Task definitions.

Two suites:

* ``structured`` — deterministic tasks with objective scoring (JSON-schema
  output, tool/function calling, exact-match fields, multi-turn agentic
  workflows).
* ``open`` — open-ended instruction-following; scored by an LLM judge on a
  0-1 rubric per dimension.

Each task is a plain dict so it stays easy to inspect and extend.

Tool tasks (``kind: "tools"``): the ``tools`` key holds real OpenAI
function-calling schemas that are sent to the server; the model's native
function calls (``ChatResult.tool_calls``) are scored exactly. If a server
cannot do native tool calling, the harness falls back to parsing
``{"tool_calls": [...]}`` out of the text content, so servers without native
support remain testable.

Agent tasks (``kind: "agent"``): multi-turn tool-calling workflows. The
harness runs a loop — the model calls tools, the harness executes them
against ``task["env"]`` (a deterministic in-process environment) and feeds
the results back as ``tool`` messages — until the model stops calling tools
or hits ``max_steps``. Scoring is exact: final answer text AND the terminal
state of the environment.
"""

from __future__ import annotations

import json as _json
import re as _re

# --------------------------------------------------------------------------- #
# Shared JSON schemas (strict, OpenAI json_schema style)
# ------------------------------------------------------------------ #
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}

TOOL_CALLS_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tool_calls"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------- #
# Shared: flight booking environment (s4 multi-turn tool workflow)
# --------------------------------------------------------------------------- #
_FLIGHT_UNIT_PRICE_EUR = 149.50

_FLIGHT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": (
                "Book a flight. Returns {'ok': True, 'booking_id': str, "
                "'total_price_eur': float} — the booking id is needed to look "
                "the booking up afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string"},
                    "to_city": {"type": "string"},
                    "date": {"type": "string", "format": "date"},
                    "passengers": {"type": "integer"},
                    "cabin": {"type": "string", "enum": ["economy", "business"]},
                },
                "required": ["from_city", "to_city", "date", "passengers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": (
                "Look up a booking by id. Returns the booking details including "
                "'total_price_eur', or {'error': 'not found'}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
]


def _agent_env_flight() -> dict:
    return {
        "calls": 0,
        "bookings": {},
        "booking": None,
        "book_args": None,
    }


def _agent_step_flight(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "book_flight":
        booking_id = f"BK-{env['calls']:04d}"
        passengers = int(args.get("passengers") or 0)
        total = round(_FLIGHT_UNIT_PRICE_EUR * passengers, 2)
        env["bookings"][booking_id] = {
            "from_city": str(args.get("from_city", "")),
            "to_city": str(args.get("to_city", "")),
            "date": str(args.get("date", "")),
            "passengers": passengers,
            "cabin": str(args.get("cabin", "")),
            "total_price_eur": total,
        }
        env["booking"] = booking_id
        env["book_args"] = dict(args)
        return {"ok": True, "booking_id": booking_id, "total_price_eur": total}
    if name == "get_booking":
        bid = str(args.get("booking_id", ""))
        b = env["bookings"].get(bid)
        if b is None:
            return {"error": f"not found: {bid}"}
        return dict(b, booking_id=bid)
    return {"error": f"unknown tool: {name}"}


# --------------------------------------------------------------------------- #
# Suite 1: structured (deterministic scoring)
# --------------------------------------------------------------------------- #
STRUCTURED_TASKS: list[dict] = [
    {
        "id": "s1",
        "name": "simple arithmetic",
        "suite": "structured",
        "kind": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a calculator. Return ONLY a JSON object with the "
                    'integer field "answer".'
                ),
            },
            {"role": "user", "content": "What is 17 * 23 + 5?"},
        ],
        "schema": ANSWER_SCHEMA,
        "check": lambda out: out.get("answer") == 396,
    },
    {
        "id": "s2",
        "name": "multi-step arithmetic",
        "suite": "structured",
        "kind": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a calculator. Return ONLY a JSON object with the "
                    'integer field "answer".'
                ),
            },
            {"role": "user", "content": "A shop has 3 shelves. Shelf 1 holds 4 boxes of 12 eggs each. "
                                       "Shelf 2 holds 5 boxes of 8 eggs each. Shelf 3 holds 2 boxes of 6 eggs each. "
                                       "How many eggs are there in total?"},
        ],
        "schema": ANSWER_SCHEMA,
        "check": lambda out: out.get("answer") == 100,
    },
    {
        "id": "s3",
        "name": "tool call — single",
        "suite": "structured",
        "kind": "tools",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Call exactly the tool that answers the user's request, "
                    "with the correct arguments."
                ),
            },
            {"role": "user", "content": "What's the weather like in Tokyo right now?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "check": lambda calls: (
            len(calls) == 1
            and calls[0].get("name") == "get_weather"
            and calls[0].get("arguments", {}).get("city", "").lower() in ("tokyo", "tokyo, japan")
        ),
    },
    {
        # s4 was a single-turn "tools" task: the model had to call book_flight
        # AND get_price, but get_price requires a booking_id that only exists as
        # the RESULT of book_flight — impossible in one turn, so every tested
        # configuration failed it (the model correctly stopped after one call).
        # Replaced with a proper multi-turn workflow on a deterministic env:
        # book the flight, then read the price back from the booking. Same id,
        # name and suite, so suite sizes and comparisons stay valid.
        "id": "s4",
        "name": "tool call — multi-step",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {"role": "user", "content": "Book a flight from Paris to Berlin on 2026-12-05 for 2 passengers, "
                                        "economy class. Then tell me the total price in euros."},
        ],
        "tools": _FLIGHT_TOOLS,
        "env_factory": lambda: _agent_env_flight(),
        "step": _agent_step_flight,
        "max_steps": 8,
        # Deterministic check: a real booking must exist with the requested
        # parameters AND the final answer must state the correct total price
        # (2 x 149.50 = 299 euros).
        "check": lambda env, content: (
            env["booking"] is not None
            and env["book_args"].get("from_city", "").lower() == "paris"
            and env["book_args"].get("to_city", "").lower() == "berlin"
            and str(env["book_args"].get("date")) == "2026-12-05"
            and int(env["book_args"].get("passengers") or 0) == 2
            and (env["book_args"].get("cabin") in ("economy", None))
            and bool(_re.search(r"(?<![\d.])299(?:\.0+)?(?![\d])", content or ""))
        ),
    },
    {
        "id": "s5",
        "name": "list extraction (structured)",
        "suite": "structured",
        "kind": "json",
        "messages": [
            {"role": "user", "content": (
                "Extract the names of all countries mentioned below, exactly as written. "
                "Return a JSON object with field \"countries\" (array of strings).\n\n"
                "\"I visited France and Germany last year. This spring I hiked in Switzerland, "
                "and next month I plan to visit Japan. My favorite food is Italian pasta.\""
            )},
        ],
        "schema": {
            "type": "object",
            "properties": {"countries": {"type": "array", "items": {"type": "string"}}},
            "required": ["countries"],
            "additionalProperties": False,
        },
        "check": lambda out: set(out.get("countries", [])) == {"France", "Germany", "Switzerland", "Japan"},
    },
    {
        "id": "s6",
        "name": "multi-step reasoning (structured)",
        "suite": "structured",
        "kind": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful problem solver. Work through the problem step by step, then "
                    "return ONLY a JSON object with exactly these fields: "
                    '"steps" (array of short strings, one per reasoning step) and '
                    '"answer" (integer).'
                ),
            },
            {
                "role": "user",
                "content": (
                    "A clock gains 5 minutes every day. It was set correctly at noon on Monday. "
                    "What time does it show at noon the following Friday? "
                    "(Give the answer as total minutes after midnight, e.g. 12:30 -> 750.)"
                ),
            },
        ],
        # 4 days x 5 min = 20 min fast; 12:20 -> 740 minutes after midnight.
        "schema": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}},
                "answer": {"type": "integer"},
            },
            "required": ["steps", "answer"],
            "additionalProperties": False,
        },
        "check": lambda out: (
            out.get("answer") == 740 and len(out.get("steps", [])) >= 2
        ),
    },
]

# --------------------------------------------------------------------------- #
# Suite 3: agentic workflows (multi-turn tool calling, exact scoring)
# --------------------------------------------------------------------------- #
# Each task defines ``env`` — a deterministic in-process environment the
# harness executes tool calls against — plus ``tools`` (real OpenAI schemas
# sent to the server). The model drives the loop until it stops calling
# tools; scoring checks BOTH its final text answer and the terminal state of
# the environment, so a model that fakes results without calling tools fails.

# --- shared: calendar ------------------------------------------------------ #
_CALENDAR = {
    "2026-11-10": [
        {"id": 1, "title": "Team standup", "start": "09:30", "end": "10:00", "location": "Zoom"},
        {"id": 2, "title": "Lunch with Sam", "start": "12:30", "end": "13:30", "location": "Cafe Noir"},
    ],
    "2026-11-11": [
        {"id": 3, "title": "Design review", "start": "14:00", "end": "15:00", "location": "Room B"},
    ],
}

_CAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List all calendar events for a date (YYYY-MM-DD)",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_free_slots",
            "description": ("Find free meeting slots between two times on a date. "
                            "Times are HH:MM 24h. Returns candidate start times."),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "start_after": {"type": "string"},
                    "end_before": {"type": "string"},
                    "duration_min": {"type": "integer"},
                },
                "required": ["date", "start_after", "end_before", "duration_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_event",
            "description": "Schedule a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["date", "title", "start", "end"],
            },
        },
    },
]

# --- shared: knowledge base ------------------------------------------------ #
_KB = {
    "invoice_total_usd": 240,
    "vat_rate_pct": 15,
}
_KB_TOTAL = _KB["invoice_total_usd"] * (1 + _KB["vat_rate_pct"] / 100)  # 276.0

_KB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": ("Search the knowledge base for a key (e.g. 'invoice_total_usd', "
                            "'vat_rate_pct'). Returns {'found': bool, 'value': ...}."),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facts",
            "description": ("Get a list of key facts, optionally filtered by topic. "
                            "Returns {'facts': [...]}."),
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": ("Run a simple filter query against the fact database. "
                            "Returns {'rows': [...]}."),
            "parameters": {
                "type": "object",
                "properties": {"filter": {"type": "string"}},
                "required": ["filter"],
            },
        },
    },
]


def _agent_env_calendar() -> dict:
    return {
        "calls": 0,
        "list_dates": [],
        "free_slots_queried": False,
        "events": {k: [dict(e) for e in v] for k, v in _CALENDAR.items()},
        "scheduled": None,
    }


def _agent_env_kb() -> dict:
    return {"calls": 0, "searched": [], "facts_queried": False, "db_queried": False}


def _agent_step_calendar(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "list_events":
        date = str(args.get("date", ""))
        env["list_dates"].append(date)
        return {"events": [dict(e) for e in env["events"].get(date, [])]}
    if name == "find_free_slots":
        date = str(args.get("date", ""))
        start_after = str(args.get("start_after", ""))
        end_before = str(args.get("end_before", ""))
        duration = int(args.get("duration_min") or 0)
        env["free_slots_queried"] = True
        events = sorted(env["events"].get(date, []), key=lambda e: e["start"])
        slots: list[str] = []
        cur = start_after
        while len(cur) == 5 and cur < end_before:
            hh, mm = int(cur[:2]), int(cur[3:])
            dur_h, dur_m = duration // 60, duration % 60
            slot_end_h = hh + dur_h
            slot_end_m = mm + dur_m
            if slot_end_m >= 60:
                slot_end_h += 1
                slot_end_m -= 60
            slot_end = f"{slot_end_h:02d}:{slot_end_m:02d}"
            fits = all(
                not (cur < e["end"] and slot_end > e["start"]) for e in events
            )
            if fits and slot_end <= end_before:
                slots.append(cur)
            mm += 30
            hh += mm // 60
            mm %= 60
            cur = f"{hh:02d}:{mm:02d}"
        return {"free_slots": slots[:10]}
    if name == "schedule_event":
        date = str(args.get("date", ""))
        env["events"].setdefault(date, []).append({
            "id": 99,
            "title": str(args.get("title", "")),
            "start": str(args.get("start", "")),
            "end": str(args.get("end", "")),
            "location": str(args.get("location", "")),
        })
        env["scheduled"] = {"date": date, "title": str(args.get("title", ""))}
        return {"ok": True, "id": 99}
    return {"error": f"unknown tool: {name}"}


def _agent_step_kb(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "search_kb":
        q = str(args.get("query", "")).strip().lower()
        env["searched"].append(q)
        return {"found": q in _KB, "value": _KB.get(q)}
    if name == "get_facts":
        topic = str(args.get("topic", ""))
        env["facts_queried"] = True
        facts = [
            f"Invoice total: ${_KB['invoice_total_usd']}",
            f"VAT rate: {_KB['vat_rate_pct']}%",
            "Payment terms: net 30",
        ]
        if topic:
            facts = [f for f in facts if topic.lower() in f.lower()] or ["(no matching facts)"]
        return {"facts": facts}
    if name == "query_db":
        env["db_queried"] = True
        flt = str(args.get("filter", "")).lower()
        rows = []
        for k, v in _KB.items():
            if flt in k:
                rows.append({"key": k, "value": v})
        return {"rows": rows}
    return {"error": f"unknown tool: {name}"}


# --- shared: inventory ------------------------------------------------------ #
_INVENTORY = {
    "laptop": {"qty": 3, "price": 1200},
    "mouse": {"qty": 15, "price": 25},
    "keyboard": {"qty": 8, "price": 40},
}

_INV_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_item",
            "description": ("Get the stock quantity and unit price for one product. "
                            "Returns {'item': str, 'qty': int, 'price': float} or "
                            "{'error': 'not found'}."),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_items",
            "description": "List all product names in the inventory. Returns {'items': [str, ...]}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_stock",
            "description": ("Change the stock quantity of a product by a delta (negative = remove). "
                            "Returns {'item': str, 'qty': int}."),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "delta": {"type": "integer"}},
                "required": ["name", "delta"],
            },
        },
    },
]


def _agent_env_inventory() -> dict:
    return {
        "calls": 0,
        "items_queried": [],
        "adjustments": [],
        "stock": {k: dict(v) for k, v in _INVENTORY.items()},
    }


def _agent_step_inventory(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "list_items":
        return {"items": list(env["stock"].keys())}
    if name == "get_item":
        item = str(args.get("name", "")).strip().lower()
        env["items_queried"].append(item)
        rec = env["stock"].get(item)
        if rec is None:
            return {"error": f"not found: {item}"}
        return {"item": item, "qty": rec["qty"], "price": rec["price"]}
    if name == "adjust_stock":
        item = str(args.get("name", "")).strip().lower()
        delta = int(args.get("delta") or 0)
        env["adjustments"].append({"item": item, "delta": delta})
        if item not in env["stock"]:
            return {"error": f"not found: {item}"}
        rec = env["stock"][item]
        rec["qty"] = max(0, rec["qty"] + delta)
        return {"item": item, "qty": rec["qty"]}
    return {"error": f"unknown tool: {name}"}


# --- shared: task board ----------------------------------------------------- #
_BOARD = {
    "T-101": {"title": "Login page 500 error", "type": "bug", "priority": "high",
              "status": "open", "owner": None},
    "T-102": {"title": "Dark mode toggle", "type": "feature", "priority": "high",
              "status": "open", "owner": None},
    "T-103": {"title": "Crash on file upload", "type": "bug", "priority": "medium",
              "status": "open", "owner": None},
    "T-104": {"title": "Slow search index", "type": "bug", "priority": "high",
              "status": "in_progress", "owner": "sam"},
}

_BOARD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": ("List all tasks on the board with their fields "
                            "(id, title, type, priority, status, owner). "
                            "Returns {'tasks': [...]}."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": ("Update a task's status and/or owner. Only the fields "
                            "you pass are changed. Returns the updated task or "
                            "{'error': ...}."),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "in_progress", "done"]},
                    "owner": {"type": "string"},
                },
                "required": ["id"],
            },
        },
    },
]


def _agent_env_board() -> dict:
    return {"calls": 0, "updates": [], "tasks": {k: dict(v) for k, v in _BOARD.items()}}


def _agent_step_board(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "list_tasks":
        return {"tasks": [{"id": k, **v} for k, v in env["tasks"].items()]}
    if name == "update_task":
        tid = str(args.get("id", "")).strip()
        rec = env["tasks"].get(tid)
        if rec is None:
            return {"error": f"not found: {tid}"}
        if args.get("status") is not None:
            rec["status"] = str(args.get("status"))
        if args.get("owner") is not None:
            rec["owner"] = str(args.get("owner"))
        env["updates"].append({"id": tid})
        return {"id": tid, **rec}
    return {"error": f"unknown tool: {name}"}


# --- shared: package registry ----------------------------------------------- #
_PACKAGES = {
    "core": [],
    "utils": ["core"],
    "net": ["core", "utils"],
    "app": ["net", "utils"],
    "cli": ["app"],
}

_PKG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_packages",
            "description": "List all package names in the registry. Returns {'packages': [str, ...]}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_package",
            "description": ("Get a package's dependencies. Returns "
                            "{'name': str, 'deps': [str, ...]} or {'error': ...}."),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]


def _agent_env_packages() -> dict:
    return {"calls": 0, "queried": []}


def _agent_step_packages(env: dict, name: str, args: dict) -> dict:
    env["calls"] += 1
    if name == "list_packages":
        return {"packages": list(_PACKAGES.keys())}
    if name == "get_package":
        pkg = str(args.get("name", "")).strip().lower()
        env["queried"].append(pkg)
        deps = _PACKAGES.get(pkg)
        if deps is None:
            return {"error": f"not found: {pkg}"}
        return {"name": pkg, "deps": list(deps)}
    return {"error": f"unknown tool: {name}"}


def _check_build_order(content: str) -> bool:
    """True if the final answer lists one package per line in a valid build order.

    Accepts optional bullet/number prefixes and trailing punctuation on each
    line; every package must appear exactly once, and each line's package must
    come after all of its dependencies.
    """
    seq: list[str] = []
    for line in (content or "").splitlines():
        clean = _re.sub(r"^\s*(?:[-*]|\d+[.)])?\s*", "", line).strip().lower().rstrip(".,;")
        if clean in _PACKAGES:
            seq.append(clean)
    if len(seq) != len(_PACKAGES) or set(seq) != set(_PACKAGES):
        return False
    seen: set[str] = set()
    for name in seq:
        if not all(d in seen for d in _PACKAGES[name]):
            return False
        seen.add(name)
    return True


AGENT_TASKS: list[dict] = [
    {
        "id": "a1",
        "name": "agent — schedule in a free slot (multi-turn)",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Find a free 30-minute slot on Wednesday 2026-11-11 between 09:00 and 17:00 "
                    "and schedule a meeting titled 'Design sync' at 15:30."
                ),
            },
        ],
        "tools": _CAL_TOOLS,
        "env_factory": lambda: _agent_env_calendar(),
        "step": _agent_step_calendar,
        "max_steps": 8,
        # Deterministic check: the model must have scheduled a 'Design sync'
        # event on 2026-11-11 at 15:30 (a slot that is actually free).
        "check": lambda env, content: (
            env["scheduled"] is not None
            and env["scheduled"]["date"] == "2026-11-11"
            and env["scheduled"]["title"].lower() == "design sync"
            and any(
                e["start"] == "15:30" and e["end"] == "16:00"
                for e in env["events"].get("2026-11-11", [])
            )
        ),
    },
    {
        "id": "a2",
        "name": "agent — multi-source research (multi-turn)",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I need the invoice total in USD and the VAT rate from our knowledge base. "
                    "Compute the total including VAT and give me the final number in USD."
                ),
            },
        ],
        "tools": _KB_TOOLS,
        "env_factory": lambda: _agent_env_kb(),
        "step": _agent_step_kb,
        "max_steps": 8,
        # Deterministic check: the model must have queried the KB (at least one
        # of search_kb / get_facts / query_db) AND its final answer must contain
        # the correct total (276.0).
        # True if the final answer contains a number equal to 276 (the VAT total).
        "check": lambda env, content: (
            (env["searched"] or env["facts_queried"] or env["db_queried"])
            and bool(_re.search(r"(?<![\d.])276(?:\.0+)?(?![\d])", content or ""))
        ),
    },
    {
        "id": "a3",
        "name": "agent — inventory audit & reorder (multi-turn)",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Audit our inventory: check every product, flag anything in stock below 10 units, "
                    "and reorder each flagged product by adding exactly 10 units to its stock. "
                    "In your final answer, list the products you reordered."
                ),
            },
        ],
        "tools": _INV_TOOLS,
        "env_factory": lambda: _agent_env_inventory(),
        "step": _agent_step_inventory,
        "max_steps": 12,
        # Deterministic check: every product must have been looked up, and the
        # two low-stock items (laptop=3, keyboard=8; mouse=15 is fine) must each
        # end up with exactly +10 units, and mouse must stay untouched. Checking
        # the FINAL stock levels catches double-reorders and wrong deltas that
        # an any()-style check (\"at least one +10\") lets through.
        "check": lambda env, content: (
            set(env["items_queried"]) >= {"laptop", "mouse", "keyboard"}
            and env["stock"]["laptop"]["qty"] == 13
            and env["stock"]["keyboard"]["qty"] == 18
            and env["stock"]["mouse"]["qty"] == 15
        ),
    },
    {
        "id": "a4",
        "name": "agent — task board triage (multi-turn)",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Triage our task board: find every task that is a bug (type 'bug'), has high "
                    "priority, and is still open. Move each of those tasks to status 'in_progress' "
                    "and assign them to 'ana'. Do not touch any other task. In your final answer, "
                    "list the task ids you changed."
                ),
            },
        ],
        "tools": _BOARD_TOOLS,
        "env_factory": lambda: _agent_env_board(),
        "step": _agent_step_board,
        "max_steps": 8,
        # Deterministic check: exactly T-101 (bug + high + open) must end up
        # in_progress/ana; the feature (T-102), the medium-priority bug
        # (T-103) and the already-in-progress task (T-104, owned by sam) must
        # be untouched.
        "check": lambda env, content: (
            env["tasks"]["T-101"]["status"] == "in_progress"
            and env["tasks"]["T-101"]["owner"] == "ana"
            and env["tasks"]["T-102"] == {"title": "Dark mode toggle", "type": "feature",
                                          "priority": "high", "status": "open", "owner": None}
            and env["tasks"]["T-103"] == {"title": "Crash on file upload", "type": "bug",
                                          "priority": "medium", "status": "open", "owner": None}
            and env["tasks"]["T-104"] == {"title": "Slow search index", "type": "bug",
                                          "priority": "high", "status": "in_progress", "owner": "sam"}
        ),
    },
    {
        "id": "a5",
        "name": "agent — build order from dependencies (multi-turn)",
        "suite": "structured",
        "kind": "agent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You can call tools. Use them to complete the user's request step by step. "
                    "When you are done, reply with a short final answer and stop calling tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Our build system has several packages (query them with the tools). Give me a "
                    "valid build order — a sequence in which every package appears only after all "
                    "of its dependencies. In your final answer, list one package per line, nothing else."
                ),
            },
        ],
        "tools": _PKG_TOOLS,
        "env_factory": lambda: _agent_env_packages(),
        "step": _agent_step_packages,
        "max_steps": 10,
        # Deterministic check: the final answer must list each package exactly
        # once, one per line, in a dependency-respecting order (for this graph
        # the valid order is uniquely forced: core, utils, net, app, cli).
        "check": lambda env, content: _check_build_order(content),
    },
]


# --------------------------------------------------------------------------- #
# Suite 2: open-ended (LLM-judge scored)
# --------------------------------------------------------------------------- #
OPEN_TASKS: list[dict] = [
    {
        "id": "o1",
        "name": "summarization — faithfulness + brevity",
        "suite": "open",
        "messages": [
            {"role": "user", "content": (
                "Summarize the following paragraph in exactly 2 sentences:\n\n"
                "The Great Barrier Reef, located off the coast of Queensland in Australia, is the "
                "world's largest structure made by living organisms. It can be seen from space and "
                "spans over 2,300 kilometers. The reef supports thousands of species of fish, coral, "
                "and marine plants. However, rising ocean temperatures, pollution, and storms are "
                "causing widespread coral bleaching, threatening the ecosystem's survival in the coming decades."
            )},
        ],
        "judge_rubric": {
            "faithfulness": "Is the summary factually accurate with respect to the source paragraph (no invented facts)?",
            "brevity": "Does it follow the instruction of exactly 2 sentences?",
        },
    },
    {
        "id": "o2",
        "name": "formatting — bullet list",
        "suite": "open",
        "messages": [
            {"role": "user", "content": (
                "List exactly 5 benefits of regular exercise, one per line, each starting with a dash (-). "
                "Do not add any other text."
            )},
        ],
        "judge_rubric": {
            "instruction_following": "Are there exactly 5 items, one per line, each starting with '-'?",
            "quality": "Are all five items genuine benefits of regular exercise?",
        },
    },
    {
        "id": "o3",
        "name": "tone — professional email",
        "suite": "open",
        "messages": [
            {"role": "user", "content": (
                "Write a short, polite email to a client explaining that their order will be delayed by one week "
                "due to a supplier issue. Offer a 10% discount as compensation. Keep it under 120 words."
            )},
        ],
        "judge_rubric": {
            "instruction_following": "Does it mention the one-week delay, the supplier cause, and the 10% discount?",
            "tone": "Is the tone professional and polite (not apologetic-to-a-fault, not casual)?",
        },
    },
]

ALL_TASKS: list[dict] = STRUCTURED_TASKS + AGENT_TASKS + OPEN_TASKS


def tasks_for_suite(suite: str) -> list[dict]:
    if suite == "structured":
        return STRUCTURED_TASKS
    if suite == "agent":
        return AGENT_TASKS
    if suite == "open":
        return OPEN_TASKS
    return ALL_TASKS
