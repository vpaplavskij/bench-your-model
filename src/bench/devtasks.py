"""Developer / coding task definitions.

Each task is a self-contained coding problem with a DETERMINISTIC checker
(a pure predicate over the model's code + its runtime behavior), so scoring
is exact and reproducible — no LLM judge, no flakiness from prose.

Task kinds (each maps to a real developer skill):
  * ``code``   — write a function that passes hidden unit tests
  * ``debug``  — fix a buggy implementation so it passes the tests
  * ``refactor`` — rewrite to satisfy a behavior-preserving constraint
  * ``multi``  — implement several related functions / a small module

A task dict:
    id, name, kind, prompt (full instruction), starter (code shown in prompt),
    check(code_src, run) -> {"pass": bool, "detail": str}

``run`` is a helper that executes the model's code and returns a result dict
(see :func:`devcheck.run_code`). Checkers stay pure so they're easy to audit.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Task 1 — code: implement a function (hidden unit tests)
# --------------------------------------------------------------------------- #
T1_PROMPT = """\
Implement the Python function `parse_log_line(line: str) -> dict`.

A log line has the form:
    [2026-11-09T10:30:00Z] INFO service=auth msg="login ok" code=200

Write a function that parses it into a dict with keys:
    - "timestamp": the ISO timestamp string (without brackets)
    - "level":     the log level (e.g. "INFO")
    - "fields":    dict of remaining key=value pairs; values that are quoted
                   strings keep their text without quotes; integer-looking
                   values become int.

Rules:
  * There is exactly one space-separated token per field after the level.
  * A value may be a bare word, an integer, or a double-quoted string.
  * Return {} for an empty/invalid line.

Return ONLY the code (no markdown fences, no explanation).
"""

T1_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("parse_log_line")
    if fn is None:
        return {"pass": False, "detail": "no function named parse_log_line found"}
    cases = [
        ('[2026-11-09T10:30:00Z] INFO service=auth msg="login ok" code=200',
         {"timestamp": "2026-11-09T10:30:00Z", "level": "INFO",
          "fields": {"service": "auth", "msg": "login ok", "code": 200}}),
        ('[2026-11-09T10:31:05Z] ERROR service=pay code=500',
         {"timestamp": "2026-11-09T10:31:05Z", "level": "ERROR",
          "fields": {"service": "pay", "code": 500}}),
        ("", {}),
    ]
    for line, want in cases:
        got = fn(line)
        if got != want:
            return {"pass": False, "detail": f"parse_log_line({line!r}) -> {got!r}, wanted {want!r}"}
    return {"pass": True, "detail": "all 3 cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 2 — debug: fix a buggy implementation
# --------------------------------------------------------------------------- #
T2_PROMPT = """\
The function below is supposed to compute the sum of a list of numbers, but it
has bugs. Fix it so it works correctly. Do not change the function name or
signature.

    def total(nums):
        s = 0
        for x in nums:
            s += x
        return s - 1

Return ONLY the corrected code (no markdown fences, no explanation).
"""

T2_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("total")
    if fn is None:
        return {"pass": False, "detail": "no function named total found"}
    cases = [([1, 2, 3], 6), ([], 0), ([5], 5), ([-1, 1, -1], -1), ([10, 20, 30, 40], 100)]
    for nums, want in cases:
        got = fn(nums)
        if got != want:
            return {"pass": False, "detail": f"total({nums}) -> {got!r}, wanted {want}"}
    return {"pass": True, "detail": "all 5 cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 3 — refactor: behavior-preserving constraint
# --------------------------------------------------------------------------- #
T3_PROMPT = """\
Refactor the function below so that it uses NO `for` or `while` at all — not
even in a list/dict/set comprehension or generator expression — and still
returns exactly the same result for every input. (Hint: `filter(lambda ...)`
combined with `next(..., None)` can help.)

    def first_even(nums):
        for n in nums:
            if n % 2 == 0:
                return n
        return None

Return ONLY the refactored code (no markdown fences, no explanation).
"""

T3_CHECKER = """\
def check(code_src, run):
    # Behavior must be preserved AND no explicit loop may remain.
    import re
    if re.search(r'\\b(for|while)\\b', code_src):
        return {"pass": False, "detail": "refactor still contains an explicit for/while loop"}
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("first_even")
    if fn is None:
        return {"pass": False, "detail": "no function named first_even found"}
    cases = [([1, 3, 4], 4), ([2], 2), ([1, 3, 5], None), ([], None),
             ([7, 8, 9], 8), ([-2, -3], -2)]
    for nums, want in cases:
        got = fn(nums)
        if got != want:
            return {"pass": False, "detail": f"first_even({nums}) -> {got!r}, wanted {want}"}
    return {"pass": True, "detail": "no loop + all 6 behavior cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 4 — multi: implement a small module (several functions)
# --------------------------------------------------------------------------- #
T4_PROMPT = """\
Implement two Python functions for a simple text analyzer:

1. `word_frequencies(text: str) -> dict` — count how many times each word
   appears. A "word" is a run of letters; ignore case and any punctuation
   attached to it. Return {} for empty text.

2. `top_words(text: str, k: int) -> list` — return the k most common words as
   a list of (word, count) tuples, sorted by count descending, then alphabetically.
   If there are fewer than k distinct words, return all of them.

Return ONLY the code defining both functions (no markdown fences, no explanation).
"""

T4_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    wf = r["ns"].get("word_frequencies")
    tw = r["ns"].get("top_words")
    if wf is None or tw is None:
        return {"pass": False, "detail": "need both word_frequencies and top_words defined"}
    text = "The cat sat on the mat. The dog barked at the cat!"
    want_freq = {"the": 4, "cat": 2, "sat": 1, "on": 1, "mat": 1, "dog": 1, "barked": 1, "at": 1}
    got_freq = wf(text)
    if got_freq != want_freq:
        return {"pass": False, "detail": f"word_frequencies -> {got_freq!r}, wanted {want_freq!r}"}
    # top_words: the(4), cat(2), then count-1 words alphabetically: at, barked, dog, mat, on, sat
    want_top = [("the", 4), ("cat", 2), ("at", 1), ("barked", 1)]
    got_top = tw(text, 4)
    if list(got_top) != want_top:
        return {"pass": False, "detail": f"top_words(...,4) -> {got_top!r}, wanted {want_top}"}
    # edge: k larger than distinct count returns all, still sorted
    got_all = tw("a a b", 10)
    if list(got_all) != [("a", 2), ("b", 1)]:
        return {"pass": False, "detail": f"top_words('a a b',10) -> {got_all!r}, wanted [('a',2),('b',1)]"}
    return {"pass": True, "detail": "freq + top_words (incl. tie-break & overflow) all passed"}
"""

# --------------------------------------------------------------------------- #
# Task 5 — code: implement a small data structure (deterministic)
# --------------------------------------------------------------------------- #
T5_PROMPT = """\
Implement a Python class `Counter` with these methods:

- `__init__(self)` — start empty.
- `add(self, n: int)` — add an integer to the internal list of values.
- `mean(self) -> float | None` — return the arithmetic mean of all added
  values as a float, or None if the list is empty.
- `median(self) -> float | None` — return the median (middle value for odd
  length; average of the two middle values for even length) as a float, or
  None if the list is empty.

Return ONLY the class definition (no markdown fences, no explanation).
"""

T5_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    cls = r["ns"].get("Counter")
    if cls is None:
        return {"pass": False, "detail": "no class named Counter found"}
    # mean cases
    c = cls()
    for n in [1, 2, 3, 4]:
        c.add(n)
    if abs(c.mean() - 2.5) > 1e-9:
        return {"pass": False, "detail": f"mean([1,2,3,4]) -> {c.mean()!r}, wanted 2.5"}
    # median odd
    c2 = cls()
    for n in [3, 1, 2]:
        c2.add(n)
    if abs(c2.median() - 2.0) > 1e-9:
        return {"pass": False, "detail": f"median([3,1,2]) -> {c2.median()!r}, wanted 2.0"}
    # median even
    c3 = cls()
    for n in [1, 2, 3, 4]:
        c3.add(n)
    if abs(c3.median() - 2.5) > 1e-9:
        return {"pass": False, "detail": f"median([1,2,3,4]) -> {c3.median()!r}, wanted 2.5"}
    # empty cases
    c4 = cls()
    if c4.mean() is not None or c4.median() is not None:
        return {"pass": False, "detail": "empty Counter must return None for mean/median"}
    return {"pass": True, "detail": "Counter (mean + median incl. empty & even) all passed"}
"""

# --------------------------------------------------------------------------- #
# Task 6 — thinking: multi-step reasoning with exact numeric answer
# --------------------------------------------------------------------------- #
T6_PROMPT = """\
Solve this step by step and return ONLY a JSON object (no markdown fences,
no explanation) with exactly two fields:
  - "steps": an array of short strings, one per reasoning step
  - "answer": an integer

Problem: A clock gains 5 minutes every day. It was set correctly at noon on
Monday. What time does it show at noon the following Friday? Give the answer
as total minutes after midnight (e.g. 12:30 -> 750).
"""

T6_CHECKER = """\
import json as _json
import re as _re

def check(code_src, run):
    text = code_src.strip()
    # Strip markdown fences if present.
    m = _re.search(r"```(?:json)?\\s*\\n(.*?)```", text, _re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return {"pass": False, "detail": "response is not valid JSON"}
    if not isinstance(parsed, dict):
        return {"pass": False, "detail": "JSON must be an object"}
    steps = parsed.get("steps")
    answer = parsed.get("answer")
    if not isinstance(steps, list) or len(steps) < 2:
        return {"pass": False, "detail": f"steps must be a list of >=2 strings, got {steps!r}"}
    if not all(isinstance(s, str) for s in steps):
        return {"pass": False, "detail": "every step must be a string"}
    # 4 days x 5 min = 20 min fast; 12:20 -> 740 minutes after midnight.
    if answer != 740:
        return {"pass": False, "detail": f"answer -> {answer!r}, wanted 740"}
    return {"pass": True, "detail": "steps + correct answer (740) verified"}
"""

# --------------------------------------------------------------------------- #
# Task 7 — code: algorithm challenge (longest increasing subsequence)
# --------------------------------------------------------------------------- #
T7_PROMPT = """\
Implement the Python function `lis(nums: list[int]) -> int`.

It must return the length of the longest strictly increasing subsequence of
`nums`. A subsequence keeps relative order but may skip elements.
Examples: lis([10, 9, 2, 5, 3, 7, 101]) == 4 (e.g. [2, 3, 7, 101]),
lis([]) == 0, lis([5]) == 1, lis([5, 4, 3, 2, 1]) == 1.

Return ONLY the code defining `lis` (no markdown fences, no explanation).
"""

T7_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("lis")
    if fn is None:
        return {"pass": False, "detail": "no function named lis found"}
    cases = [
        ([10, 9, 2, 5, 3, 7, 101], 4),
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7], 1),
        ([], 0),
        ([-1, -2, -3, -4, -5], 1),
        ([1, 2, 3, 4, 5], 5),
        ([3, 1, 2, 5, 4], 3),
    ]
    for nums, want in cases:
        got = fn(list(nums))
        if got != want:
            return {"pass": False, "detail": f"lis({nums}) -> {got!r}, wanted {want}"}
    return {"pass": True, "detail": "all 7 LIS cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 8 — code: algorithm challenge (merge overlapping intervals)
# --------------------------------------------------------------------------- #
T8_PROMPT = """\
Implement the Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]`.

It must merge all overlapping intervals and return the resulting list sorted
by start time. Two intervals overlap when one starts at or before the other
ends (touching endpoints like [1,4] and [4,5] merge into [1,5]).
Examples:
    merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]
    merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]
    merge_intervals([]) == []

Return ONLY the code defining `merge_intervals` (no markdown fences, no explanation).
"""

T8_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("merge_intervals")
    if fn is None:
        return {"pass": False, "detail": "no function named merge_intervals found"}
    cases = [
        ([(1, 3), (2, 6), (8, 10)], [(1, 6), (8, 10)]),
        ([(1, 4), (4, 5)], [(1, 5)]),
        ([(1, 4), (0, 4), (5, 6), (2, 3)], [(0, 4), (5, 6)]),
        ([], []),
        ([(1, 2), (3, 4)], [(1, 2), (3, 4)]),
    ]
    for intervals, want in cases:
        got = fn([tuple(iv) for iv in intervals])
        if list(got) != [tuple(w) for w in want]:
            return {"pass": False, "detail": f"merge_intervals({intervals}) -> {got!r}, wanted {want!r}"}
    return {"pass": True, "detail": "all 5 interval-merge cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 9 — thinking: multi-step reasoning (train fare word problem)
# --------------------------------------------------------------------------- #
T9_PROMPT = """\
Solve this step by step and return ONLY a JSON object (no markdown fences,
no explanation) with exactly two fields:
  - "steps": an array of short strings, one per reasoning step
  - "answer": an integer

Problem: A train ride costs $10 for the first 30 km. Each additional full 15
km (or part thereof) adds $4. What is the cost of a 120 km ride? Give the
answer as a whole number of dollars.
"""

T9_CHECKER = """\
import json as _json
import re as _re

def check(code_src, run):
    text = code_src.strip()
    m = _re.search(r"```(?:json)?\\s*\\n(.*?)```", text, _re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return {"pass": False, "detail": "response is not valid JSON"}
    if not isinstance(parsed, dict):
        return {"pass": False, "detail": "JSON must be an object"}
    steps = parsed.get("steps")
    answer = parsed.get("answer")
    if not isinstance(steps, list) or len(steps) < 2:
        return {"pass": False, "detail": f"steps must be a list of >=2 strings, got {steps!r}"}
    if not all(isinstance(s, str) for s in steps):
        return {"pass": False, "detail": "every step must be a string"}
    # 120 km: first 30 km -> $10; remaining 90 km = six full 15-km chunks -> 6 x $4 = $24. Total $34.
    if answer != 34:
        return {"pass": False, "detail": f"answer -> {answer!r}, wanted 34"}
    return {"pass": True, "detail": "steps + correct answer (34) verified"}
"""

# --------------------------------------------------------------------------- #
# Task 10 — debug: fix a buggy cycle detector
# --------------------------------------------------------------------------- #
T10_PROMPT = """\
The function below is supposed to detect whether a linked list has a cycle
(a node whose `next` chain loops back to an earlier node). It has bugs. Fix
it so it returns True exactly when the list headed by `head` contains a
cycle, and False otherwise (including for `head is None`). Do not change the
function name or signature.

    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None

    def has_cycle(head):
        seen = set()
        while head is not None:
            if id(head) in seen:
                return False
            seen.add(id(head))
            head = head.next
        return True

Return ONLY the corrected code (no markdown fences, no explanation).
"""

T10_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("has_cycle")
    if fn is None:
        return {"pass": False, "detail": "no function named has_cycle found"}
    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None
    # cyclic list: 1 -> 2 -> 3 -> (back to 1)
    a, b, c = Node(1), Node(2), Node(3)
    a.next, b.next, c.next = b, c, a
    if fn(a) is not True:
        return {"pass": False, "detail": f"has_cycle(cyclic list) -> {fn(a)!r}, wanted True"}
    # plain two-node list, no cycle
    d, e = Node(1), Node(2)
    d.next = e
    if fn(d) is not False:
        return {"pass": False, "detail": f"has_cycle(plain list) -> {fn(d)!r}, wanted False"}
    # single node, no cycle
    s = Node(9)
    if fn(s) is not False:
        return {"pass": False, "detail": f"has_cycle(single node) -> {fn(s)!r}, wanted False"}
    # empty list
    if fn(None) is not False:
        return {"pass": False, "detail": f"has_cycle(None) -> {fn(None)!r}, wanted False"}
    return {"pass": True, "detail": "cycle / no-cycle / single-node / empty cases all passed"}
"""

# --------------------------------------------------------------------------- #
# Task 11 — code: algorithm (8-puzzle solvability via inversion parity)
# --------------------------------------------------------------------------- #
T11_PROMPT = """\
Implement the Python function `solvable(puzzle: list[int]) -> bool`.

`puzzle` is a 3x3 sliding-puzzle board in row-major order: the tiles 1..8
plus one blank, represented by 0. A state is solvable (reachable from the
solved state [1, 2, 3, 4, 5, 6, 7, 8, 0]) if and only if the number of tile
inversions is even — an inversion is a pair of tiles (ignoring the blank) in
which a larger tile appears before a smaller one.

Examples:
    solvable([1, 2, 3, 4, 5, 6, 7, 8, 0]) == True   # solved state
    solvable([2, 1, 3, 4, 5, 6, 7, 8, 0]) == False  # one swap from solved

Return ONLY the code defining `solvable` (no markdown fences, no explanation).
"""

T11_CHECKER = """\
def check(code_src, run):
    r = run(code_src)
    if not r["ok"]:
        return {"pass": False, "detail": f"code did not import/run: {r['error']}"}
    fn = r["ns"].get("solvable")
    if fn is None:
        return {"pass": False, "detail": "no function named solvable found"}
    cases = [
        ([1, 2, 3, 4, 5, 6, 7, 8, 0], True),   # solved state
        ([2, 1, 3, 4, 5, 6, 7, 8, 0], False),  # one swap from solved
        ([1, 2, 3, 4, 5, 0, 7, 8, 6], True),   # one legal move from solved
        ([1, 2, 3, 4, 5, 6, 7, 0, 8], True),   # blank mid-bottom row
        ([1, 2, 3, 4, 5, 6, 0, 8, 7], False),  # 7/8 swapped across the blank
        ([3, 2, 1, 4, 5, 6, 7, 8, 0], False),  # reversed top row (odd inversions)
    ]
    for puzzle, want in cases:
        got = fn(list(puzzle))
        if bool(got) != want:
            return {"pass": False, "detail": f"solvable({puzzle}) -> {got!r}, wanted {want}"}
    return {"pass": True, "detail": "all 6 8-puzzle solvability cases passed"}
"""

# --------------------------------------------------------------------------- #
# Task 12 — thinking: rate-reasoning trap (widgets)
# --------------------------------------------------------------------------- #
T12_PROMPT = """\
Solve this step by step and return ONLY a JSON object (no markdown fences,
no explanation) with exactly two fields:
  - "steps": an array of short strings, one per reasoning step
  - "answer": an integer

Problem: It takes 5 machines 5 minutes to make 5 widgets. How many minutes
does it take 100 machines to make 100 widgets? Assume all machines work at
the same constant rate and independently. Give the answer as a whole number
of minutes.
"""

T12_CHECKER = """\
import json as _json
import re as _re

def check(code_src, run):
    text = code_src.strip()
    m = _re.search(r"```(?:json)?\\s*\\n(.*?)```", text, _re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return {"pass": False, "detail": "response is not valid JSON"}
    if not isinstance(parsed, dict):
        return {"pass": False, "detail": "JSON must be an object"}
    steps = parsed.get("steps")
    answer = parsed.get("answer")
    if not isinstance(steps, list) or len(steps) < 2:
        return {"pass": False, "detail": f"steps must be a list of >=2 strings, got {steps!r}"}
    if not all(isinstance(s, str) for s in steps):
        return {"pass": False, "detail": "every step must be a string"}
    # 5 machines make 5 widgets in 5 min -> each machine makes 1 widget per 5 min.
    # 100 machines make 100 widgets (one each) in the same 5 minutes.
    if answer != 5:
        return {"pass": False, "detail": f"answer -> {answer!r}, wanted 5"}
    return {"pass": True, "detail": "steps + correct answer (5) verified"}
"""

# --------------------------------------------------------------------------- #
# Task 13 — thinking: constraint-satisfaction logic puzzle
# --------------------------------------------------------------------------- #
T13_PROMPT = """\
Solve this step by step and return ONLY a JSON object (no markdown fences,
no explanation) with exactly four fields:
  - "steps": an array of short strings, one per reasoning step
  - "alice": the seat number (integer) Alice sits in
  - "bob":   the seat number (integer) Bob sits in
  - "carol": the seat number (integer) Carol sits in

Problem: Alice, Bob and Carol sit in seats 1, 2 and 3 (one person per seat).
Clues: Alice is NOT in seat 1. Bob sits to the left of Carol (a lower seat
number means further left). Carol is NOT in seat 3. The clues determine a
unique seating — work it out.
"""

T13_CHECKER = """\
import json as _json
import re as _re

def check(code_src, run):
    text = code_src.strip()
    m = _re.search(r"```(?:json)?\\s*\\n(.*?)```", text, _re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return {"pass": False, "detail": "response is not valid JSON"}
    if not isinstance(parsed, dict):
        return {"pass": False, "detail": "JSON must be an object"}
    steps = parsed.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return {"pass": False, "detail": f"steps must be a list of >=2 strings, got {steps!r}"}
    if not all(isinstance(s, str) for s in steps):
        return {"pass": False, "detail": "every step must be a string"}
    # Bob left of Carol + Carol not in 3 -> Bob=1, Carol=2; Alice not in 1 -> Alice=3.
    want = {"alice": 3, "bob": 1, "carol": 2}
    for person, seat in want.items():
        if parsed.get(person) != seat:
            return {"pass": False, "detail": f"{person} -> {parsed.get(person)!r}, wanted seat {seat}"}
    return {"pass": True, "detail": "steps + unique seating (Alice=3, Bob=1, Carol=2) verified"}
"""

DEV_TASKS: list[dict] = [
    {"id": "d1", "name": "implement parser function", "kind": "code",
     "prompt": T1_PROMPT, "checker_src": T1_CHECKER},
    {"id": "d2", "name": "fix buggy sum function", "kind": "debug",
     "prompt": T2_PROMPT, "checker_src": T2_CHECKER},
    {"id": "d3", "name": "refactor without loops", "kind": "refactor",
     "prompt": T3_PROMPT, "checker_src": T3_CHECKER},
    {"id": "d4", "name": "implement text analyzer module", "kind": "multi",
     "prompt": T4_PROMPT, "checker_src": T4_CHECKER},
    {"id": "d5", "name": "implement Counter class", "kind": "code",
     "prompt": T5_PROMPT, "checker_src": T5_CHECKER},
    {"id": "d6", "name": "multi-step reasoning (clock problem)", "kind": "thinking",
     "prompt": T6_PROMPT, "checker_src": T6_CHECKER},
    {"id": "d7", "name": "algorithm: longest increasing subsequence", "kind": "code",
     "prompt": T7_PROMPT, "checker_src": T7_CHECKER},
    {"id": "d8", "name": "algorithm: merge overlapping intervals", "kind": "code",
     "prompt": T8_PROMPT, "checker_src": T8_CHECKER},
    {"id": "d9", "name": "multi-step reasoning (train fare)", "kind": "thinking",
     "prompt": T9_PROMPT, "checker_src": T9_CHECKER},
    {"id": "d10", "name": "fix buggy cycle detector", "kind": "debug",
     "prompt": T10_PROMPT, "checker_src": T10_CHECKER},
    {"id": "d11", "name": "algorithm: 8-puzzle solvability (inversion parity)", "kind": "code",
     "prompt": T11_PROMPT, "checker_src": T11_CHECKER},
    {"id": "d12", "name": "multi-step reasoning (widgets rate trap)", "kind": "thinking",
     "prompt": T12_PROMPT, "checker_src": T12_CHECKER},
    {"id": "d13", "name": "constraint logic puzzle (seating)", "kind": "thinking",
     "prompt": T13_PROMPT, "checker_src": T13_CHECKER},
]


def extract_code(text: str) -> str:
    """Pull the most likely Python source out of a model response.

    Handles plain code, ```python fences, and prose-wrapped snippets.
    """
    import re
    text = text.strip()
    # Prefer a python fence if present.
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Otherwise, if the whole thing looks like code (has a def/def-like line),
    # drop leading/trailing prose lines so exec() only sees code.
    if re.search(r"^\s*(def |class |\w+\s*=\s*lambda)", text, re.MULTILINE):
        code_line = re.compile(r"^\s*(def |class |import |from |@|return |if |for |while |\w+\s*=)")
        lines = text.splitlines()
        # A line is "code" if it looks like a statement or is indented (block body).
        idx = [i for i, ln in enumerate(lines) if code_line.match(ln) or ln[:1] in (" ", "\t")]
        if idx:
            return "\n".join(lines[idx[0]:idx[-1] + 1]).strip()
        return text
    # Last resort: first fenced block of any language.
    m = re.search(r"```[a-z]*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text
