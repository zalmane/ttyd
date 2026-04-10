"""Direct LLM runner for CRL talk-to-your-data queries.

Calls Bedrock directly — no agent middleware. Handles:
  1. Send CRL entities + sources context + question to LLM
  2. LLM returns modified CRL entities
  3. Compile locally
  4. If errors, send errors back to LLM for fix
  5. Repeat up to max_attempts

Usage:
    from llm_runner import run_query
    result = run_query("What is our total ARR?", "subscription_arr")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import boto3

POLICY_DIR = Path(__file__).parent / "policies"

# Bedrock config
AWS_PROFILE = "DevProfile"
AWS_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_TOKENS = 4096
MAX_FIX_ATTEMPTS = 3

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        session = boto3.Session(profile_name=AWS_PROFILE)
        _bedrock_client = session.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def _call_llm(system: str, messages: list[dict], max_tokens: int = MAX_TOKENS) -> tuple[str, dict]:
    """Call Bedrock Claude. Returns (response_text, usage_dict)."""
    client = _get_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    resp = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=json.dumps(body),
    )
    result = json.loads(resp["body"].read())
    text = result["content"][0]["text"] if result.get("content") else ""
    usage = result.get("usage", {})
    return text, usage


SYSTEM_PROMPT = """You are a data modeling expert. You write CRL (Compact RiverLang) entity definitions.

## CRL Syntax
ent ID from MODEL.ENTITY [filter EXPR] [group FIELDS as GRP] [order by EXPRS] [limit N] [output] {
  [@ann] prop_id:type = expression
  rel rel_id -> one|many MODEL.ENTITY on base.X == target.Y
}

## Types & Annotations
Types: str int float date bool
Annotations: @key @dim @met (short for @dimension, @metric)

## Critical Rules
1. ALL property expressions MUST start with `base.` or use `.X` shorthand (= `base.X`)
2. Relations MUST have `on` clause: `on base.X == target.Y` (use `on 1 == 1` for cross-join)
3. Grouping: `group base.field1, base.field2 as grp` — fields MUST be direct properties of the base entity (not relation traversals). After grouping, access non-grouped fields via aggregation: `sum(base.grp.field)`
4. To aggregate ALL rows, use `group (1) as dummy as all_rows` on an ENTITY (not a source). If the base is a source, first create an intermediate entity, then group on that.
5. Date filtering: use `year()`, `month()`, `day()` functions — NOT string comparison. Examples:
   - `filter year(base.start_date) >= 2025` (dates in/after 2025)
   - `filter year(base.end_date) == 2025` (expiring in 2025)
   - `filter year(base.order_date) == 2025 and month(base.order_date) == 3` (March 2025)
   NEVER use `date("...")` — it does not exist. NEVER compare dates to strings.
6. Each new entity references MODEL.ENTITY_ID where MODEL is the model name

## Output Format
- You receive sources (read-only) and existing entities
- Return ONLY entity definitions — do NOT include sources or model wrapper
- Mark the answer entity with `output`
- Include ALL existing entities (modified or not) plus any new ones

## Patterns (generic — use MODEL_ID.ENTITY_ID for all references)

Filter + group:
ent result from model.base_entity filter CONDITION group base.dim_field as grp output {
  @dim dim_field:type = .dim_field
  @met total:float = sum(base.grp.measure_field)
}

Aggregate all rows into scalar — group (1):
ent totals from model.some_entity group (1) as d as all output {
  @met total:float = sum(base.all.amount)
}

Count distinct — dedup then count:
ent unique from model.some_entity dedup { key:str = .key }
ent result from model.unique group (1) as d as all output {
  @met cnt:int = count(base.all.key)
}

Top/bottom N:
ent result from model.entity order by base.field desc limit N output { ...props... }

Date filtering — use year()/month()/day(), NEVER string comparison:
ent result from model.entity filter year(base.dt) == 2025 and month(base.dt) >= 3 { ... }

Having (aggregate then filter on result):
ent agg from model.entity group base.key as grp { @dim key:str = .key  @met cnt:int = count(base.grp.id) }
ent filtered from model.agg filter base.cnt > 1 output { ... }

Cross-join two scalars (for ratios/comparisons):
ent a from model.entity_a { val:float = .val  rel to_b -> one model.entity_b on 1 == 1 }
ent result from model.a output { @met ratio:float = .val / base.to_b.val }

Join to enrich with related data:
ent enriched from model.base { id:str = .id  rel to_ref -> one model.ref on base.fk == target.pk }
ent result from model.enriched filter base.to_ref.field == "x" output { ... }"""


def run_query(
    question: str,
    policy_key: str,
    sources: str,
    entities: str,
    max_attempts: int = MAX_FIX_ATTEMPTS,
) -> dict:
    """Run a CRL query with compilation feedback loop.

    Args:
        question: Business question to answer
        policy_key: Policy name (for model ID in CRL)
        sources: CRL sources section (read-only context)
        entities: CRL entities section (editable)
        max_attempts: Max compilation fix attempts

    Returns:
        dict with: status, entities, elapsed_s, attempts, usage, errors
    """
    from compact_rl import compact_to_modelset, merge_crl, number_lines, format_compile_errors
    from riverlang.compiler.compiler import Compiler

    result = {"attempts": 0, "total_input_tokens": 0, "total_output_tokens": 0}
    t0 = time.time()

    # Initial request
    user_msg = f"Sources (read-only context — do NOT output these):\n```\n{sources}\n```\n\nCurrent entities:\n```\n{entities}\n```\n\nQuestion: {question}\n\nReturn only the modified/new entity definitions in CRL format."

    messages = [{"role": "user", "content": user_msg}]

    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt

        text, usage = _call_llm(SYSTEM_PROMPT, messages)
        result["total_input_tokens"] += usage.get("input_tokens", 0)
        result["total_output_tokens"] += usage.get("output_tokens", 0)

        # Extract CRL from response (may be in code fences)
        agent_entities = _extract_crl(text)

        # Try compile
        header = f"model {policy_key} {{"
        full_crl = merge_crl(header, sources, agent_entities)

        try:
            ms = compact_to_modelset(full_crl)
            cr = Compiler(ms).verify()
            if not cr.has_errors():
                result["status"] = "compiled"
                result["entities"] = agent_entities
                result["full_crl"] = full_crl
                result["elapsed_s"] = round(time.time() - t0, 1)
                result["compiler_result"] = cr
                result["model_set"] = ms
                return result
            errors = [str(e) for e in cr.errors]
        except Exception as e:
            errors = [str(e)]

        result["last_errors"] = errors

        if attempt >= max_attempts:
            break

        # Send errors back for fix
        error_feedback = format_compile_errors(full_crl, errors, attempt, max_attempts)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": error_feedback})

    result["status"] = "compile_fail"
    result["entities"] = agent_entities
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def _extract_crl(text: str) -> str:
    """Extract CRL from LLM response, handling code fences."""
    import re
    # Try code fences first
    m = re.search(r"```(?:crl|river|riverlang)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If no fences, return stripped text
    stripped = text.strip()
    # Remove any leading explanation text before first ent/const line
    lines = stripped.split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("ent ") or s.startswith("const ") or s.startswith("@"):
            start = i
            break
    return "\n".join(lines[start:]).strip()
