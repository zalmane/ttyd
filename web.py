#!/usr/bin/env python3
"""Simple web UI for Talk to Your Data.

Run: uv run python web.py
Open: http://localhost:8899
"""

from __future__ import annotations

import json
import sys
import time
import html
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from setup_db import create_db
from compact_rl import riverlang_to_compact, compact_to_modelset, split_crl, merge_crl
from llm_runner import run_query
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

# Init DB
create_db()
DB_PATH = str(Path(__file__).parent / "demo.db")

POLICY_DIR = Path(__file__).parent / "policies"
POLICIES_RL = {p.stem: p.read_text() for p in sorted(POLICY_DIR.glob("*.river"))}
POLICIES_CRL = {k: riverlang_to_compact(v) for k, v in POLICIES_RL.items()}

MODEL_ID = "query"
ALL_SOURCES = open(Path(__file__).parent / "ask.py").read().split('ALL_SOURCES = """')[1].split('"""')[0]

# Import planner from ask.py
from ask import plan, CONCEPT_CATALOG, execute_reuse, execute_patch, execute_fresh, _last


def run_question(question: str) -> dict:
    """Run a question and capture all output."""
    con = duckdb.connect(DB_PATH, read_only=True)
    t0 = time.time()

    # Plan
    p = plan(question)
    approach = p.get("approach", "fresh")
    policy = p.get("policy")
    reasoning = p.get("reasoning", "")

    result = {
        "question": question,
        "plan": {
            "approach": approach,
            "policy": policy,
            "reasoning": reasoning,
            "time": p.get("_time", 0),
            "tokens": p.get("_tokens", 0),
        },
        "entities_crl": None,
        "sql": None,
        "rows": None,
        "columns": None,
        "error": None,
    }

    # Execute
    try:
        if approach == "reuse" and policy and policy in POLICIES_RL:
            ms = RiverLangParser.from_str(POLICIES_RL[policy])
            cr = Compiler(ms).verify()
            if not cr.has_errors():
                _last["model_set"] = ms
                _last["compiler_result"] = cr
                _last["con"] = con
                model = ms.models[0]
                entities = [e.split(".")[-1] for e in p.get("entities", [])]
                target = [e for e in model.elements
                          if hasattr(e.element, "is_output") and e.element.is_output
                          and (not entities or e.id in entities)]
                if target:
                    ent = target[0]
                    sql = cr.get_sql_generator().generate_sql(FQN((model.id, ent.id)), dialect="duckdb")
                    result["sql"] = sql
                    rows = con.execute(sql).fetchall()
                    result["columns"] = [d[0] for d in con.description]
                    result["rows"] = [[str(v) for v in row] for row in rows]
            else:
                approach = "fresh"

        if approach == "patch" and policy and policy in POLICIES_CRL:
            crl = POLICIES_CRL[policy]
            header, sources, entities = split_crl(crl)
            qr = run_query(p.get("description", question), policy, sources, entities)
            if qr["status"] == "compiled":
                result["entities_crl"] = qr.get("entities", "")
                cr = qr["compiler_result"]
                ms = qr["model_set"]
                _last["model_set"] = ms
                _last["compiler_result"] = cr
                _last["con"] = con
                model = ms.models[0]
                orig_ids = set(e.split(".")[-1] for e in split_crl(crl)[2].split("\n") if e.strip().startswith("ent "))
                output_ents = [e for e in model.elements
                               if hasattr(e.element, "is_output") and e.element.is_output and e.id not in orig_ids]
                if not output_ents:
                    output_ents = [e for e in model.elements if hasattr(e.element, "is_output") and e.element.is_output]
                if output_ents:
                    ent = output_ents[0]
                    sql = cr.get_sql_generator().generate_sql(FQN((model.id, ent.id)), dialect="duckdb")
                    result["sql"] = sql
                    rows = con.execute(sql).fetchall()
                    result["columns"] = [d[0] for d in con.description]
                    result["rows"] = [[str(v) for v in row] for row in rows]
            else:
                approach = "fresh"

        if approach == "fresh" or (result["rows"] is None and result["error"] is None):
            qr = run_query(p.get("description", question), MODEL_ID, ALL_SOURCES, entities="")
            if qr["status"] == "compiled":
                result["entities_crl"] = qr.get("entities", "")
                cr = qr["compiler_result"]
                ms = qr["model_set"]
                _last["model_set"] = ms
                _last["compiler_result"] = cr
                _last["con"] = con
                model = ms.models[0]
                output_ents = [e for e in model.elements if hasattr(e.element, "is_output") and e.element.is_output]
                if output_ents:
                    ent = output_ents[0]
                    sql = cr.get_sql_generator().generate_sql(FQN((model.id, ent.id)), dialect="duckdb")
                    result["sql"] = sql
                    rows = con.execute(sql).fetchall()
                    result["columns"] = [d[0] for d in con.description]
                    result["rows"] = [[str(v) for v in row] for row in rows]
            else:
                result["error"] = "; ".join(qr.get("last_errors", ["Compilation failed"])[:2])

    except Exception as e:
        result["error"] = str(e)

    result["total_time"] = round(time.time() - t0, 1)
    con.close()
    return result


HTML = """<!DOCTYPE html>
<html>
<head>
<title>Talk to Your Data</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 960px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 8px; }
  .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
  .glossary { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .glossary h3 { color: #58a6ff; font-size: 13px; margin-bottom: 8px; }
  .glossary .concept { display: inline-block; background: #1f6feb22; border: 1px solid #1f6feb44; border-radius: 4px; padding: 2px 8px; margin: 2px; font-size: 12px; color: #58a6ff; }
  .input-row { display: flex; gap: 8px; margin-bottom: 20px; }
  input[type=text] { flex: 1; padding: 10px 14px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; font-size: 15px; outline: none; }
  input[type=text]:focus { border-color: #58a6ff; }
  button { padding: 10px 20px; background: #238636; border: none; border-radius: 6px; color: white; font-size: 15px; cursor: pointer; }
  button:hover { background: #2ea043; }
  button:disabled { background: #30363d; cursor: wait; }
  .result { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .plan { font-size: 13px; color: #8b949e; margin-bottom: 12px; }
  .plan .approach { font-weight: bold; }
  .plan .reuse { color: #3fb950; }
  .plan .patch { color: #d29922; }
  .plan .fresh { color: #58a6ff; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th { text-align: left; padding: 6px 12px; border-bottom: 2px solid #30363d; color: #58a6ff; font-size: 13px; }
  td { padding: 6px 12px; border-bottom: 1px solid #21262d; font-size: 13px; font-family: monospace; }
  tr:hover td { background: #1f242b; }
  pre { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 12px; overflow-x: auto; color: #8b949e; margin: 8px 0; }
  .toggle { cursor: pointer; color: #58a6ff; font-size: 12px; }
  .error { color: #f85149; }
  .timing { color: #8b949e; font-size: 12px; margin-top: 8px; }
  .spinner { display: none; color: #58a6ff; }
</style>
</head>
<body>
<h1>Talk to Your Data</h1>
<div class="subtitle">orders (54) &middot; customers (10) &middot; subscriptions (15) &middot; products (5) &middot; Jul 2024 - Apr 2025</div>

<div class="glossary">
<h3>Glossary</h3>
GLOSSARY_PLACEHOLDER
</div>

<div class="input-row">
  <input type="text" id="q" placeholder="Ask a question..." autofocus>
  <button id="btn" onclick="ask()">Ask</button>
</div>
<div class="spinner" id="spinner">Thinking...</div>
<div id="results"></div>

<script>
const q = document.getElementById('q');
const btn = document.getElementById('btn');
const spinner = document.getElementById('spinner');
q.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

async function ask() {
  const question = q.value.trim();
  if (!question) return;
  btn.disabled = true;
  spinner.style.display = 'block';
  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await resp.json();
    render(data);
  } catch(e) {
    document.getElementById('results').innerHTML = '<div class="result error">Error: '+e+'</div>';
  }
  btn.disabled = false;
  spinner.style.display = 'none';
}

function render(d) {
  let h = '<div class="result">';
  // Plan
  const ac = d.plan.approach;
  h += '<div class="plan"><span class="approach '+ac+'">'+ac+'</span>';
  if (d.plan.policy) h += ' &rarr; '+d.plan.policy;
  h += ' <span>('+d.plan.time+'s, '+d.plan.tokens+' tok)</span></div>';
  h += '<div class="plan">'+esc(d.plan.reasoning)+'</div>';

  // Table
  if (d.rows && d.columns) {
    h += '<table><tr>';
    d.columns.forEach(c => h += '<th>'+esc(c)+'</th>');
    h += '</tr>';
    d.rows.forEach(r => {
      h += '<tr>';
      r.forEach(v => h += '<td>'+esc(v)+'</td>');
      h += '</tr>';
    });
    h += '</table>';
  }

  if (d.error) h += '<div class="error">'+esc(d.error)+'</div>';

  // Toggles
  if (d.entities_crl) h += '<div><span class="toggle" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\\'none\\'?\\'block\\':\\'none\\'">CRL &#9662;</span><pre style="display:none">'+esc(d.entities_crl)+'</pre></div>';
  if (d.sql) h += '<div><span class="toggle" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\\'none\\'?\\'block\\':\\'none\\'">SQL &#9662;</span><pre style="display:none">'+esc(d.sql)+'</pre></div>';

  h += '<div class="timing">'+d.total_time+'s</div>';
  h += '</div>';
  document.getElementById('results').insertAdjacentHTML('afterbegin', h);
}

function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        glossary_html = ""
        for c in CONCEPT_CATALOG:
            if c["output"]:
                name = c["entity"].replace("_", " ").title()
                glossary_html += f'<span class="concept">{html.escape(name)}</span>\n'
        page = HTML.replace("GLOSSARY_PLACEHOLDER", glossary_html)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(page.encode())

    def do_POST(self):
        if self.path == "/ask":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            result = run_question(body["question"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"  {args[0]}", flush=True)


def main():
    port = 8899
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Talk to Your Data — http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
