"""Status dashboard (design Section 11): batches per state, per-stage progress, workers, retries.

Plain stdlib HTTP server; the page re-renders from Postgres on every request and refreshes every 2 s.
GET /            HTML dashboard
GET /api/status  the same data as JSON
"""

from __future__ import annotations

import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import db

PORT = int(os.environ.get("PORT", "8080"))
STAGES = ["parse", "embed", "load", "index", "verify"]
STATE_ORDER = ["registered", "parsed", "embedded", "loaded", "indexed", "verified", "needs_review", "failed"]


def snapshot(conn) -> dict:
    states = {r["state"]: r["n"] for r in conn.execute("SELECT state, count(*) AS n FROM batches GROUP BY state")}
    jobs = {r["status"]: r["n"] for r in conn.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status")}
    batches = conn.execute("""
        SELECT b.batch_id, b.pub_week::text AS week, b.state, b.stats,
               to_char(b.updated_at, 'HH24:MI:SS') AS updated,
               coalesce((SELECT json_object_agg(j.stage, j.status) FROM jobs j WHERE j.batch_id = b.batch_id),
                        '{}') AS stages
        FROM batches b ORDER BY b.batch_id DESC""").fetchall()
    incidents = conn.execute("""
        SELECT job_id, batch_id, stage, status, attempts, worker, last_error,
               to_char(coalesce(finished_at, heartbeat_at), 'HH24:MI:SS') AS at
        FROM jobs WHERE last_error IS NOT NULL ORDER BY job_id DESC LIMIT 25""").fetchall()
    workers = conn.execute("""
        SELECT worker, count(*) FILTER (WHERE status = 'done') AS done,
               (array_agg(stage ORDER BY started_at DESC) FILTER (WHERE status = 'running'))[1] AS current,
               to_char(max(heartbeat_at), 'HH24:MI:SS') AS last_seen
        FROM jobs WHERE worker IS NOT NULL GROUP BY worker ORDER BY worker""").fetchall()
    totals = conn.execute("""
        SELECT (SELECT count(*) FROM patents) AS patents, (SELECT count(*) FROM chunks) AS chunks,
               (SELECT coalesce(sum(attempts - 1), 0) FROM jobs WHERE attempts > 1) AS retries,
               (SELECT count(*) FROM jobs WHERE last_error LIKE 'worker lost%%') AS reaped,
               (SELECT extract(epoch FROM max(updated_at) - min(created_at)) FROM batches) AS elapsed_s
    """).fetchone()
    return {"states": states, "jobs": jobs, "totals": totals, "workers": workers,
            "batches": batches, "incidents": incidents}


CSS = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1c2230}
header{background:#232f3e;color:#fff;padding:14px 22px}header h1{font-size:17px;margin:0}
header p{margin:2px 0 0;color:#c6ccd6;font-size:12.5px}main{padding:18px 22px;max-width:1500px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:18px}
.tile{background:#fff;border:1px solid #e2e5eb;border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:22px}.tile span{color:#687385;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
h2{font-size:14px;margin:20px 0 8px}table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e5eb}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid #eef0f3;font-size:13px;vertical-align:top}
th{color:#687385;font-weight:600;background:#fafbfc}.pill{padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.verified,.done{background:#e2f3e8;color:#23784a}.running{background:#e9eefd;color:#2f5bea}
.queued,.registered,.parsed,.embedded,.loaded,.indexed{background:#eef0f3;color:#475467}
.failed{background:#fdeaea;color:#a11d1d}.needs_review{background:#fff5dc;color:#7a5200}
.stages{white-space:nowrap}.stages span{display:inline-block;width:46px;margin-right:2px;text-align:center;font-size:11px;padding:1px 0}
.nw{white-space:nowrap}.missing{background:#fff;color:#c3c8d1;border:1px dashed #d5d9e0}
.grid{display:grid;grid-template-columns:3fr 2fr;gap:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.muted{color:#687385}code{font-size:12px}
"""


def render(s: dict) -> str:
    t, states, jobs = s["totals"], s["states"], s["jobs"]
    n_batches = sum(states.values())
    tiles = [
        ("batches registered", n_batches), ("verified", states.get("verified", 0)),
        ("needs review", states.get("needs_review", 0)), ("failed", states.get("failed", 0)),
        ("patents loaded", t["patents"]), ("chunks loaded", t["chunks"]),
        ("jobs running", jobs.get("running", 0)), ("jobs queued", jobs.get("queued", 0)),
        ("retries", t["retries"]), ("jobs re-queued by reaper", t["reaped"]),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{v}</b><span>{html.escape(k)}</span></div>' for k, v in tiles)

    rows = []
    for b in s["batches"]:
        cells = "".join(
            f'<span class="pill {b["stages"].get(st, "missing")}">{st}</span>' for st in STAGES)
        st = b["stats"] or {}
        detail = []
        if "patents_loaded" in st:
            detail.append(f'{st["patents_loaded"]} patents, {st["chunks_loaded"]} chunks')
        elif "patents" in st:
            detail.append(f'{st["patents"]} patents, {st["chunks"]} chunks')
        if st.get("aliases"):
            detail.append("aliases: " + ", ".join(st["aliases"]))
        if st.get("review_reason"):
            detail.append("review: " + st["review_reason"])
        rows.append(f'<tr><td>{b["batch_id"]}</td><td class="nw">{b["week"] or "?"}</td>'
                    f'<td><span class="pill {b["state"]}">{b["state"]}</span></td>'
                    f'<td class="stages">{cells}</td><td class="muted">{html.escape("; ".join(detail))}</td>'
                    f'<td class="muted nw">{b["updated"]}</td></tr>')

    incidents = "".join(
        f'<tr><td>{i["job_id"]}</td><td>{i["batch_id"]}</td><td>{i["stage"]}</td>'
        f'<td><span class="pill {i["status"]}">{i["status"]}</span></td><td>{i["attempts"]}</td>'
        f'<td><code>{html.escape((i["last_error"] or "").split(": ", 1)[-1])}</code></td>'
        f'<td class="muted nw">{i["at"] or ""}</td></tr>'
        for i in s["incidents"]) or '<tr><td colspan="7" class="muted">none</td></tr>'
    workers = "".join(
        f'<tr><td><code>{html.escape(w["worker"])}</code></td><td>{w["done"]}</td>'
        f'<td>{w["current"] or "idle"}</td><td class="muted">{w["last_seen"]}</td></tr>'
        for w in s["workers"]) or '<tr><td colspan="4" class="muted">no workers yet</td></tr>'
    elapsed = t["elapsed_s"] or 0
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>patsearch pipeline status</title><style>{CSS}</style></head><body>
<header><h1>patsearch ingestion pipeline: proof of concept</h1>
<p>Postgres job queue · {len(s["workers"])} workers · elapsed {elapsed:.0f} s · refreshes every 2 s</p></header>
<main><div class="tiles">{tiles_html}</div>
<div class="grid"><div><h2>Batches (one per weekly file)</h2>
<table><tr><th>Batch</th><th>Week</th><th>State</th><th>Stages</th><th>Details</th><th>Updated</th></tr>
{"".join(rows) or '<tr><td colspan="6" class="muted">waiting for the scheduler</td></tr>'}</table></div>
<div><h2>Workers</h2><table><tr><th>Worker</th><th>Jobs done</th><th>Current stage</th><th>Last heartbeat</th></tr>
{workers}</table>
<h2>Errors and recoveries</h2><table><tr><th>Job</th><th>Batch</th><th>Stage</th><th>Status</th><th>Attempts</th>
<th>Last error</th><th>At</th></tr>{incidents}</table></div></div></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        conn = db.connect()
        try:
            snap = snapshot(conn)
        finally:
            conn.close()
        if self.path.startswith("/api/status"):
            body, ctype = json.dumps(snap, default=str).encode(), "application/json"
        else:
            body, ctype = render(snap).encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep container logs quiet
        pass


def main() -> None:
    print(f"[dashboard] http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
