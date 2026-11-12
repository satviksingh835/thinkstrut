"""Architecture diagram for docs/part2_system_design.md (diagram-as-code, https://diagrams.mingrammer.com).

Regenerate:
    brew install graphviz                      # system dependency (once)
    uv run --group docs python docs/diagrams/architecture.py
Writes docs/images/architecture.png.

Layout: three rows, top to bottom — ingestion stages (left to right), data stores, query path.
Only "downward" edges (stage writes a store; invisible store -> query-row links) set the rows. Every
other edge uses constraint=false, so it is drawn without changing the layout: edges inside a row
(stage hand-offs, user -> API) become left-to-right "flat" edges, and upward edges (a stage reading a
store, the API calling a store) don't pull rows out of order.
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import AmazonOpensearchService
from diagrams.aws.compute import EC2, ECS
from diagrams.aws.database import RDSPostgresqlInstance
from diagrams.aws.integration import Eventbridge
from diagrams.aws.management import Cloudwatch
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.programming.framework import Fastapi

OUT = Path(__file__).resolve().parent.parent / "images" / "architecture"
OUT.parent.mkdir(parents=True, exist_ok=True)

INGEST = "#2f5bea"  # ingestion data flow (writes)
READ = "#7d9bf0"  # a stage reading its input
QUERY = "#12a087"  # query flow
CONTROL = "#8a94a6"  # job hand-off / schedule / metrics

graph_attr = {"fontsize": "24", "pad": "0.5", "nodesep": "0.9", "ranksep": "1.1", "splines": "spline",
              "labelloc": "t"}
row = {"fontsize": "15", "margin": "20"}


def e(color: str, label: str = "", style: str = "solid", free: bool = False) -> Edge:
    """free=True: draw the edge but don't let it influence the row layout (constraint=false)."""
    kw = {"constraint": "false"} if free else {}
    return Edge(color=color, label=label, style=style, fontsize="11", fontcolor=color, penwidth="1.7", **kw)


with Diagram("patsearch at 10 M patents", filename=str(OUT), direction="TB", show=False,
             outformat="png", graph_attr=graph_attr):

    # ---- row 1: ingestion ---------------------------------------------------------------------
    with Cluster("Ingestion  —  batch workers on ECS / AWS Batch  ·  one batch = one weekly file  ·  "
                 "every stage idempotent", graph_attr=row):
        schedule = Eventbridge("weekly\nschedule")
        source = Server("USPTO weekly\nbulk files")
        register = ECS("1 register\nsha256 + batch row")
        parse = ECS("2 parse · clean · chunk\nPart 1 code")
        embed = EC2("3 embed\nGPU spot (backfill)\nCPU (weekly)")
        load = ECS("4 load\none txn per batch")
        index = ECS("5 index\nbulk upsert")
        verify = ECS("6 verify\ncounts + canaries")

    # ---- row 2: data stores -----------------------------------------------------------------------
    with Cluster("Data stores", graph_attr=row):
        raw = S3("S3 raw/\nimmutable")
        processed = S3("S3 processed/\npatents + chunks parquet")
        embeddings = S3("S3 embeddings/<model>/\nfp16 parquet")
        postgres = RDSPostgresqlInstance("Postgres\nrecords · claim + chunk text\njob queue + batch status")
        opensearch = AmazonOpensearchService("OpenSearch\npatents: BM25 + filters\n"
                                             "chunks: 378 M 1-bit vectors\n+ CPC / date filters")
        monitoring = Cloudwatch("CloudWatch\nmetrics · dashboards · alerts")

    # ---- row 3: query path --------------------------------------------------------------------------
    with Cluster("Query path", graph_attr=row):
        users = Users("examiners\nweb UI · CLI")
        api = Fastapi("API (CPU containers)\nembed query (ONNX) · plan filters\ngroup by patent · evidence")

    # stage hand-offs (left to right in row 1): the next stage's job is enqueued in Postgres
    schedule >> e(CONTROL, style="dashed", free=True) >> source
    source >> e(INGEST, free=True) >> register
    register >> e(CONTROL, "next job", "dashed", free=True) >> parse
    parse >> e(CONTROL, style="dashed", free=True) >> embed
    embed >> e(CONTROL, style="dashed", free=True) >> load
    load >> e(CONTROL, style="dashed", free=True) >> index
    index >> e(CONTROL, style="dashed", free=True) >> verify

    # writes (stage -> store, downward)
    register >> e(INGEST) >> raw
    parse >> e(INGEST) >> processed
    embed >> e(INGEST) >> embeddings
    load >> e(INGEST) >> postgres
    index >> e(INGEST) >> opensearch
    # (verify also runs canary queries against OpenSearch — stated in its label; an extra edge here
    #  makes Graphviz re-route the read edges around the diagram)
    verify >> e(CONTROL, style="dotted") >> monitoring

    # reads (store -> stage, upward; don't affect layout)
    raw >> e(READ, style="dashed", free=True) >> parse
    processed >> e(READ, style="dashed", free=True) >> embed
    processed >> e(READ, style="dashed", free=True) >> load
    embeddings >> e(READ, style="dashed", free=True) >> index

    # keep the query row below the stores
    postgres >> Edge(style="invis") >> users
    opensearch >> Edge(style="invis") >> api

    # query flow
    users >> e(QUERY, "query + filters", free=True) >> api
    api >> e(QUERY, "1 keyword filter → doc ids\n2 filtered ANN on chunks", free=True) >> opensearch
    api >> e(QUERY, "3 titles + evidence text", free=True) >> postgres
    api >> e(CONTROL, style="dotted", free=True) >> monitoring
