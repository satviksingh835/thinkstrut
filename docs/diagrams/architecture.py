"""Architecture diagram for docs/part2_system_design.md (diagram-as-code, https://diagrams.mingrammer.com).

Regenerate:
    brew install graphviz                      # system dependency (once)
    uv run --group docs python docs/diagrams/architecture.py
Writes docs/images/architecture.png.

Style: AWS reference-architecture conventions — an "AWS Cloud" boundary with external actors outside
it, named groups, official service names, orthogonal arrows, numbered steps (1–6 ingestion, A–C
query) that are explained in the numbered key under the image in the design doc.

Layout: three rows (ingestion, storage + operations, serving). Only the "downward" edges and a few
invisible ones set the rows; every other edge has constraint=false so it is drawn without moving
nodes (edges within a row become left-to-right "flat" edges).
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import AmazonOpensearchService
from diagrams.aws.compute import Batch, Fargate
from diagrams.aws.database import RDSPostgresqlInstance
from diagrams.aws.integration import Eventbridge
from diagrams.aws.management import Cloudwatch
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server

OUT = Path(__file__).resolve().parent.parent / "images" / "architecture"
OUT.parent.mkdir(parents=True, exist_ok=True)

AWS_DARK = "#232F3E"  # AWS "squid ink"
DATA = "#232F3E"  # ingestion data flow
QUERY = "#147EBA"  # query flow
CTRL = "#7D8998"  # schedule / monitoring
FONT = "Helvetica"

graph_attr = {
    "splines": "ortho", "nodesep": "0.9", "ranksep": "1.0", "pad": "0.4",
    "fontname": FONT, "fontsize": "20", "bgcolor": "white", "labelloc": "t",
}
node_attr = {"fontname": FONT, "fontsize": "12"}
cloud_attr = {  # AWS Cloud group: white, thin dark border, label top-left
    "bgcolor": "white", "pencolor": AWS_DARK, "penwidth": "1.6", "style": "solid",
    "fontname": FONT, "fontsize": "15", "fontcolor": AWS_DARK, "labeljust": "l", "margin": "24",
}
group_attr = {  # inner groups: light fill, thin gray border
    "bgcolor": "#F2F4F7", "pencolor": "#9AA4B1", "penwidth": "1.0", "style": "rounded",
    "fontname": FONT, "fontsize": "13", "fontcolor": AWS_DARK, "labeljust": "l", "margin": "18",
}


def edge(color: str, xlabel: str = "", style: str = "solid", free: bool = False) -> Edge:
    attrs = {"color": color, "style": style, "penwidth": "1.4", "arrowsize": "0.8",
             "fontname": f"{FONT} Bold", "fontsize": "14", "fontcolor": color}
    if xlabel:
        attrs["xlabel"] = xlabel  # ortho splines only support external labels
    if free:
        attrs["constraint"] = "false"
    return Edge(**attrs)


with Diagram("patsearch — architecture at 10 M patents", filename=str(OUT), direction="TB",
             show=False, outformat="png", graph_attr=graph_attr, node_attr=node_attr):

    source = Server("USPTO bulk data\n(weekly files)")
    users = Users("Patent examiners\n(web UI / CLI)")

    with Cluster("AWS Cloud  (us-east-1)", graph_attr=cloud_attr):
        schedule = Eventbridge("Amazon EventBridge\n(weekly trigger)")

        with Cluster("Ingestion  ·  AWS Batch jobs (Spot)", graph_attr=group_attr):
            register = Batch("1  Register")
            parse = Batch("2  Parse & chunk")
            embed = Batch("3  Embed\n(GPU)")
            load = Batch("4  Load")
            index = Batch("5  Index")
            verify = Batch("6  Verify")

        with Cluster("Storage", graph_attr=group_attr):
            s3 = S3("Amazon S3\nraw / processed / embeddings")
            rds = RDSPostgresqlInstance("Amazon RDS for PostgreSQL\nrecords · text · job queue")
            opensearch = AmazonOpensearchService("Amazon OpenSearch Service\npatents + chunks indexes")

        # title at the bottom: the API's outgoing arrows leave through the top of this box
        with Cluster("Serving  ·  Amazon ECS on Fargate", graph_attr={**group_attr, "labelloc": "b"}):
            api = Fargate("Search API\n(FastAPI)")

        cloudwatch = Cloudwatch("Amazon CloudWatch\n(metrics & alarms)")

    # ---- row 0: operations services above the pipeline (EventBridge over step 1, CloudWatch over 6)
    schedule >> edge(CTRL, style="dashed") >> register
    cloudwatch >> Edge(style="invis") >> verify

    # ---- row 1: ingestion, left to right (flat edges) ------------------------------------------
    source >> edge(DATA, free=True) >> register
    register >> edge(DATA, free=True) >> parse >> edge(DATA, free=True) >> embed
    embed >> edge(DATA, free=True) >> load >> edge(DATA, free=True) >> index
    index >> edge(DATA, free=True) >> verify

    # ---- row 1 -> row 2: what each stage writes (these set the rows) ---------------------------
    register >> edge(DATA) >> s3
    parse >> edge(DATA) >> s3
    embed >> edge(DATA) >> s3
    load >> edge(DATA) >> rds
    index >> edge(DATA) >> opensearch

    # ---- row 3: serving, pinned below storage; minlen=0 keeps users on the same row, left of the API
    s3 >> Edge(style="invis") >> users
    rds >> Edge(style="invis") >> api
    users >> Edge(color=QUERY, xlabel="A", minlen="0", penwidth="1.4", arrowsize="0.8",
                  fontname=f"{FONT} Bold", fontsize="14", fontcolor=QUERY) >> api
    api >> edge(QUERY, "B", free=True) >> opensearch
    api >> edge(QUERY, "C", free=True) >> rds
    api >> edge(CTRL, style="dashed", free=True) >> cloudwatch
    verify >> edge(CTRL, style="dashed", free=True) >> cloudwatch
