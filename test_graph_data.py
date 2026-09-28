import os
from dotenv import load_dotenv
import pyTigerGraph

load_dotenv()

conn = pyTigerGraph.TigerGraphConnection(
    host=os.getenv("TG_HOST"),
    graphname=os.getenv("TG_GRAPHNAME"),
    gsqlSecret=os.getenv("TG_SECRET"),
)

print("Connected to:", os.getenv("TG_GRAPHNAME"))

try:
    counts = conn.getVertexCount()

    print("\n✅ Vertex counts:\n")

    for vertex_type, count in counts.items():
        print(f"{vertex_type}: {count}")

except Exception as e:
    print("\n❌ Could not retrieve vertex counts.")
    print("Error:", e)