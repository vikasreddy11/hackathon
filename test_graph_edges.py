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

merchant_id = "fraud_Hahn, Bahringer and McLaughlin"

try:
    edges = conn.getEdges(
        "Merchant",
        merchant_id
    )

    print(f"\n✅ Relationships for {merchant_id}:\n")

    for edge in edges[:10]:
        print(edge)

except Exception as e:
    print("\n❌ Edge query failed.")
    print("Error:", e)