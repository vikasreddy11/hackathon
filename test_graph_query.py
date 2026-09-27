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
    result = conn.getVertices("Merchant", limit=5)

    print("\n✅ Sample Merchant vertices:\n")

    for vertex in result:
        print(vertex)

except Exception as e:
    print("\n❌ Query failed.")
    print("Error:", e)