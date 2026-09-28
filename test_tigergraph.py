import os
from dotenv import load_dotenv
import pyTigerGraph

load_dotenv()

host = os.getenv("TG_HOST")
graphname = os.getenv("TG_GRAPHNAME")
secret = os.getenv("TG_SECRET")

print("Connecting to TigerGraph...")
print("Host:", host)
print("Graph:", graphname)

try:
    conn = pyTigerGraph.TigerGraphConnection(
        host=host,
        graphname=graphname,
        gsqlSecret=secret,
    )

    print("\nGetting token...")
    token = conn.getToken()

    print("\n✅ TigerGraph connection successful!")
    print("Graph:", graphname)
    print("Token received successfully.")

except Exception as e:
    print("\n❌ Connection failed.")
    print("Error:", e)