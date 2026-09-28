# ============================================================
# graph/tigergraph_client.py
# ============================================================

import os

from dotenv import load_dotenv
import pyTigerGraph as tg


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

TIGERGRAPH_HOST = os.getenv(
    "TIGERGRAPH_HOST"
)

TIGERGRAPH_GRAPH = os.getenv(
    "TIGERGRAPH_GRAPH",
    "Transaction_Fraud"
)

TIGERGRAPH_SECRET = os.getenv(
    "TIGERGRAPH_SECRET"
)

TIGERGRAPH_USERNAME = os.getenv(
    "TIGERGRAPH_USERNAME"
)

TIGERGRAPH_PASSWORD = os.getenv(
    "TIGERGRAPH_PASSWORD"
)


# ============================================================
# DEBUG CONFIG
# ============================================================

def _debug_configuration():

    print(
        "========== TIGERGRAPH CONNECTION =========="
    )

    print(
        "Host:",
        TIGERGRAPH_HOST
    )

    print(
        "Graph:",
        TIGERGRAPH_GRAPH
    )

    print(
        "Secret loaded:",
        bool(TIGERGRAPH_SECRET)
    )

    print(
        "Username loaded:",
        bool(TIGERGRAPH_USERNAME)
    )

    print(
        "Password loaded:",
        bool(TIGERGRAPH_PASSWORD)
    )


# ============================================================
# CONNECTION
# ============================================================

def get_tigergraph_connection():

    _debug_configuration()

    # --------------------------------------------------------
    # Validate host
    # --------------------------------------------------------

    if not TIGERGRAPH_HOST:

        raise RuntimeError(
            "TIGERGRAPH_HOST is missing.\n"
            "Set TIGERGRAPH_HOST in your .env file."
        )

    # --------------------------------------------------------
    # Validate credentials
    # --------------------------------------------------------

    if (
        not TIGERGRAPH_SECRET
        and not (
            TIGERGRAPH_USERNAME
            and TIGERGRAPH_PASSWORD
        )
    ):

        raise RuntimeError(
            "TigerGraph credentials are missing.\n\n"
            "Set ONE of the following in .env:\n"
            "TIGERGRAPH_SECRET=<secret>\n"
            "OR\n"
            "TIGERGRAPH_USERNAME=<username>\n"
            "TIGERGRAPH_PASSWORD=<password>"
        )

    # --------------------------------------------------------
    # Create connection
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # Secret-based connection
        # ----------------------------------------------------

        if TIGERGRAPH_SECRET:

            print(
                "Authentication mode: secret"
            )

            conn = tg.TigerGraphConnection(
                host=TIGERGRAPH_HOST,
                graphname=TIGERGRAPH_GRAPH,
                gsqlSecret=TIGERGRAPH_SECRET
            )

        # ----------------------------------------------------
        # Username/password connection
        # ----------------------------------------------------

        else:

            print(
                "Authentication mode: username/password"
            )

            conn = tg.TigerGraphConnection(
                host=TIGERGRAPH_HOST,
                graphname=TIGERGRAPH_GRAPH,
                username=TIGERGRAPH_USERNAME,
                password=TIGERGRAPH_PASSWORD
            )

        print(
            "✓ TigerGraph connection created."
        )

        return conn

    except Exception as e:

        print(
            "TigerGraph connection error:",
            type(e).__name__,
            ":",
            e
        )

        raise


# ============================================================
# TEST CONNECTION
# ============================================================

def test_tigergraph_connection():

    conn = get_tigergraph_connection()

    print()
    print(
        "========== TIGERGRAPH AUTH TEST =========="
    )

    try:

        # ----------------------------------------------------
        # Generate token using secret if available.
        # ----------------------------------------------------

        if TIGERGRAPH_SECRET:

            token = conn.getToken(
                TIGERGRAPH_SECRET,
                setToken=True
            )

        else:

            # pyTigerGraph handles username/password
            # authentication through the connection.
            token = conn.getToken(
                None,
                setToken=True
            )

        if token:

            print(
                "✓ TigerGraph authentication successful."
            )

            return True

        print(
            "✗ TigerGraph authentication returned no token."
        )

        return False

    except Exception as e:

        print(
            "✗ TigerGraph authentication failed:"
        )

        print(
            type(e).__name__,
            ":",
            e
        )

        return False


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    test_tigergraph_connection()