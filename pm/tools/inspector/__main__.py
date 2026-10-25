"""Run the inspector straight from the command line: ``python -m pm.tools.inspector``."""
from __future__ import annotations

import sys

from pm.tools.inspector.launch import main

if __name__ == "__main__":
    sys.exit(main())
