"""Start the private-machine terminal session.

``python main.py`` runs the companion assembly through the runtime CLI.
``python dev/smoke_demo.py`` runs the finite scenario checks.
"""
from pm.app import cli
from workflows import companion

if __name__ == "__main__":
    raise SystemExit(cli.main(assembly=companion))
