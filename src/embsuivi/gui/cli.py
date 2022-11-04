import argparse
import sys
from pathlib import Path

from embsuivi.gui.load_utils import LOAD_FUNCTIONS
from streamlit import cli as stcli

GUI_DIR = Path(__file__).parent
APP_FILE = GUI_DIR / "app.py"


cli_parser = argparse.ArgumentParser("embsuivi-compare")
cli_parser.add_argument("embeddings", nargs="+", help="embeddings files or directory")
cli_parser.add_argument("-n", "--names", nargs="+", help="embeddings names")
cli_parser.add_argument(
    "-f",
    "--formats",
    nargs="+",
    choices=list(LOAD_FUNCTIONS),
    help="embeddings formats",
)


def parse_args(args=None) -> argparse.Namespace:
    return cli_parser.parse_args(args)


def main():
    assert parse_args(sys.argv[1:])
    sys.argv = ["streamlit", "run", APP_FILE.as_posix(), "--", *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
