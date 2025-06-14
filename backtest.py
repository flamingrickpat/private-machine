import sys

from pm_lida import run_shell

if __name__ == '__main__':
    db_path = sys.argv[1]
    run_shell(False, db_path)