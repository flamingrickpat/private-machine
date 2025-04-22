import os

from pm.character import database_uri
from pm.system_read_chatlog import init_from_chatlog

if __name__ == '__main__':
    import sys
    # Accept path to CSV file as an optional argument
    chatlog_path = sys.argv[1] if len(sys.argv) > 1 else 'chatlog.csv'
    init_from_chatlog(chatlog_path)
