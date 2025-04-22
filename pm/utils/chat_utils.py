from pydantic import BaseModel
from typing import List
from datetime import datetime
import csv


class ChatTurn(BaseModel):
    turn: str
    content: str
    timestamp: datetime


def parse_chatlog(csv_file_path: str) -> List[ChatTurn]:
    """
    Parse a CSV chat log and return a list of ChatTurn models.

    Consecutive messages from the same source are merged into a single turn.
    """
    turns: List[ChatTurn] = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        current_source = None
        current_content_parts: List[str] = []
        current_timestamp: datetime = None
        turn_counter = 1

        for row in reader:
            source = row['source']
            content = row['content']
            # Parse timestamp assuming ISO format
            timestamp = datetime.fromisoformat(row['timestamp'])

            if source == current_source:
                # Same speaker, accumulate content
                current_content_parts.append(content)
            else:
                # Different speaker, finalize previous turn
                if current_source is not None:
                    turns.append(ChatTurn(
                        turn=current_source,
                        content=' '.join(current_content_parts),
                        timestamp=current_timestamp
                    ))
                    turn_counter += 1

                # Start new turn for the new speaker
                current_source = source
                current_content_parts = [content]
                current_timestamp = timestamp

        # Finalize the last turn after loop
        if current_source is not None:
            turns.append(ChatTurn(
                turn=current_source,
                content=' '.join(current_content_parts),
                timestamp=current_timestamp
            ))

    return turns


if __name__ == '__main__':
    import sys
    # Accept path to CSV file as an optional argument
    chatlog_path = sys.argv[1] if len(sys.argv) > 1 else 'chatlog.csv'
    turns = parse_chatlog(chatlog_path)
    for t in turns:
        # Print each turn as JSON
        print(t.json())
