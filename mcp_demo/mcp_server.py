import os.path
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
from threading import Thread
from time import sleep
import logging
import os

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import requests
import fastmcp
from fastmcp import FastMCP
from trafilatura import extract

logger = logging.getLogger(__name__)

# --- Database Setup ---
DB_FILE = "mcp_data.db"

def get_db_conn():
    """Establishes a connection to the SQLite database."""
    path = os.path.join(os.path.dirname(__file__), DB_FILE)
    conn = sqlite3.connect(path)
    # This will allow you to access columns by name
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    logger.info("Initializing database...")
    conn = get_db_conn()
    cursor = conn.cursor()

    # Notes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        note_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        metadata TEXT -- Stored as a JSON string
    )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


# --- Server Initialization ---
mcp = FastMCP("Improved Persistent MCP Server")


# --- Helper Functions ---
def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a sqlite3.Row object to a dictionary."""
    if row is None:
        return None
    return dict(row)


# --- Notes CRUD ---
@mcp.tool()
def create_note(note_id: str, content: str) -> Dict[str, Any]:
    """
    Create a new note and save it to the database.
    :param note_id: Unique identifier for the note. A UUID is recommended.
    :param content: Text content of the note.
    :return: A confirmation dictionary.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO notes (note_id, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (note_id, content, now, now)
        )
        conn.commit()
        return {"status": "success", "note_id": note_id}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Note with id '{note_id}' already exists."}
    finally:
        conn.close()


@mcp.tool()
def get_note(note_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a note by its ID from the database.
    :param note_id: Unique identifier for the note.
    :return: Dictionary containing note fields or None if not found.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,))
        note = cursor.fetchone()
        return _row_to_dict(note)
    finally:
        conn.close()


@mcp.tool()
def update_note(note_id: str, content: str) -> Dict[str, Any]:
    """
    Update an existing note's content in the database.
    :param note_id: Unique identifier for the note.
    :param content: New text content of the note.
    :return: A confirmation dictionary.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "UPDATE notes SET content = ?, updated_at = ? WHERE note_id = ?",
            (content, now, note_id)
        )
        if cursor.rowcount == 0:
            return {"status": "error", "message": f"Note with id '{note_id}' not found."}
        conn.commit()
        return {"status": "success", "note_id": note_id, "updated_at": now}
    finally:
        conn.close()


@mcp.tool()
def delete_note(note_id: str) -> Dict[str, Any]:
    """
    Delete a note by its ID from the database.
    :param note_id: Unique identifier for the note.
    :return: A confirmation dictionary.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
        if cursor.rowcount == 0:
            return {"status": "error", "message": f"Note with id '{note_id}' not found."}
        conn.commit()
        return {"status": "success", "deleted_note_id": note_id}
    finally:
        conn.close()


# --- Calendar & Reminders CRUD ---
@mcp.tool()
def create_event(event_id: str, title: str, start: datetime, end: datetime, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a calendar event and save it to the database.
    :param event_id: Unique identifier for the event. A UUID is recommended.
    :param title: Title of the event.
    :param start: Event start datetime.
    :param end: Event end datetime.
    :param metadata: Additional event metadata (e.g., location, notes).
    :return: A confirmation dictionary.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO events (event_id, title, start_time, end_time, metadata) VALUES (?, ?, ?, ?, ?)",
            (event_id, title, start.isoformat(), end.isoformat(), json.dumps(metadata))
        )
        conn.commit()
        return {"status": "success", "event_id": event_id}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Event with id '{event_id}' already exists."}
    finally:
        conn.close()


@mcp.tool()
def get_event(event_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve an event by its ID from the database.
    :param event_id: Unique identifier for the event.
    :return: Dictionary containing event fields or None if not found.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
        event_row = cursor.fetchone()
        if event_row:
            event = _row_to_dict(event_row)
            event['metadata'] = json.loads(event['metadata'])  # Deserialize JSON
            return event
        return None
    finally:
        conn.close()


@mcp.tool()
def update_event(
        event_id: str,
        title: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update fields of an existing event in the database.
    :param event_id: Unique identifier for the event.
    :param title: New title (optional).
    :param start: New start datetime (optional).
    :param end: New end datetime (optional).
    :param metadata: New metadata dictionary (optional).
    :return: A confirmation dictionary.
    """
    updates = []
    params = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if start is not None:
        updates.append("start_time = ?")
        params.append(start.isoformat())
    if end is not None:
        updates.append("end_time = ?")
        params.append(end.isoformat())
    if metadata is not None:
        updates.append("metadata = ?")
        params.append(json.dumps(metadata))

    if not updates:
        return {"status": "no_change", "message": "No fields provided to update."}

    sql = f"UPDATE events SET {', '.join(updates)} WHERE event_id = ?"
    params.append(event_id)

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        if cursor.rowcount == 0:
            return {"status": "error", "message": f"Event with id '{event_id}' not found."}
        conn.commit()
        return {"status": "success", "event_id": event_id}
    finally:
        conn.close()


@mcp.tool()
def delete_event(event_id: str) -> Dict[str, Any]:
    """
    Delete an event by its ID from the database.
    :param event_id: Unique identifier for the event.
    :return: A confirmation dictionary.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        if cursor.rowcount == 0:
            return {"status": "error", "message": f"Event with id '{event_id}' not found."}
        conn.commit()
        return {"status": "success", "deleted_event_id": event_id}
    finally:
        conn.close()


# --- Search Operations ---
@mcp.tool()
def search_notes(query: str) -> List[Dict[str, Any]]:
    """
    Search notes where the content contains the given query string.
    :param query: Keyword or phrase to search for.
    :return: List of matching notes as dictionaries.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        search_term = f"%{query}%"
        cursor.execute("SELECT * FROM notes WHERE content LIKE ?", (search_term,))
        notes = cursor.fetchall()
        return [_row_to_dict(note) for note in notes]
    finally:
        conn.close()


@mcp.tool()
def search_events(query: str) -> List[Dict[str, Any]]:
    """
    Search calendar events where the title or metadata contains the query string.
    :param query: Keyword or phrase to search for.
    :return: List of matching events as dictionaries.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        search_term = f"%{query}%"
        cursor.execute(
            "SELECT * FROM events WHERE title LIKE ? OR metadata LIKE ?",
            (search_term, search_term)
        )
        event_rows = cursor.fetchall()
        results = []
        for row in event_rows:
            event = _row_to_dict(row)
            event['metadata'] = json.loads(event['metadata'])
            results.append(event)
        return results
    finally:
        conn.close()


# --- Upcoming Events / Reminders ---
@mcp.tool()
def check_upcoming_events(window_minutes: int = 60) -> List[Dict[str, Any]]:
    """
    List events or reminders starting within the next time window.
    :param window_minutes: Number of minutes ahead to check.
    :return: List of upcoming events/reminders as dictionaries.
    """
    now = datetime.now()
    end_window = now + timedelta(minutes=window_minutes)

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time ASC",
            (now.isoformat(), end_window.isoformat())
        )
        event_rows = cursor.fetchall()
        results = []
        for row in event_rows:
            event = _row_to_dict(row)
            event['metadata'] = json.loads(event['metadata'])
            results.append(event)
        return results
    finally:
        conn.close()


# --- DUMMY IMPLEMENTATIONS FOR COMPLEX ACTIONS ---

# --- Device information ---
@mcp.tool()
def get_device_info(user_id: str) -> Dict[str, Any]:
    """
    Retrieve dummy device metadata for a given user.
    :param user_id: Identifier for the user.
    :return: Dictionary with plausible device info.
    """
    logger.info(f"INFO: Faking device info for user '{user_id}'")
    return {
        "user_id": user_id,
        "device_type": "smartphone",
        "os": "Android 14",
        "model": "Pixel 8 Pro",
        "location": {
            "city": "Mountain View",
            "country": "USA",
            "latitude": 37.3861,
            "longitude": -122.0839,
        },
        "battery_level": 0.85,
        "is_charging": False,
        "network": "WiFi"
    }


# --- Conversation initiation ---
@mcp.tool()
def init_conversation(user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize a dummy conversation session with the user.
    :param user_id: Identifier for the user.
    :param context: Optional context data to seed the conversation.
    :return: Initial conversation payload.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"INFO: Faking conversation init for user '{user_id}' with session '{session_id}'")
    return {
        "status": "success",
        "message": "Conversation session initialized.",
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }


# --- Home automation controls ---
_DUMMY_ROOMS = {
    "Living Room": {"temperature": 21.5, "humidity": 45.0, "light_on": False},
    "Bedroom": {"temperature": 20.0, "humidity": 50.0, "light_on": False},
    "Kitchen": {"temperature": 22.0, "humidity": 55.0, "light_on": True},
    "Office": {"temperature": 21.0, "humidity": 48.0, "light_on": True},
}


@mcp.tool()
def list_rooms() -> List[str]:
    """
    List all rooms available for home automation (dummy data).
    :return: List of room names.
    """
    return list(_DUMMY_ROOMS.keys())


@mcp.tool()
def get_room_status(room: str) -> Dict[str, Any]:
    """
    Get current status for a specific room (dummy data).
    :param room: Name of the room.
    :return: Dictionary with status fields, or an error if room is not found.
    """
    if room in _DUMMY_ROOMS:
        return _DUMMY_ROOMS[room]
    return {"error": "Room not found", "room": room}


@mcp.tool()
def switch_light(room: str, on: bool) -> Dict[str, Any]:
    """
    Turn lights on or off in a specific room (dummy action).
    :param room: Name of the room.
    :param on: True to switch lights on, False to switch off.
    :return: Success status.
    """
    if room in _DUMMY_ROOMS:
        logger.info(f"ACTION: Faking light switch in '{room}' to {'ON' if on else 'OFF'}.")
        _DUMMY_ROOMS[room]['light_on'] = on
        return {"status": "success", "room": room, "light_on": on}
    return {"status": "error", "message": "Room not found", "room": room}


@mcp.tool()
def set_thermostat(room: str, temperature: float) -> Dict[str, Any]:
    """
    Set the thermostat temperature for a specific room (dummy action).
    :param room: Name of the room.
    :param temperature: Desired temperature in degrees Celsius.
    :return: Success status.
    """
    if room in _DUMMY_ROOMS:
        logger.info(f"ACTION: Faking thermostat set in '{room}' to {temperature}°C.")
        _DUMMY_ROOMS[room]['temperature'] = temperature
        return {"status": "success", "room": room, "temperature_set_to": temperature}
    return {"status": "error", "message": "Room not found", "room": room}


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo and returns the top results.
    :param query: The search query string.
    :param max_results: The maximum number of results to return.
    :return: A list of search results, each with a 'title', 'href' (URL), and 'body' (snippet).
    """
    logger.info(f"WEB_SEARCH: Performing search for '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        return results
    except Exception as e:
        logger.info(f"WEB_SEARCH: Error during search: {e}")
        return [{"error": "Failed to perform web search.", "details": str(e)}]


@mcp.tool()
def browse_website(url: str) -> Dict[str, Any]:
    """
    Fetches the content of a URL, extracts the main text, and lists all links.
    This is ideal for an LLM to "read" a page without the clutter of HTML.
    :param url: The URL of the website to browse.
    :return: A dictionary containing the page's main text and a list of links.
    """
    logger.info(f"WEB_BROWSE: Browsing URL '{url}'")
    try:
        # Set a user-agent to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Fetch the webpage content using requests
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        html_content = response.text

        # 1. Extract the main, clean text using trafilatura
        main_text = extract(html_content, include_comments=False, include_tables=True)
        if not main_text:
            # Fallback for pages where trafilatura might fail (e.g., simple pages)
            soup_fallback = BeautifulSoup(html_content, 'html.parser')
            main_text = soup_fallback.get_text(separator='\n', strip=True)

        # 2. Extract all hyperlinks from the page using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            link_text = a_tag.get_text(strip=True)
            link_href = a_tag['href']
            if link_text and link_href and not link_href.startswith('#'):
                links.append({"text": link_text, "href": link_href})

        return {
            "status": "success",
            "url": url,
            "main_text": main_text,
            "links": links[:30]  # Limit number of links to avoid overwhelming the LLM
        }
    except Exception as e:
        logger.info(f"WEB_BROWSE: An unexpected error occurred: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


def threaded_function(arg):
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp", uvicorn_config={"log_level": "critical"})

def start_server():
    init_db()
    thread = Thread(target = threaded_function, args = (10, ), daemon=True)
    thread.start()
    logger.info("server started...")

if __name__ == "__main__":
    start_server()
    while True:
        time.sleep(1)