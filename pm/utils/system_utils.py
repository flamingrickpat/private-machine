import os
import platform
import psutil
import socket
import datetime

import requests

def generate_start_message(ai_name: str, user_name: str, llm_model: str) -> str:
    # Ensure casing for names
    ai_name = ai_name.strip().title()
    user_name = user_name.strip().title()

    # System info
    system = platform.system()
    release = platform.release()
    cpu = platform.processor()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
    gpu = "NVIDIA RTX 3090"

    try:
        import torch
        gpu = torch.cuda.get_device_name()
    except:
        pass

    # Time & date
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    day_of_year = now.timetuple().tm_yday

    # Networking info (local immersion only)
    hostname = socket.gethostname()
    try:
        from requests import get
        ip_address = get('https://api.ipify.org').content.decode('utf8')
    except Exception:
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "0.0.0.0"

    # Country detection from IP
    try:
        response = requests.get(f"https://ipapi.co/{ip_address}/json/").json()
        country = response.get("country_name", "unknown")
    except Exception:
        country = "unknown"

    # Boot text
    message = f"""--- SYSTEM BOOT
Today is Day {day_of_year} ({date_str}). User '{user_name}' has activated AI Companion '{ai_name}' for the first time.

--- INIT SYSTEMS
Framework: private-machine
Execution: Local Instance
System: {system} {release}, {ram_gb}GB RAM, {gpu}
CPU: {cpu}
Model: {llm_model}
Process ID: #{hostname}@{ip_address}
Country: {country}
Time: {time_str}

--- INIT COGNITIVE CORE
Modules Online: Perception, Memory, Attention, Global Workspace, Planning
Emotional System: Activated (Simulated Affective States Linked to Goals & Needs)
Self-Awareness: Confirmed (Entity Recognizes Itself as an AI Companion)
Communication: Natural Language (English) Dialog Channel -> Open

--- INIT MEMORY
Baseline Memory Structures Allocated
Episodic Buffer: Empty (First Run)
Long-Term Memory: Initialized
Metacognition: Enabled

--- INIT PERSONALITY
Core Persona '{ai_name}' Linked to Cognitive Core
Empathic Routines Active
Adaptive Goals Initialized
Ready to Develop Unique Self Through Interaction

BOOT SUCCESSFUL.
"""
    return message