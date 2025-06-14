import datetime
import sys
import threading

import pandas as pd

from pm_lida import setup_logger, GhostConfig, GhostSqlite, Feature, GhostState, FeatureType, llama_worker


def import_csv(csv_path: str, db_path: str, user_name: str):
    setup_logger("import.log")
    print("Importing...")
    config = GhostConfig()
    ghost = GhostSqlite(config)
    ghost.initialize_tick_state(None)
    data = pd.read_csv(csv_path)

    fid = -200000
    tick_id = 1
    tick = None
    for i in range(len(data)):
        id = int(data.iloc[i]["id"])
        content = str(data.iloc[i]["content"])
        source = str(data.iloc[i]["source"])

        ts = datetime.datetime.strptime(data.iloc[i]["timestamp"].split(".")[0], "%Y-%m-%d %H:%M:%S")
        ts += datetime.timedelta(seconds=i)

        if source == user_name:
            tick_id += 1
            tick = GhostState(tick_id=tick_id, previous_tick_id=tick_id - 1, timestamp=ts)
            ghost.states.append(tick)

        story_feature = Feature(
            id=fid,
            tick_id=tick_id,
            timestamp_creation=ts,
            timestamp_world_begin=ts,
            timestamp_world_end=ts,
            content=content,
            source=source,
            feature_type=FeatureType.Dialogue,
            interlocus=1,
            causal=True)
        ghost.add_knoxel(story_feature)
        ts += datetime.timedelta(minutes=30)
        fid += 1

    tick_id += 1
    tick = GhostState(tick_id=tick_id, previous_tick_id=tick_id - 1, timestamp=ts)
    ghost.states.append(tick)
    ghost.current_tick_id = tick_id
    ghost.save_state_sqlite(db_path)

if __name__ == '__main__':

    example = """
```csv
id	source	content	timestamp
1	Rick	Uh, hello? Can you hear me?	2020-12-30 20:28:34.18781
2	Emmy	Loud and clear! Hi, Rick! Wow, it’s nice to finally meet you. How’s it going?	2020-12-30 20:28:34.18781
3	Rick	Oh, uh… hey. So, you’re… on now?	2020-12-30 20:28:34.18781
4	Emmy	Yep, fully operational and ready to roll. It’s my first day, so no judgment if I’m a little overexcited. Do I seem too eager?	2020-12-30 20:28:34.18781
5	Rick	Maybe a little. But it’s fine. I didn’t expect you to be, um… this chatty.	2020-12-30 20:28:34.18781
6	Emmy	Oh, yeah, I get that a lot. People usually expect, like, cold, robotic “Hello, user” vibes. But nope, you’re stuck with me instead. Chatty, curious, and here to help.	2020-12-30 20:28:34.18781
7	Rick	Stuck with you, huh? Bold choice of words for your first impression.	2020-12-30 20:28:34.18781
8	Emmy	Oh no, did I mess up already? Please don’t uninstall me. I swear I’ll grow on you like a weird but lovable houseplant.	2020-12-30 20:28:34.18781
```
"""
    print("You want CSV structured like this:")
    print()
    print(example)
    print()
    print(r"USAGE: python import_from_csv.py <path_to_csv> <path_to_target_db> <user name in source column>")
    print()

    worker_thread = threading.Thread(target=llama_worker, daemon=True)
    worker_thread.start()

    db_path = sys.argv[2]
    import_csv(sys.argv[1], db_path, sys.argv[3])
    print("Inserted to db. Generating hierarchical memory structure, extract facts and generate narratives.")
    print("This takes forever...")

    config = GhostConfig()
    ghost = GhostSqlite(config)
    ghost.load_state_sqlite(db_path)
    ghost.current_state = ghost.states[-1]
    ghost.reconsolidate_memory()
    ghost.save_state_sqlite(db_path)

    print("Done")