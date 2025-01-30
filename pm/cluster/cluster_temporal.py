from datetime import datetime, timedelta
from http.cookies import Morsel
from random import randrange
from typing import List, Dict
from enum import Enum, auto, StrEnum
from collections import defaultdict

from pydantic import BaseModel, Field

class TempRangeType(StrEnum):
    TIME_OF_DAY = auto()
    MORNING = auto()
    NOON = auto()
    AFTERNOON = auto()
    EVENING = auto()
    NIGHT = auto()
    DAY = auto()
    WEEK = auto()
    MONTH = auto()
    SEASON = auto()
    YEAR = auto()


def temp_cluster_to_level(trt: TempRangeType):
    if trt == TempRangeType.YEAR:
        return 1
    elif trt == TempRangeType.SEASON:
        return 2
    elif trt == TempRangeType.MONTH:
        return 3
    elif trt == TempRangeType.WEEK:
        return 4
    elif trt == TempRangeType.DAY:
        return 5
    else:
        return 6

class MiniContextCluster(BaseModel):
    id: int = Field(...)
    timestamp: datetime = Field(...)

class TemporalCluster:
    def __init__(self, timestamp_begin: datetime, timestamp_end: datetime, temp_range: TempRangeType, items: List, type: str, identifier: str, raw_type):
        self.timestamp_begin = timestamp_begin
        self.timestamp_end = timestamp_end
        self.temp_range = temp_range
        self.items = items
        self.type = type
        self.identifier = identifier
        self.raw_type = raw_type

    def __repr__(self):
        return (f"TemporalCluster({self.type}, " #{self.identifier}, "
                f"{self.timestamp_begin} - {self.timestamp_end}, "
                f"{len(self.items)} IDs)")

def categorize_time(timestamp: datetime) -> TempRangeType:
    """Categorize a timestamp into a time of day."""
    hour = timestamp.hour
    if 5 <= hour < 12:
        return TempRangeType.MORNING
    elif 12 <= hour < 14:
        return TempRangeType.NOON
    elif 14 <= hour < 18:
        return TempRangeType.AFTERNOON
    elif 18 <= hour < 22:
        return TempRangeType.EVENING
    else:
        return TempRangeType.NIGHT

def group_events(events: List[MiniContextCluster], grouping: TempRangeType) -> List[TemporalCluster]:
    """Group events by the specified temporal range."""
    events_sorted = sorted(events, key=lambda e: e.timestamp)  # Ensure events are sorted by time
    clusters = defaultdict(list)

    for event in events_sorted:
        timestamp = event.timestamp

        if grouping == TempRangeType.TIME_OF_DAY:
            range_type = f"{timestamp.date()} - " + str(categorize_time(timestamp).value)
            clusters[range_type].append(event)
        else:
            timestamp = timestamp.date()
            if grouping == TempRangeType.DAY:
                range_key = timestamp
                clusters[range_key].append(event)
            elif grouping == TempRangeType.WEEK:
                week_start = timestamp - timedelta(days=timestamp.weekday())  # Start of the week (Monday)
                clusters[week_start].append(event)
            elif grouping == TempRangeType.MONTH:
                month_start = datetime(timestamp.year, timestamp.month, 1)
                clusters[month_start].append(event)
            elif grouping == TempRangeType.SEASON:
                season = (timestamp.month % 12 + 3) // 3  # Map month to season
                clusters[(timestamp.year, season)].append(event)
            elif grouping == TempRangeType.YEAR:
                year_start = datetime(timestamp.year, 1, 1)
                clusters[year_start].append(event)

    temporal_clusters = [
        TemporalCluster(
            timestamp_begin=min([c.timestamp for c in cluster]),
            timestamp_end=max([c.timestamp for c in cluster]),
            temp_range=key,
            items=[c.id for c in cluster],
            type=grouping.value if grouping != TempRangeType.TIME_OF_DAY else key.split(" - ")[1],
            raw_type=grouping,
            identifier=key
        )
        for key, cluster in clusters.items()
    ]

    return temporal_clusters


def temporally_cluster_context(context_cluster: List[MiniContextCluster]) -> List[TemporalCluster]:
    res = []
    cts = [
        TempRangeType.TIME_OF_DAY,
        TempRangeType.DAY,
        TempRangeType.WEEK,
        TempRangeType.MONTH,
        TempRangeType.SEASON,
        TempRangeType.YEAR,
    ]
    for ct in cts:
        clusters = group_events(context_cluster, ct)
        for cluster in clusters:
            res.append(cluster)

    res.sort(key=lambda x: x.timestamp_begin)
    return res


if __name__ == '__main__':
    events = []
    dt = datetime(2025, 1, 22, 8, 0)
    for i in range(1000):
        dt += timedelta(minutes=randrange(5, 360))
        events.append(MiniContextCluster(id=i, timestamp=dt))

    clusters = temporally_cluster_context(events)
    for cluster in clusters:
        print(cluster)
