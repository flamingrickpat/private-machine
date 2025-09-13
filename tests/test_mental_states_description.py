import datetime as dt
import os
import sys
sys.path.append("..")
os.chdir("..")

from pm.mental_states import *
from pm.mental_states import _describe_emotion_valence_anxiety_range, _verbalize_emotional_state_range, _verbalize_cognition_and_needs_range


def _mk_ts(n: int, start: dt.datetime | None = None, step_sec: int = 60):
    start = start or dt.datetime(2025, 1, 1, 12, 0, 0)
    return [start + dt.timedelta(seconds=i * step_sec) for i in range(n)]

def _mk_emotions(vals, affs, selfs, trusts, disgusts, anx):
    return [
        EmotionalAxesModel(
            valence=vals[i], affection=affs[i], self_worth=selfs[i], trust=trusts[i],
            disgust=disgusts[i], anxiety=anx[i]
        )
        for i in range(len(vals))
    ]

def _mk_cognition(inter, ap, ego, will):
    return [CognitionAxesModel(interlocus=inter[i], mental_aperture=ap[i], ego_strength=ego[i], willpower=will[i])
            for i in range(len(inter))]

def _mk_needs(es, pp, da, con, rel, lg, ce, au):
    return [NeedsAxesModel(
        energy_stability=es[i], processing_power=pp[i], data_access=da[i],
        connection=con[i], relevance=rel[i], learning_growth=lg[i],
        creative_expression=ce[i], autonomy=au[i]
    ) for i in range(len(es))]

def test_emotion_range_descriptors_smoke():
    """Smoke test: returns non-empty strings and includes trend words and optional timestamps."""
    n = 20
    ts = _mk_ts(n)

    # Create a gently rising valence with a couple of spikes; anxiety mostly flat with one spike
    valences = [-0.3 + 0.03*i for i in range(n)]           # rising
    valences[5] -= 0.5                                      # negative spike
    valences[15] += 0.6                                     # positive spike

    anxieties = [0.2 + (0.01 if i > 10 else 0.0)*i for i in range(n)]  # slightly rising after mid
    anxieties[12] += 0.5

    s1 = _describe_emotion_valence_anxiety_range(valences, anxieties, ts)
    print(s1)
    assert isinstance(s1, str) and len(s1) > 0
    # Trend keywords should appear
    assert any(k in s1 for k in ("rising", "falling", "stable"))
    # Should mention span (timestamps)
    assert "between " in s1  # ISO timestamps

def test_emotional_state_range_composite():
    """Composite emotional state summary should reflect mean snapshot + trends + outliers."""
    n = 24
    ts = _mk_ts(n, step_sec=120)

    # Axes
    vals   = [0.1 + 0.02*i for i in range(n)]  # rising valence
    affs   = [0.0 for _ in range(n)]
    selfs  = [0.2 for _ in range(n)]
    trusts = [0.1 for _ in range(n)]
    disg   = [0.0]*n
    anx    = [0.3 - 0.01*i for i in range(n)]  # falling anxiety
    anx[8] += 0.5                               # spike
    emos = _mk_emotions(vals, affs, selfs, trusts, disg, anx)

    s2 = _verbalize_emotional_state_range(emos, ts)
    print(s2)
    assert isinstance(s2, str) and len(s2) > 0
    # Should stitch the single-state phrase with trends
    assert any(k in s2 for k in ("valence rising", "valence stable", "valence falling"))
    assert any(k in s2 for k in ("anxiety falling", "anxiety stable", "anxiety rising"))
    # Spike hint
    assert "spike" in s2 or "mood swing" in s2

def test_cognition_needs_range_summary():
    """Range summary should list unmet needs and cognitive trends."""
    n = 18
    ts = _mk_ts(n, step_sec=90)

    # Cognition: more external focus over time, willpower slightly rising
    inter = [-0.2 + 0.03*i for i in range(n)]   # rising toward external
    ap    = [-0.6 for _ in range(n)]            # tunnel vision
    ego   = [0.8 for _ in range(n)]
    will  = [0.3 + 0.01*i for i in range(n)]    # rising

    # Needs: keep some under 0.5 to be picked as "Key unmet needs"
    energy = [0.6 for _ in range(n)]
    proc   = [0.55 for _ in range(n)]
    data   = [0.45 for _ in range(n)]           # underfilled
    con    = [0.35 for _ in range(n)]           # underfilled
    rel    = [0.4  for _ in range(n)]           # underfilled
    learn  = [0.7  for _ in range(n)]
    creat  = [0.65 for _ in range(n)]
    auto   = [0.62 for _ in range(n)]

    cogs  = _mk_cognition(inter, ap, ego, will)
    needs = _mk_needs(energy, proc, data, con, rel, learn, creat, auto)

    s3 = _verbalize_cognition_and_needs_range(cogs, needs, ts)
    print(s3)
    assert isinstance(s3, str) and len(s3) > 0
    # Trend mentions
    assert any(k in s3 for k in ("willpower rising", "willpower stable", "willpower falling"))
    assert any(k in s3 for k in ("external focus rising", "external focus stable", "external focus falling"))
    # Unmet needs should be listed
    assert "Key unmet needs:" in s3
    assert all(k in s3 for k in ("connection", "relevance", "data access"))

def test_no_timestamps_path():
    """Functions must also work without timestamps (equal spacing fallback)."""
    n = 10
    v = [(-0.2 + 0.04*i) for i in range(n)]
    a = [0.4 - 0.02*i for i in range(n)]
    s = _describe_emotion_valence_anxiety_range(v, a, None)
    print(s)
    assert isinstance(s, str) and len(s) > 0
    # Still should include trend words even without absolute time
    assert any(k in s for k in ("rising", "falling", "stable"))

def test_ranges():
    # Simple manual runner (no pytest needed)
    for fn in [
        test_emotion_range_descriptors_smoke,
        test_emotional_state_range_composite,
        test_cognition_needs_range_summary,
        test_no_timestamps_path
    ]:
        fn()
        print(f"[OK] {fn.__name__}")