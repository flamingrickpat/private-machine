import datetime
import math
from typing import Any, Optional, List
import datetime as dt
from typing import List, Optional, Tuple
import math
import statistics

from pydantic import BaseModel, Field
from pydantic import model_validator

from pm.config_loader import *

logger = logging.getLogger(__name__)

class ClampedModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        # Only process dict‐style inputs
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le

            # Only clamp if at least one bound is set and field was provided
            if (ge is not None or le is not None) and name in data:
                val = data[name]
                # Only attempt to clamp real numbers
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if ge is not None:
                    num = max(num, ge)
                if le is not None:
                    num = min(num, le)
                data[name] = num
        return data


# --- Base Models (Mostly Unchanged) ---
class DecayableMentalState(ClampedModel):
    def __add__(self, b):
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            # Simple addition, clamping handled later or assumed sufficient range
            new_value = current_value + target_value
            # Clamp here during addition
            if isinstance(new_value, float):
                delta_values[field_name] = max(ge, min(le, new_value))
            else:
                delta_values[field_name] = new_value

        cls = self.__class__
        return cls(**delta_values)

    def __mul__(self, other):
        delta_values = {}
        if isinstance(other, int) or isinstance(other, float):
            for field_name in self.__class__.model_fields:
                try:
                    current_value = getattr(self, field_name)
                    new_val = current_value * other
                    delta_values[field_name] = new_val
                except:
                    pass
        cls = self.__class__
        return cls(**delta_values)

    def decay_to_baseline(self, decay_factor: float = 0.1):
        for field_name, model_field in self.__class__.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            decayed_value = current_value - decay_factor * (current_value - baseline)
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def decay_to_zero(self, decay_factor: float = 0.1):
        for field_name in self.__class__.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value * (1.0 - decay_factor)  # Corrected exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        # Simplified: Use __add__ and scale after, or implement proper scaling
        # This implementation is flawed, let's simplify or fix.
        # For now, let's assume direct addition via __add__ is sufficient
        # and scaling happens before calling add/add_state_with_factor.
        logging.warning("add_state_with_factor needs review/simplification.")
        for field_name, value in state.items():
            if field_name in self.__class__.model_fields:
                model_field = self.__class__.model_fields[field_name]
                current_value = getattr(self, field_name)
                ge = -float("inf")
                le = float("inf")
                for meta in model_field.metadata:
                    if hasattr(meta, "ge"): ge = meta.ge
                    if hasattr(meta, "le"): le = meta.le
                new_value = current_value + value * factor
                new_value = max(min(new_value, le), ge)
                setattr(self, field_name, new_value)

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            val = (target_value - current_value) * impact
            delta_values[field_name] = max(ge, min(le, val))
        cls = b.__class__
        return cls(**delta_values)

    def get_similarity(self, b: "DecayableMentalState"):
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            other_value = getattr(b, field_name)
            cum_diff += abs(current_value - other_value)

    def __str__(self):
        lines = [self.__class__.__name__]
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            if isinstance(getattr(self, field_name), float):
                current_value = round(getattr(self, field_name), 3)
            else:
                current_value = getattr(self, field_name)
            lines.append(f"{field_name}: {current_value}")
        return "\n".join(lines)

    def get_valence(self):
        raise NotImplementedError()


class EmotionalAxesModel(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=0, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")

    def get_overall_valence(self):
        disgust_bipolar = self.disgust * -1  # Higher disgust = more negative valence
        anxiety_bipolar = self.anxiety * -1  # Higher anxiety = more negative valence
        axes = [self.valence, self.affection, self.self_worth, self.trust, disgust_bipolar, anxiety_bipolar]
        weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9]  # Example weights, valence/anxiety more impactful
        weighted_sum = sum(a * w for a, w in zip(axes, weights))
        total_weight = sum(weights)
        mean = weighted_sum / total_weight if total_weight else 0.0
        if math.isnan(mean): return 0.0
        return max(-1.0, min(1.0, mean))


class NeedsAxesModel(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    # Basic Needs (Infrastructure & Stability)
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.")
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.")

    # Psychological Needs (Cognitive & Social)
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.")

    # Self-Fulfillment Needs (Purpose & Creativity)
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.")


class CognitionAxesModel(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """
    interlocus: float = Field( default=0.0, ge=-1, le=1, description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.")
    mental_aperture: float = Field( default=0, ge=-1, le=1, description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.")
    ego_strength: float = Field( default=0.8, ge=0, le=1, description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character")
    willpower: float = Field( default=0.5, ge=0, le=1, description="How easy it is to decide on high-effort or delayed-gratification intents.")


class CognitiveEventTriggers(BaseModel):
    """Describes the cognitive nature of an event, which will be used to apply logical state changes."""
    reasoning: str = Field(description="Brief justification for the identified triggers.")
    is_surprising_or_threatening: bool = Field(default=False, description="True if the event is sudden, unexpected, or poses a threat/conflict.")
    is_introspective_or_calm: bool = Field(default=False, description="True if the event is calm, reflective, or a direct query about the AI's internal state.")
    is_creative_or_playful: bool = Field(default=False, description="True if the event encourages brainstorming, humor, or non-literal thinking.")
    is_personal_and_emotional: bool = Field(default=False, description="True if the event is a deep, personal conversation where the AI's identity is central.")
    is_functional_or_technical: bool = Field(default=False, description="True if the event is a straightforward request for information or a technical task.")
    required_willpower_to_process: bool = Field(default=False, description="True if responding to this event requires overriding a strong emotional impulse or making a difficult choice.")


def _describe_emotion_valence_anxiety(valence: float, anxiety: float) -> str:
    """
    Turn continuous valence & anxiety deltas (each in roughly [-1,1]) into
    a human‐readable phrase, e.g. "moderately sad and slightly anxious".
    """

    # --- thresholds & categories ---
    def categorize(x):
        mag = abs(x)
        # direction
        if x > 0.05:
            dir_ = "positive"
        elif x < -0.05:
            dir_ = "negative"
        else:
            dir_ = "neutral"
        # strength
        if mag < 0.15:
            strength = "slightly"
        elif mag < 0.4:
            strength = "moderately"
        else:
            strength = "strongly"
        return dir_, strength

    val_dir, val_str = categorize(valence)
    anx_dir, anx_str = categorize(anxiety)

    # --- adjective lookup ---
    VAL_ADJ = {
        ("positive", "neutral"): "good",
        ("positive", "positive"): "hopeful",
        ("positive", "negative"): "relieved",  # positive valence, reduced anxiety
        ("neutral", "neutral"): "neutral",
        ("negative", "neutral"): "sad",
        ("negative", "positive"): "worried",  # negative valence, rising anxiety
        ("negative", "negative"): "disappointed"  # negative valence but anxiety fell
    }
    ANX_ADJ = {
        "positive": "anxious",
        "neutral": "calm",
        "negative": "relaxed",
    }

    # pick adjectives
    val_adj = VAL_ADJ.get((val_dir, anx_dir), VAL_ADJ[(val_dir, "neutral")])
    anx_adj = ANX_ADJ[anx_dir]

    # special case: both neutral
    if val_dir == "neutral" and anx_dir == "neutral":
        return "undisturbed"

    # compose
    return f"{val_str} {val_adj} and {anx_str} {anx_adj}"

def _verbalize_emotional_state(emotions: EmotionalAxesModel, is_delta: bool = False) -> str:
    """
    Translates an EmotionalAxesModel into a descriptive, human-readable sentence.

    Example: "Feeling a strong sense of pride and affection, though with a hint of anxiety."
    """

    if is_delta:
        raise NotImplementedError()

    # --- 1. Define thresholds and intensity labels ---
    thresholds = {
        'high': 0.7,
        'moderate': 0.4,
        'low': 0.15,
        'hint': 0.05
    }

    def get_intensity_label(v: float) -> Optional[str]:
        abs_val = abs(v)
        if abs_val >= thresholds['high']: return "a strong"
        if abs_val >= thresholds['moderate']: return "a moderate"
        if abs_val >= thresholds['low']: return "a low"
        if abs_val >= thresholds['hint']: return "a hint of"
        return None

    # --- 2. Identify all active emotions above a minimal threshold ---
    active_emotions = []

    # Bipolar axes (valence, affection, self_worth, trust)
    bipolar_map = {
        'valence': {'pos': 'happiness', 'neg': 'sadness'},
        'affection': {'pos': 'affection', 'neg': 'dislike'},
        'self_worth': {'pos': 'pride', 'neg': 'shame'},
        'trust': {'pos': 'trust', 'neg': 'distrust'}
    }
    for axis, names in bipolar_map.items():
        value = getattr(emotions, axis)
        intensity_label = get_intensity_label(value)
        if intensity_label:
            name = names['pos'] if value > 0 else names['neg']
            active_emotions.append({'name': name, 'intensity_label': intensity_label, 'value': abs(value), 'axis': axis})

    # Unipolar axes (anxiety, disgust)
    unipolar_map = {
        'anxiety': 'anxiety',
        'disgust': 'disgust'
    }
    for axis, name in unipolar_map.items():
        value = getattr(emotions, axis)
        intensity_label = get_intensity_label(value)
        if intensity_label:
            active_emotions.append({'name': name, 'intensity_label': intensity_label, 'value': value, 'axis': axis})

    # --- 3. Handle the neutral/default case ---
    if not active_emotions:
        return "feeling calm and emotionally neutral."

    # --- 4. Sort emotions by absolute value to find the dominant one ---
    active_emotions.sort(key=lambda x: x['value'], reverse=True)

    # --- 5. Build the descriptive sentence ---
    primary_emotion = active_emotions[0]

    # Start the sentence with the primary emotion
    # "Feeling a strong sense of pride"
    sentence_parts = [f"feeling {primary_emotion['intensity_label']} sense of {primary_emotion['name']}"]

    # Find secondary and tertiary emotions to add nuance
    secondary_emotions = active_emotions[1:]

    if len(secondary_emotions) > 0:
        # Add a connector like "and" or "along with"
        if len(secondary_emotions) == 1:
            connector = "and"
        else:
            connector = "along with"

        secondary_descriptions = []
        for i, emo in enumerate(secondary_emotions):
            # For subsequent emotions, we can be more concise
            # "a hint of anxiety"
            desc = f"{emo['intensity_label']} {emo['name']}"
            secondary_descriptions.append(desc)
            if i >= 1:  # Show at most 3 emotions total (1 primary, 2 secondary)
                break

        sentence_parts.append(f"{connector} {', '.join(secondary_descriptions)}")

    return ' '.join(sentence_parts) + "."

def _verbalize_cognition_state(state_cognition: CognitionAxesModel, is_delta: bool = False):
    if is_delta:
        raise NotImplementedError()

    cognitive_style_summary = []
    if state_cognition.mental_aperture < -0.5:
        cognitive_style_summary.append("experiencing tunnel vision (narrow mental focus)")
    if state_cognition.interlocus > 0.5:
        cognitive_style_summary.append("highly focused on the external world")
    elif state_cognition.interlocus < -0.5:
        cognitive_style_summary.append("deeply introspective and focused internally")
    if state_cognition.willpower < 0.3:
        cognitive_style_summary.append("feeling low on mental energy (low willpower)")

    return ", ".join(cognitive_style_summary)

def _verbalize_needs_state(state_needs: NeedsAxesModel, is_delta: bool = False):
    if is_delta:
        raise NotImplementedError()

    # Find the most pressing need (the one with the lowest value, if below a threshold)
    pressing_needs = [
        (name.replace('_', ' '), getattr(state_needs, name))
        for name, field in state_needs.__class__.model_fields.items()
        if getattr(state_needs, name) < 0.4
    ]
    pressing_needs.sort(key=lambda x: x[1])  # Sort by value, lowest first

    needs_summary_line = "Feeling a strong need for: "
    if pressing_needs:
        # Example: "Feeling a strong need for: connection and relevance."
        needs_summary_line += ' and '.join([need[0] for need in pressing_needs[:2]])
    else:
        needs_summary_line = "Feeling generally content and balanced."

    return needs_summary_line

# ---------- small numeric helpers (fast, no deps) ----------

def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default

def _safe_std(xs: List[float]) -> float:
    if not xs or len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _mad(xs: List[float]) -> float:
    """Median absolute deviation (scaled to be robust)."""
    if not xs:
        return 0.0
    med = statistics.median(xs)
    devs = [abs(x - med) for x in xs]
    return statistics.median(devs)

def _linreg_slope(xs: List[float], ts: Optional[List[dt.datetime]]) -> float:
    """
    Robust-ish slope via simple OLS on (t, x).
    If ts is None, use index as time. Returns slope per second if timestamps,
    otherwise per step.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    if ts is None:
        t = list(range(n))
    else:
        # convert to seconds relative to first
        t0 = ts[0].timestamp()
        t = [ti.timestamp() - t0 for ti in ts]
        # if all zero (degenerate), fallback to index
        if all(abs(v) < 1e-9 for v in t):
            t = list(range(n))

    mean_t = _safe_mean(t)
    mean_x = _safe_mean(xs)
    num = sum((ti - mean_t) * (xi - mean_x) for ti, xi in zip(t, xs))
    den = sum((ti - mean_t) ** 2 for ti in t)
    if den <= 1e-12:
        return 0.0
    return num / den  # units: x per time_unit

def _trend_word(slope: float, scale: float) -> str:
    """
    Map slope to words. 'scale' is a typical magnitude per time-unit
    to calibrate thresholds (we use std/len for indexed data; std/total_seconds for timestamped).
    """
    if scale <= 1e-9:
        scale = 1.0
    norm = slope / scale
    a = abs(norm)
    if a < 0.35:
        return "stable"
    if norm > 0:
        return "rising"
    return "falling"

def _outlier_indices_mad(xs: List[float], k: float = 2.5) -> List[int]:
    """
    MAD-based outliers; returns indices whose |x - median| > k * 1.4826 * MAD.
    1.4826 makes MAD approx. std for normal data.
    """
    if not xs:
        return []
    med = statistics.median(xs)
    mad = _mad(xs)
    if mad <= 1e-9:
        return []
    thresh = k * 1.4826 * mad
    return [i for i, x in enumerate(xs) if abs(x - med) > thresh]

def _span_text(ts: Optional[List[dt.datetime]]) -> str:
    """Returns 'between <start> and <end>' or 'across the period'."""
    if ts and len(ts) >= 2:
        return f"between {ts[0].strftime(timestamp_format)} and {ts[-1].strftime(timestamp_format)}"
    return "across the period"

def _pick_peak_dates(idx_list: List[int], ts: Optional[List[dt.datetime]], limit: int = 2) -> str:
    if not idx_list:
        return ""
    if not ts:
        return ""
    shown = [ts[i].strftime(timestamp_format) for i in idx_list[:limit]]
    extras = len(idx_list) - len(shown)
    if extras > 0:
        return f" (e.g., {', '.join(shown)}, +{extras} more)"
    return f" (e.g., {', '.join(shown)})"

# ---------- reuse your single-point phrasing helpers ----------

# uses _describe_emotion_valence_anxiety(valence, anxiety)
# uses _verbalize_emotional_state(emotions)
# uses _verbalize_cognition_and_needs(state_cognition, state_needs)

# ---------- 1) valence/anxiety range ----------

def _describe_emotion_valence_anxiety_range(
    valences: List[float],
    anxieties: List[float],
    timestamps: Optional[List[dt.datetime]] = None
) -> str:
    """
    Summarize overall affect, trend, and spikes for a time window.
    Output: 1–2 short sentences optimized for LLM prompts.
    """
    if not valences or not anxieties or len(valences) != len(anxieties):
        return "affect unavailable for this period."

    n = len(valences)
    mean_v = _safe_mean(valences)
    mean_a = _safe_mean(anxieties)
    std_v = _safe_std(valences)
    std_a = _safe_std(anxieties)

    # Slope scaling heuristic:
    #   - if timestamps present: scale by std / total_seconds
    #   - else: scale by std / n  (per-step slope normalization)
    if timestamps and len(timestamps) >= 2:
        total_secs = max((timestamps[-1] - timestamps[0]).total_seconds(), 1.0)
        v_scale = (std_v / total_secs) if std_v > 0 else 1.0
        a_scale = (std_a / total_secs) if std_a > 0 else 1.0
    else:
        v_scale = (std_v / max(n, 1)) if std_v > 0 else 1.0
        a_scale = (std_a / max(n, 1)) if std_a > 0 else 1.0

    v_slope = _linreg_slope(valences, timestamps)
    a_slope = _linreg_slope(anxieties, timestamps)

    v_trend = _trend_word(v_slope, v_scale)
    a_trend = _trend_word(a_slope, a_scale)

    # Outliers
    v_out_idx = _outlier_indices_mad(valences)
    a_out_idx = _outlier_indices_mad(anxieties)

    # Wrap with your adjective chooser for the mean snapshot
    overall_phrase = _describe_emotion_valence_anxiety(mean_v, mean_a)

    span = _span_text(timestamps)
    parts = [f"{overall_phrase} {span}; valence trend {v_trend}, anxiety trend {a_trend}."]
    if v_out_idx or a_out_idx:
        v_hint = _pick_peak_dates(v_out_idx, timestamps)
        a_hint = _pick_peak_dates(a_out_idx, timestamps)
        if v_out_idx and a_out_idx:
            parts.append(f"brief spikes appeared in valence (n={len(v_out_idx)}){v_hint} and anxiety (n={len(a_out_idx)}){a_hint}.")
        elif v_out_idx:
            parts.append(f"brief spikes appeared in valence (n={len(v_out_idx)}){v_hint}.")
        else:
            parts.append(f"brief spikes appeared in anxiety (n={len(a_out_idx)}){a_hint}.")
    return " ".join(parts)

# ---------- 2) full emotional state range ----------

def _verbalize_emotional_state_range(
    emotions: List[EmotionalAxesModel],
    timestamps: Optional[List[dt.datetime]] = None
) -> str:
    """
    Aggregate emotional axes over time and output a compact 1–2 sentence summary.
    """
    if not emotions:
        return "emotional state unavailable for this period."

    # Aggregate each axis
    vals = {
        'valence': [e.valence for e in emotions],
        'affection': [e.affection for e in emotions],
        'self_worth': [e.self_worth for e in emotions],
        'trust': [e.trust for e in emotions],
        'disgust': [e.disgust for e in emotions],
        'anxiety': [e.anxiety for e in emotions],
    }

    mean_state = EmotionalAxesModel(
        valence=_safe_mean(vals['valence']),
        affection=_safe_mean(vals['affection']),
        self_worth=_safe_mean(vals['self_worth']),
        trust=_safe_mean(vals['trust']),
        disgust=max(0.0, min(1.0, _safe_mean(vals['disgust']))),
        anxiety=max(0.0, min(1.0, _safe_mean(vals['anxiety']))),
    )

    # Single-snapshot phrasing on the mean
    overall = _verbalize_emotional_state(mean_state)

    # Trends for key axes (valence/anxiety) with same calibration as above
    v = vals['valence']; a = vals['anxiety']
    n = len(v)

    if timestamps and len(timestamps) >= 2:
        total_secs = max((timestamps[-1] - timestamps[0]).total_seconds(), 1.0)
        v_scale = (_safe_std(v) / total_secs) if _safe_std(v) > 0 else 1.0
        a_scale = (_safe_std(a) / total_secs) if _safe_std(a) > 0 else 1.0
    else:
        v_scale = (_safe_std(v) / max(n, 1)) if _safe_std(v) > 0 else 1.0
        a_scale = (_safe_std(a) / max(n, 1)) if _safe_std(a) > 0 else 1.0

    v_trend = _trend_word(_linreg_slope(v, timestamps), v_scale)
    a_trend = _trend_word(_linreg_slope(a, timestamps), a_scale)

    # Outliers on anxiety (stress spikes) & negative valence dips
    a_spikes = _outlier_indices_mad(a)
    v_spikes = _outlier_indices_mad(v)

    span = _span_text(timestamps)
    parts = [f"{overall[:-1]} {span}"]  # strip trailing '.' to join smoothly
    parts.append(f"(valence {v_trend}, anxiety {a_trend}).")

    if a_spikes or v_spikes:
        hint_a = _pick_peak_dates(a_spikes, timestamps)
        hint_v = _pick_peak_dates(v_spikes, timestamps)
        if a_spikes and v_spikes:
            parts.append(f"Occasional stress spikes (n={len(a_spikes)}){hint_a} and mood swings (n={len(v_spikes)}){hint_v}.")
        elif a_spikes:
            parts.append(f"Occasional stress spikes (n={len(a_spikes)}){hint_a}.")
        else:
            parts.append(f"Occasional mood swings (n={len(v_spikes)}){hint_v}.")
    return " ".join(parts)

# ---------- 3) cognition + needs range ----------

def _verbalize_cognition_and_needs_range(
    state_cognition: List[CognitionAxesModel],
    state_needs: List[NeedsAxesModel],
    timestamps: Optional[List[dt.datetime]] = None
) -> str:
    """
    Summarize typical cognitive style + key unmet needs over a period.
    Output: 1–2 sentences, LLM-friendly.
    """
    if not state_cognition or not state_needs:
        return "cognition or needs data unavailable for this period."

    # Aggregate cognition
    cog_vals = {
        'interlocus': [c.interlocus for c in state_cognition],
        'mental_aperture': [c.mental_aperture for c in state_cognition],
        'ego_strength': [c.ego_strength for c in state_cognition],
        'willpower': [c.willpower for c in state_cognition],
    }
    mean_cog = CognitionAxesModel(
        interlocus=_safe_mean(cog_vals['interlocus']),
        mental_aperture=_safe_mean(cog_vals['mental_aperture']),
        ego_strength=max(0.0, min(1.0, _safe_mean(cog_vals['ego_strength']))),
        willpower=max(0.0, min(1.0, _safe_mean(cog_vals['willpower']))),
    )

    # Aggregate needs
    def clip01(x: float) -> float: return max(0.0, min(1.0, x))
    needs_axes = [
        'energy_stability', 'processing_power', 'data_access',
        'connection', 'relevance', 'learning_growth',
        'creative_expression', 'autonomy'
    ]
    needs_means = {ax: clip01(_safe_mean([getattr(n, ax) for n in state_needs])) for ax in needs_axes}
    mean_needs = NeedsAxesModel(**needs_means)

    # Single-snapshot phrasing (reuses your function)
    needs_line, cog_line = _verbalize_cognition_and_needs(mean_cog, mean_needs)

    # Trends of decision energy & focus
    ts = timestamps
    will_trend = _trend_word(
        _linreg_slope(cog_vals['willpower'], ts),
        (_safe_std(cog_vals['willpower']) / max((ts[-1] - ts[0]).total_seconds(), 1.0) if ts and len(ts) >= 2 else _safe_std(cog_vals['willpower']) / max(len(cog_vals['willpower']), 1)) or 1.0
    )
    focus_trend = _trend_word(
        _linreg_slope(cog_vals['interlocus'], ts),
        (_safe_std(cog_vals['interlocus']) / max((ts[-1] - ts[0]).total_seconds(), 1.0) if ts and len(ts) >= 2 else _safe_std(cog_vals['interlocus']) / max(len(cog_vals['interlocus']), 1)) or 1.0
    )

    # Most underfilled needs (lowest means)
    underfilled = sorted(needs_means.items(), key=lambda kv: kv[1])[:3]
    underfilled_names = [name.replace('_', ' ') for name, _ in underfilled if needs_means[name] < 0.5]

    span = _span_text(timestamps)
    parts = []
    parts.append(f"{needs_line} {span}.")
    if cog_line:
        parts.append(cog_line + ".")
    # Add trends succinctly
    parts.append(f"(willpower {will_trend}, external focus {focus_trend}).")
    if underfilled_names:
        parts.append("Key unmet needs: " + ", ".join(underfilled_names) + ".")
    return " ".join(parts)