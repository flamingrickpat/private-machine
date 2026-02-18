import sys
import types
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_unit_config() -> dict:
    cfg_path = _ROOT / "config.unit.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _install_py_linq_stub() -> None:
    if "py_linq" in sys.modules:
        return

    mod = types.ModuleType("py_linq")

    class Enumerable:
        def __init__(self, iterable=None):
            self._items = list(iterable or [])

        def to_list(self):
            return list(self._items)

        def where(self, predicate):
            return Enumerable([x for x in self._items if predicate(x)])

        def order_by(self, key_selector):
            return Enumerable(sorted(self._items, key=key_selector))

        def order_by_descending(self, key_selector):
            return Enumerable(sorted(self._items, key=key_selector, reverse=True))

        def take(self, count):
            return Enumerable(self._items[:count])

        def reverse(self):
            return Enumerable(list(reversed(self._items)))

        def count(self):
            return len(self._items)

        def any(self, predicate=None):
            if predicate is None:
                return bool(self._items)
            return any(predicate(x) for x in self._items)

        def last_or_default(self, predicate=None, default=None):
            if predicate is None:
                return self._items[-1] if self._items else default
            for item in reversed(self._items):
                if predicate(item):
                    return item
            return default

        def first_or_default(self, predicate=None, default=None):
            if predicate is None:
                return self._items[0] if self._items else default
            for item in self._items:
                if predicate(item):
                    return item
            return default

        def __iter__(self):
            return iter(self._items)

    mod.Enumerable = Enumerable
    sys.modules["py_linq"] = mod


def _install_fastmcp_stub() -> None:
    if "fastmcp" in sys.modules:
        return

    mod = types.ModuleType("fastmcp")
    utilities_mod = types.ModuleType("fastmcp.utilities")
    logging_mod = types.ModuleType("fastmcp.utilities.logging")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    mod.Client = Client
    utilities_mod.logging = logging_mod
    mod.utilities = utilities_mod
    sys.modules["fastmcp"] = mod
    sys.modules["fastmcp.utilities"] = utilities_mod
    sys.modules["fastmcp.utilities.logging"] = logging_mod


_install_py_linq_stub()
_install_fastmcp_stub()


def _install_config_loader_stub() -> None:
    if "pm.config_loader" in sys.modules:
        return

    mod = types.ModuleType("pm.config_loader")
    cfg = _load_unit_config()
    mod.companion_name = str(cfg.get("companion_name", "Companion"))
    mod.user_name = str(cfg.get("user_name", "User"))
    mod.shell_system_name = "SYSTEM"
    mod.timestamp_format = str(cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S"))
    mod.QUOTE_START = "\""
    mod.QUOTE_END = "\""
    mod.character_card_story = str(cfg.get("character_card_story", ""))
    mod.worker_context_limit_tokens = int(cfg.get("worker_context_limit_tokens", 0) or 0)
    mod.agent_context_weights = dict(
        cfg.get(
            "agent_context_weights",
            {
                "workspace": 0.25,
                "latest": 0.30,
                "timeline": 0.30,
                "static": 0.15,
            },
        )
        or {}
    )
    mod.agent_context_min_section_tokens = int(cfg.get("agent_context_min_section_tokens", 96) or 96)
    mod.agent_context_latest_messages = int(cfg.get("agent_context_latest_messages", 40) or 40)
    mod.agent_context_workspace_items = int(cfg.get("agent_context_workspace_items", 48) or 48)
    mod.agent_context_target_ratio = float(cfg.get("agent_context_target_ratio", 0.72) or 0.72)
    mod.agent_context_safety_margin_tokens = int(cfg.get("agent_context_safety_margin_tokens", 220) or 220)
    mod.supported_capabilities = list(cfg.get("supported_capabilities", []) or [])
    mod.unsupported_capabilities = list(cfg.get("unsupported_capabilities", []) or [])
    mod.capability_notes = list(cfg.get("capability_notes", []) or [])
    mod.available_tools = []
    models = dict(cfg.get("models", {}) or {})
    mapping = dict(cfg.get("model_mapping", {}) or {})
    default_key = mapping.get("Default", "")
    default_model = dict(models.get(default_key, {}) or {})
    default_ctx = int(default_model.get("context", 4096) or 4096)
    mod.model_map = {
        "Default": {
            "context": default_ctx,
            "path": str(default_model.get("path", "stub.gguf") or "stub.gguf"),
            "layers": int(default_model.get("layers", -1) or -1),
            "last_n_tokens_size": int(default_model.get("last_n_tokens_size", 64) or 64),
            "temperature": float(default_model.get("temperature", 0.7) or 0.7),
            "top_k": int(default_model.get("top_k", 40) or 40),
            "top_p": float(default_model.get("top_p", 0.9) or 0.9),
            "min_p": float(default_model.get("min_p", 0.05) or 0.05),
            "repeat_penalty": float(default_model.get("repeat_penalty", 1.0) or 1.0),
            "frequency_penalty": float(default_model.get("frequency_penalty", 0.0) or 0.0),
            "presence_penalty": float(default_model.get("presence_penalty", 0.0) or 0.0),
        }
    }
    sys.modules["pm.config_loader"] = mod


_install_config_loader_stub()


def _install_llm_proxy_stub() -> None:
    if "pm.llm.llm_proxy" in sys.modules:
        return

    mod = types.ModuleType("pm.llm.llm_proxy")

    class LlmManagerProxy:
        def completion_text(self, preset, inp, comp_settings=None, discard_thinks=True):
            return "A cat is a small domesticated animal."

    def start_llm_thread():
        return LlmManagerProxy()

    mod.LlmManagerProxy = LlmManagerProxy
    mod.start_llm_thread = start_llm_thread
    sys.modules["pm.llm.llm_proxy"] = mod


_install_llm_proxy_stub()


def _install_nltk_stub() -> None:
    if "nltk" in sys.modules:
        return

    nltk_mod = types.ModuleType("nltk")
    cluster_mod = types.ModuleType("nltk.cluster")

    def cosine_distance(a, b):
        return 0.0

    cluster_mod.cosine_distance = cosine_distance
    nltk_mod.cluster = cluster_mod
    sys.modules["nltk"] = nltk_mod
    sys.modules["nltk.cluster"] = cluster_mod


_install_nltk_stub()


def _install_json_repair_stub() -> None:
    if "json_repair" in sys.modules:
        return

    mod = types.ModuleType("json_repair")

    def repair_json(s, *args, **kwargs):
        return s

    mod.repair_json = repair_json
    sys.modules["json_repair"] = mod


_install_json_repair_stub()
