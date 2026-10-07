"""Query rewriters: language models that expand or reformulate queries.

A rewriter turns a query into a prompt (system prompt + user template),
generates one or more outputs, extracts keywords from them, and combines
them with the original query into the query that is searched.

Two backends:

- ``transformers``: a local Hugging Face causal LM (``uv sync --extra
  rewriters`` installs torch and transformers);
- ``openai``: an OpenAI-compatible chat completion server (e.g. ``vllm
  serve``), for models too large to run locally.
"""

import copy
import logging
import re
import threading
from dataclasses import asdict, dataclass, field, replace

logger = logging.getLogger(__name__)

BACKENDS = ("transformers", "openai")


COMBINATIONS: dict[str, str] = {
    "Generated only": "{outputs}",
    "Original + generated": "{query} {outputs}",
    "Original + unique keywords": "{query} {keywords}",
}
"""Ways to build the searched query (label -> ``combine`` template)"""


def combination_name(template: str) -> str | None:
    """The label of a predefined combination (None: custom template)"""
    for name, value in COMBINATIONS.items():
        if value == template:
            return name
    return None


STORM_PRESET: dict = {
    "system_prompt": "",
    "user_template": "{query}",
    "combine": "{outputs}",
    "generation": {},
}
"""STORM: lexical query expansion for BM25 (Qwen3 fine-tuned), see
https://github.com/arthur-75/storm. The model repositories hold the prompt
(the chat template adds the system prompt and wraps the query as
``[QUERY]: …\n[KEYWORDS]: ``) and the decoding (``generation_config.json``:
beam search, 3 outputs, 32 or 64 new tokens), so only the query is sent.
Group beam search (the paper's setting) is an opt-in, as it runs code from
the Hub (:func:`set_group_beam_search`).

The searched query is the concatenation of the raw outputs (repeated words
weigh more, as with Lucene): the model is trained with the generated text
alone as the BM25 query, and learns to repeat the query terms itself"""


PRESETS: dict[str, dict] = {
    f"Arthur-75/storm-qwen3-{size}": STORM_PRESET
    for size in ("0.6B", "1.7B", "4B", "8B")
}


def find_preset(name: str) -> dict | None:
    """The preset of a model (name or Hugging Face URL, any case)"""
    repo = hf_repo_id(name).lower()
    for model, preset in PRESETS.items():
        if model.lower() == repo:
            return preset
    return None


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
HF_URL_RE = re.compile(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/")

GROUP_BEAM_SEARCH = "transformers-community/group-beam-search"


def hf_repo_id(value: str) -> str:
    """``org/model`` from a Hugging Face model URL (or the value itself)"""
    value = value.strip()
    if HF_URL_RE.match(value):
        value = HF_URL_RE.sub("", value)
        # Drop /tree/main, /blob/..., query strings
        value = "/".join(value.split("?")[0].split("/")[:2])
    return value


GROUP_BEAM_SEARCH_PARAMETERS = {"num_beam_groups": 3, "diversity_penalty": 1.0}
"""STORM's best configuration (with the model's ``num_beams=6``)"""


def group_beam_search_problem(generation: dict) -> str | None:
    """Group beam search (``num_beam_groups``) is no longer part of
    transformers: it needs code from the Hub, which must be allowed
    explicitly"""
    if generation.get("num_beam_groups", 1) <= 1:
        return None
    if generation.get("custom_generate") and generation.get("trust_remote_code"):
        return None
    return (
        "Group beam search (num_beam_groups) now runs code from "
        f"https://hf.co/{GROUP_BEAM_SEARCH}: enable it in the rewriter's "
        'settings ("Group beam search"), or remove num_beam_groups '
        "and diversity_penalty there (plain beam search)"
    )


def set_group_beam_search(generation: dict, enable: bool) -> dict:
    """Generation parameters with group beam search (and the Hub code it
    needs) enabled or not"""
    generation = dict(generation)
    if enable:
        for key, value in GROUP_BEAM_SEARCH_PARAMETERS.items():
            generation.setdefault(key, value)
        generation["custom_generate"] = GROUP_BEAM_SEARCH
        generation["trust_remote_code"] = True
    else:
        for key in GROUP_BEAM_SEARCH_PARAMETERS:
            generation.pop(key, None)
        if generation.get("custom_generate") == GROUP_BEAM_SEARCH:
            generation.pop("custom_generate")
        generation.pop("trust_remote_code", None)
    return generation


def group_beam_search_enabled(generation: dict) -> bool:
    return (
        generation.get("num_beam_groups", 1) > 1
        and generation.get("custom_generate") == GROUP_BEAM_SEARCH
        and generation.get("trust_remote_code") is True
    )


class RewriterError(RuntimeError):
    pass


@dataclass
class RewriterConfig:
    name: str
    """Unique name (e.g. the model id)"""

    backend: str = "transformers"
    model: str | None = None
    """Model id (default: the name)"""

    url: str | None = None
    """Base URL of an OpenAI-compatible server (``openai`` backend)"""

    system_prompt: str = ""
    user_template: str = "{query}"
    """Prompt; ``{query}`` is replaced by the query"""

    combine: str = "{query} {keywords}"
    """Searched query; ``{query}``, ``{keywords}`` (unique keywords of all
    outputs, space separated) and ``{outputs}`` (raw outputs, space
    separated: repeated words weigh more)"""

    query_weight: int = 1
    """How many times ``{query}`` is repeated (its weight, since repeated
    words add up in BM25)"""

    generation: dict = field(default_factory=dict)
    """Generation parameters (transformers ``generate`` arguments; for the
    ``openai`` backend, ``max_new_tokens``, ``num_return_sequences`` and
    ``temperature`` are mapped to the API)"""

    device: str = "auto"
    """auto (cuda, then mps, then cpu), or a torch device"""

    dtype: str = "auto"
    """auto, float32, bfloat16 or float16"""

    @property
    def model_id(self) -> str:
        """Hugging Face repo id (URLs such as https://huggingface.co/org/model
        are accepted too)"""
        return hf_repo_id(self.model or self.name)

    def validate(self):
        if not self.name:
            raise ValueError("A rewriter name is required")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        if self.backend == "openai" and not self.url:
            raise ValueError("The openai backend needs a server URL")
        if not isinstance(self.query_weight, int) or self.query_weight < 1:
            raise ValueError("The original query weight must be an integer ≥ 1")
        for template, names in (
            (self.user_template, ("query",)),
            (self.combine, ("query", "keywords", "outputs")),
        ):
            try:
                template.format(**{n: "" for n in names})
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(f"Invalid template {template!r}: {e}") from None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "RewriterConfig":
        return RewriterConfig(**data)

    @staticmethod
    def preset(name: str, model: str | None = None) -> "RewriterConfig":
        """A configuration for a known model (or the generic defaults); the
        model is ``model`` if given, else the name"""
        config = RewriterConfig(name=name, model=model or None)
        preset = copy.deepcopy(find_preset(config.model_id) or {})
        return RewriterConfig(name=name, model=model or None, **preset)


@dataclass
class Rewrite:
    rewriter: str
    original: str
    query: str
    """The query to search"""

    outputs: list[str]
    keywords: list[str]
    combine: str = "{query} {keywords}"
    query_weight: int = 1
    timings: dict = field(default_factory=dict)
    """``loading`` (first use of the model) and ``generation``
    (:class:`~impact_explorer.engine.Timing`; CPU time of the whole
    process, as inference uses several threads)"""

    cached: bool = False
    """Taken from the cache (the timings are the original ones)"""


def clean_output(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def extract_keywords(outputs: list[str]) -> list[str]:
    """Unique keywords of comma (or line) separated outputs, in order; an
    output without separators (STORM often generates plain word sequences)
    gives its words"""
    seen: dict[str, str] = {}
    for output in outputs:
        separated = re.search(r"[,\n;]", output) is not None
        for keyword in re.split(r"[,\n;]" if separated else r"\s+", output):
            keyword = keyword.strip().strip(".").strip()
            if keyword and keyword.lower() not in seen:
                seen[keyword.lower()] = keyword
    return list(seen.values())


def make_rewrite(
    config: RewriterConfig,
    query: str,
    outputs: list[str],
    combine: str | None = None,
    query_weight: int | None = None,
) -> Rewrite:
    """The searched query (``combine`` and ``query_weight`` override the
    configuration's)"""
    combine = config.combine if combine is None else combine
    query_weight = config.query_weight if query_weight is None else query_weight
    outputs = [clean_output(o) for o in outputs]
    keywords = extract_keywords(outputs)
    rewritten = combine.format(
        query=" ".join([query] * max(1, query_weight)),
        keywords=" ".join(keywords),
        outputs=" ".join(outputs),
    )
    return Rewrite(
        rewriter=config.name,
        original=query,
        query=" ".join(rewritten.split()),
        outputs=outputs,
        keywords=keywords,
        combine=combine,
        query_weight=query_weight,
    )


def recombine(rewrite: Rewrite, combine: str, query_weight: int) -> Rewrite:
    """The same outputs, combined differently (no generation)"""
    config = RewriterConfig(name=rewrite.rewriter)
    result = make_rewrite(
        config, rewrite.original, rewrite.outputs, combine, query_weight
    )
    return replace(result, timings=rewrite.timings, cached=rewrite.cached)


def messages(config: RewriterConfig, query: str) -> list[dict]:
    result = []
    if config.system_prompt:
        result.append({"role": "system", "content": config.system_prompt})
    result.append({"role": "user", "content": config.user_template.format(query=query)})
    return result


class TransformersBackend:
    def __init__(self, config: RewriterConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RewriterError(
                "transformers and torch are needed for local rewriters: "
                "uv sync --extra rewriters"
            ) from None
        self.config = config
        self.torch = torch
        device = config.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = config.dtype
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "bfloat16"
        self.device = device
        logger.info("Loading %s on %s (%s)", config.model_id, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, dtype=getattr(torch, dtype)
        ).to(device)
        self.model.eval()

    def generate(self, query: str) -> list[str]:
        prompt = self.tokenizer.apply_chat_template(
            messages(self.config, query),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The chat template already adds the special tokens
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        # Unset parameters come from the model's generation_config.json
        generation = dict(self.config.generation)
        if problem := group_beam_search_problem(generation):
            raise RewriterError(problem)
        if self.model.generation_config.pad_token_id is None:
            generation.setdefault(
                "pad_token_id",
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generation)
        start = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(sequence[start:], skip_special_tokens=True)
            for sequence in output
        ]


class OpenAIBackend:
    def __init__(self, config: RewriterConfig):
        self.config = config

    def generate(self, query: str) -> list[str]:
        import httpx

        generation = self.config.generation
        payload = {
            "model": self.config.model_id,
            "messages": messages(self.config, query),
        }
        # Unset parameters: the server's defaults (vLLM takes max_new_tokens
        # from generation_config.json, but not do_sample=False, hence the
        # explicit greedy decoding)
        for key, api_key in (
            ("num_return_sequences", "n"),
            ("max_new_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            if key in generation:
                payload[api_key] = generation[key]
        if not generation.get("do_sample", False):
            payload.setdefault("temperature", 0.0)
        url = self.config.url.rstrip("/") + "/chat/completions"
        try:
            response = httpx.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RewriterError(f"{url}: {e}") from e
        return [choice["message"]["content"] for choice in response.json()["choices"]]


def check(config: RewriterConfig) -> tuple[bool, str]:
    """Whether a rewriter can be used, without loading or downloading it"""
    try:
        config.validate()
    except ValueError as e:
        return False, str(e)
    if config.backend == "openai":
        import httpx

        url = config.url.rstrip("/") + "/models"
        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()
            models = [m.get("id") for m in response.json().get("data", [])]
        except Exception as e:
            return False, f"Server not reachable ({url}): {e}"
        if config.model_id not in models:
            return (
                False,
                f"{config.model_id} is not served (served: {', '.join(models)})",
            )
        return True, f"Served by {config.url}"

    if problem := group_beam_search_problem(config.generation):
        return False, problem
    missing = [m for m in ("torch", "transformers") if not _importable(m)]
    if missing:
        return False, (f"{', '.join(missing)} not installed: uv sync --extra rewriters")
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(config.model_id, "config.json")
    except Exception:
        cached = None
    if isinstance(cached, str):
        return True, f"{config.model_id} is cached locally"
    return True, (
        f"{config.model_id} is not downloaded yet: it is downloaded when "
        "first used (can be large)"
    )


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


class Rewriters:
    """Loaded rewriters and their results, shared by all clients"""

    def __init__(self, factory=None):
        self.factory = factory or self._create
        self._backends: dict[str, object] = {}
        self._configs: dict[str, RewriterConfig] = {}
        self._cache: dict[tuple, Rewrite] = {}
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @staticmethod
    def _create(config: RewriterConfig):
        if config.backend == "openai":
            return OpenAIBackend(config)
        return TransformersBackend(config)

    def backend(self, config: RewriterConfig, timings: dict | None = None):
        from .engine import Timer

        with self._lock:
            if self._configs.get(config.name) != config:
                self._backends.pop(config.name, None)
                self._cache = {
                    k: v for k, v in self._cache.items() if k[0] != config.name
                }
            if config.name not in self._backends:
                with Timer(process=True) as loading:
                    self._backends[config.name] = self.factory(config)
                if timings is not None:
                    timings["loading"] = loading
                self._configs[config.name] = config
            return self._backends[config.name]

    def rewrite(self, config: RewriterConfig, query: str) -> Rewrite:
        """Rewrites a query (cached; meant to run in a thread)"""
        from .engine import Timer

        key = (config.name, query)
        timings = {}
        backend = self.backend(config, timings)
        if key in self._cache:
            return replace(self._cache[key], cached=True)
        # Models are not thread-safe: one generation at a time
        with self._generation_lock:
            with Timer(process=True) as timings["generation"]:
                outputs = backend.generate(query)
        result = make_rewrite(config, query, outputs)
        result.timings = timings
        self._cache[key] = result
        return result

    def invalidate(self, name: str):
        with self._lock:
            self._backends.pop(name, None)
            self._configs.pop(name, None)
