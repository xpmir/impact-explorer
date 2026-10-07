import pytest

from impact_explorer.config import Workspace
from impact_explorer.rewriters import (
    COMBINATIONS,
    GROUP_BEAM_SEARCH,
    RewriterConfig,
    Rewriters,
    check,
    combination_name,
    extract_keywords,
    group_beam_search_problem,
    hf_repo_id,
    make_rewrite,
    recombine,
)


class FakeBackend:
    calls = 0

    def __init__(self, config):
        self.config = config

    def generate(self, query):
        FakeBackend.calls += 1
        return ["<think>hmm</think>fox, Vixen, den.", "vixen, burrow"]


def test_keywords_and_combination():
    config = RewriterConfig.preset("Arthur-75/storm-qwen3-8B")
    result = make_rewrite(config, "quick fox", FakeBackend(config).generate("x"))
    assert result.outputs[0] == "fox, Vixen, den."
    assert result.keywords == ["fox", "Vixen", "den", "burrow"]
    # STORM: the outputs alone, repetitions kept
    assert result.query == "fox, Vixen, den. vixen, burrow"
    assert extract_keywords(["a; b\nc", "A"]) == ["a", "b", "c"]
    # Outputs without separators give their words
    assert extract_keywords(["vitamin D symptoms", "D fatigue"]) == [
        "vitamin",
        "D",
        "symptoms",
        "fatigue",
    ]


def test_combinations():
    config = RewriterConfig(name="r")
    result = make_rewrite(config, "quick fox", ["fox fox den", "den"])
    assert result.query == "quick fox fox den"
    assert recombine(result, COMBINATIONS["Generated only"], 1).query == (
        "fox fox den den"
    )
    assert recombine(result, COMBINATIONS["Original + generated"], 2).query == (
        "quick fox quick fox fox fox den den"
    )
    assert combination_name("{query} {outputs}") == "Original + generated"
    assert combination_name("{outputs} {query}") is None
    with pytest.raises(ValueError):
        RewriterConfig(name="r", query_weight=0).validate()


def test_model_ids():
    assert hf_repo_id("https://huggingface.co/Arthur-75/storm-qwen3-8B") == (
        "Arthur-75/storm-qwen3-8B"
    )
    assert hf_repo_id("https://hf.co/org/model/tree/main?x=1") == "org/model"
    config = RewriterConfig(name="org/model", model="https://huggingface.co/a/b")
    assert config.model_id == "a/b"


def test_group_beam_search_must_be_allowed():
    generation = {"num_beams": 6, "num_beam_groups": 3, "diversity_penalty": 1.0}
    assert group_beam_search_problem(generation)
    ok, message = check(RewriterConfig(name="r", generation=generation))
    assert not ok and GROUP_BEAM_SEARCH in message
    allowed = {
        **generation,
        "custom_generate": GROUP_BEAM_SEARCH,
        "trust_remote_code": True,
    }
    assert group_beam_search_problem(allowed) is None


def test_validation():
    import pytest

    with pytest.raises(ValueError):
        RewriterConfig(name="x", combine="{nope}").validate()
    with pytest.raises(ValueError):
        RewriterConfig(name="x", backend="openai").validate()


def test_cache_and_reload():
    rewriters = Rewriters(factory=FakeBackend)
    config = RewriterConfig(name="fake")
    FakeBackend.calls = 0
    first = rewriters.rewrite(config, "fox")
    assert set(first.timings) == {"loading", "generation"}
    assert not first.cached
    again = rewriters.rewrite(config, "fox")
    assert again.cached and again.query == first.query
    assert again.timings == first.timings
    assert FakeBackend.calls == 1
    # A modified configuration reloads the model and drops its results
    changed = RewriterConfig(name="fake", combine="{keywords}")
    assert rewriters.rewrite(changed, "fox").query == "fox Vixen den burrow"
    assert FakeBackend.calls == 2


def test_workspace_rewriters(tmp_path):
    workspace = Workspace(tmp_path)
    changed = []
    workspace.on_rewriter_change(changed.append)
    workspace.put_rewriter(RewriterConfig.preset("Arthur-75/storm-qwen3-8B"))
    loaded = Workspace(tmp_path).rewriters["Arthur-75/storm-qwen3-8B"]
    assert loaded.combine == "{outputs}"
    workspace.remove_rewriter("Arthur-75/storm-qwen3-8B")
    assert Workspace(tmp_path).rewriters == {}
    assert changed == ["Arthur-75/storm-qwen3-8B"] * 2


def test_storm_presets():
    from impact_explorer.rewriters import PRESETS, find_preset

    assert len(PRESETS) == 4
    for size in ["0.6B", "1.7B", "4B", "8B"]:
        preset = RewriterConfig.preset(f"Arthur-75/storm-qwen3-{size}")
        # The model's chat template and generation_config.json do the rest
        assert (preset.system_prompt, preset.user_template) == ("", "{query}")
        assert (preset.generation, preset.combine) == ({}, "{outputs}")
    # Presets are copies
    RewriterConfig.preset("Arthur-75/storm-qwen3-8B").generation["x"] = 1
    assert RewriterConfig.preset("Arthur-75/storm-qwen3-8B").generation == {}
    assert find_preset("https://huggingface.co/arthur-75/STORM-qwen3-0.6B")
    assert find_preset("Qwen/Qwen3-0.6B") is None


def test_preset_from_model_field():
    preset = RewriterConfig.preset("Storm (Qwen3-0.6B)", "Arthur-75/storm-qwen3-0.6B")
    assert preset.name == "Storm (Qwen3-0.6B)"
    assert preset.combine == "{outputs}"
    assert RewriterConfig.preset("Storm", None).combine == "{query} {keywords}"


def test_group_beam_search():
    from impact_explorer.rewriters import (
        group_beam_search_enabled,
        set_group_beam_search,
    )

    enabled = set_group_beam_search({}, True)
    assert enabled == {
        "num_beam_groups": 3,
        "diversity_penalty": 1.0,
        "custom_generate": GROUP_BEAM_SEARCH,
        "trust_remote_code": True,
    }
    assert group_beam_search_enabled(enabled)
    assert group_beam_search_problem(enabled) is None
    assert set_group_beam_search(enabled, False) == {}
    # Explicit values are kept
    assert set_group_beam_search({"num_beam_groups": 2}, True)["num_beam_groups"] == 2


def test_openai_payload(monkeypatch):
    import httpx

    from impact_explorer.rewriters import OpenAIBackend

    payloads = []

    def post(url, json, timeout):
        payloads.append(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "a, b"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    config = RewriterConfig.preset("Arthur-75/storm-qwen3-8B")
    config.backend, config.url = "openai", "http://x/v1"
    assert OpenAIBackend(config).generate("fox") == ["a, b"]
    # Only the query, greedy; the rest is the server's (model's) defaults
    assert payloads[-1] == {
        "model": "Arthur-75/storm-qwen3-8B",
        "messages": [{"role": "user", "content": "fox"}],
        "temperature": 0.0,
    }
    config.generation = {"num_return_sequences": 3, "max_new_tokens": 32}
    OpenAIBackend(config).generate("fox")
    assert (payloads[-1]["n"], payloads[-1]["max_tokens"]) == (3, 32)
