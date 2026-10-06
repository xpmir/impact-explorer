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
    assert config.user_template.startswith("[QUERY]")
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
    preset = RewriterConfig.preset("Arthur-75/storm-qwen3-8B")
    assert group_beam_search_problem(preset.generation)
    ok, message = check(preset)
    assert not ok and GROUP_BEAM_SEARCH in message
    allowed = {
        **preset.generation,
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
    assert rewriters.rewrite(config, "fox") is first
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
    assert loaded.generation["num_return_sequences"] == 3
    workspace.remove_rewriter("Arthur-75/storm-qwen3-8B")
    assert Workspace(tmp_path).rewriters == {}
    assert changed == ["Arthur-75/storm-qwen3-8B"] * 2


def test_storm_presets():
    from impact_explorer.rewriters import PRESETS, find_preset

    assert len(PRESETS) == 4
    for size, tokens in [("0.6B", 32), ("1.7B", 32), ("4B", 64), ("8B", 64)]:
        preset = RewriterConfig.preset(f"Arthur-75/storm-qwen3-{size}")
        assert preset.generation["max_new_tokens"] == tokens
        assert preset.user_template == "[QUERY]: {query}\n[KEYWORDS]: "
    assert find_preset("https://huggingface.co/arthur-75/STORM-qwen3-0.6B")
    assert find_preset("Qwen/Qwen3-0.6B") is None


def test_preset_from_model_field():
    preset = RewriterConfig.preset("Storm (Qwen3-0.6B)", "Arthur-75/storm-qwen3-0.6B")
    assert preset.name == "Storm (Qwen3-0.6B)"
    assert preset.generation["max_new_tokens"] == 32
    assert RewriterConfig.preset("Storm", None).generation == {}


def test_allow_group_beam_search():
    from impact_explorer.rewriters import (
        allow_group_beam_search,
        group_beam_search_allowed,
    )

    preset = RewriterConfig.preset("Arthur-75/storm-qwen3-0.6B").generation
    allowed = allow_group_beam_search(preset, True)
    assert group_beam_search_allowed(allowed)
    assert group_beam_search_problem(allowed) is None
    assert allow_group_beam_search(allowed, False) == preset
