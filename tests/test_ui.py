import asyncio
import json

from nicegui.testing import User

from impact_explorer.config import Workspace
from impact_explorer.store import QueryStore, SavedQuery
from impact_explorer.topics import TopicSets
from impact_explorer.ui import Services, create_app

from .conftest import TOPICS


async def topics_loaded(user: User):
    table = user.find(marker="topics").elements.pop()
    for _ in range(100):
        if table.rows:
            return table
        await asyncio.sleep(0.01)
    raise AssertionError("Topics were not loaded")


def services_for(workspace: Workspace) -> Services:
    return Services(workspace, topic_sets=TopicSets(loader=lambda _: TOPICS))


async def test_empty_workspace(user: User, tmp_path):
    create_app(services_for(Workspace(tmp_path)))
    await user.open("/")
    await user.should_see("No collection in this workspace yet")
    await user.should_see(marker="add-collection")


async def test_free_text_query(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="query").type("quick fox").trigger("keydown.enter")
    await user.should_see(marker="doc-d0")
    await user.should_see(marker="doc-d2")
    await user.should_not_see(marker="doc-d1")
    await user.should_see("Free-text query (no assessments)")


async def test_topic_with_assessments(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    await topics_loaded(user)
    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q1"}, 0])
    await user.should_see("Topic test.topics#q1 · 3 relevant")
    await user.should_see(marker="doc-d2")
    await user.should_see("P@10 0.200")
    # d9 is relevant but neither retrieved nor in the document store
    await user.should_see("Not retrieved (1)")
    await user.should_see(marker="doc-d9")
    await user.should_see("Document not found in the document store")

    # Relevance filter
    user.find(marker="filter").elements.pop().set_value("non-relevant")
    await user.should_not_see(marker="doc-d0")


async def test_save_query(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    await topics_loaded(user)
    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q2"}, 0])
    await user.should_see(marker="doc-d1")
    user.find(marker="save").click()
    user.find(marker="save-name").type("Cats")
    user.find(marker="save-confirm").click()
    await user.should_see("Query saved")

    saved = QueryStore(workspace.saved_queries).list("test")
    assert len(saved) == 1
    assert (saved[0].text, saved[0].name) == ("lazy cats", "Cats")
    assert (saved[0].source.dataset, saved[0].source.topic_id) == ("test.topics", "q2")


async def test_open_saved_query(user: User, workspace):
    store = QueryStore(workspace.saved_queries)
    query = store.save(SavedQuery(collection="test", text="lazy dog"))
    create_app(services_for(workspace))
    await user.open(f"/?saved={query.id}")
    await user.should_see(marker="doc-d0")
    await user.should_see(marker="doc-d1")


async def test_add_dataset_from_topic_panel(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    select = user.find(marker="dataset").elements.pop()
    with user:
        select.set_options([*select.options, "ds.new"], value="ds.new")
    await user.should_see("Added ds.new to test")
    on_disk = json.loads(workspace.settings_path.read_text())
    assert on_disk["collections"][0]["datasets"] == ["test.topics", "ds.new"]


async def test_settings_create_collection(user: User, tmp_path, collection_folder):
    workspace = Workspace(tmp_path / "ws")
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="add-collection").click()
    user.find(marker="settings-name").type("fresh")
    user.find(marker="settings-index").type(str(collection_folder / "index"))
    chips = user.find(marker="settings-datasets").elements.pop()
    with user:
        chips.set_value(["test.topics"])
    user.find(marker="settings-save").click()
    await user.should_see(marker="collection")

    collection = Workspace(tmp_path / "ws").collection("fresh")
    assert collection.datasets == ["test.topics"]
    assert collection.docstore_path == collection_folder / "docstore"


async def test_query_feedback(user: User, workspace):
    services = services_for(workspace)
    create_app(services)
    await user.open("/")
    # Feedback needs the engine, which is opened in the background
    for _ in range(100):
        if services.engines.loaded("test"):
            break
        await asyncio.sleep(0.01)
    query = user.find(marker="query")

    query.type("#band(fox zzz")
    await user.should_see("Syntax error: expected ')', found end of input")

    query.clear().type("#band(fox zzzunknown) the")
    await user.should_see("#band is dropped: a child can never match: zzzunknown")
    html = user.find(marker="feedback").elements.pop().content
    assert 'class="unknown"' in html and 'class="stop"' in html

    query.clear().type("#syn(fox cat")
    query.trigger("keydown.enter")
    await user.should_see("Invalid query")


async def test_query_help(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="help").click()
    await user.should_see("Query syntax")
    user.find("#1(new york) hotel").click()
    assert user.find(marker="query").elements.pop().value == "#1(new york) hotel"


async def test_settings_checks_on_open(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    # Checks run as soon as the dialog opens
    await user.should_see("5 documents · positions", marker="check-index")
    await user.should_see("ids from the 'id' key", marker="check-documents")
    await user.should_see("Unknown dataset", marker="check-dataset-test.topics")

    # Changing a field re-runs its check
    index = user.find(marker="settings-index")
    index.clear().type("/no/such/folder").trigger("blur")
    await user.should_see("No such folder", marker="check-index")

    # Saving is still possible
    user.find(marker="settings-save").click()
    await user.should_see("Saved, but some settings have problems")
    assert workspace.collection("test").index == "/no/such/folder"


async def test_evaluate_all_topics(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    table = await topics_loaded(user)
    user.find(marker="evaluate").click()
    for _ in range(300):
        if (
            "Mean over 2 topics"
            in user.find(marker="evaluate-summary").elements.pop().text
        ):
            break
        await asyncio.sleep(0.01)
    await user.should_see("Mean over 2 topics", marker="evaluate-summary")
    assert "nDCG@10" in [c["name"] for c in table.columns]
    assert all(row["nDCG@10"] is not None for row in table.rows)
    assert table.pagination["sortBy"] == "nDCG@10"


async def test_topic_fields_and_original_query(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    table = await topics_loaded(user)

    # Several fields: the table shows the selected one
    field = user.find(marker="query-field").elements.pop()
    assert field.visible and field.options == ["text", "description"]
    with user:
        field.set_value("description")
    for _ in range(100):
        if table.rows[0]["text"] == "hunts at night":
            break
        await asyncio.sleep(0.01)
    assert table.rows[0]["text"] == "hunts at night"

    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q1"}, 0])
    await user.should_see(marker="topic-panel")
    query = user.find(marker="query").elements.pop()
    for _ in range(100):
        if query.value == "hunts at night":
            break
        await asyncio.sleep(0.01)
    assert query.value == "hunts at night"
    await user.should_not_see(marker="vs-original")

    # Editing the query compares it with the original (description) query
    user.find(marker="query").clear().type("quick fox").trigger("keydown.enter")
    await user.should_see("vs. original description", marker="vs-original")
    await user.should_see("+0.333")  # R@10: 1/3 -> 2/3

    # Another field of the topic, from the topic panel
    user.find(marker="use-field-text").click()
    await user.should_not_see(marker="vs-original")


async def test_original_ranks(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    await topics_loaded(user)
    with user:
        user.find(marker="query-field").elements.pop().set_value("description")
    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q1"}, 0])
    await user.should_see(marker="doc-d2")
    await user.should_not_see(marker="orig-d2")

    # "hunts at night" only retrieves d2: d0 is new, d2 keeps a rank
    user.find(marker="query").clear().type("quick fox").trigger("keydown.enter")
    await user.should_see(marker="vs-original")
    await user.should_see("new", marker="orig-d0")
    await user.should_see("was #1", marker="orig-d2")


async def test_evaluate_all_with_saved_queries(user: User, workspace):
    from impact_explorer.store import QuerySource

    store = QueryStore(workspace.saved_queries)
    saved = store.save(
        SavedQuery(
            collection="test",
            text="hunts at night",
            source=QuerySource("test.topics", "q1"),
        )
    )
    services = services_for(workspace)
    services.store = store
    create_app(services)
    await user.open("/")
    table = await topics_loaded(user)
    user.find(marker="evaluate").click()
    for _ in range(300):
        if "saved queries" in user.find(marker="evaluate-summary").elements.pop().text:
            break
        await asyncio.sleep(0.01)
    columns = [c["name"] for c in table.columns]
    assert "saved" in columns and "delta" in columns
    row = next(r for r in table.rows if r["id"] == "q1")
    assert row["delta"] is not None and row["delta"] < 0
    await user.should_see(marker=f"saved-metrics-{saved.id}")
    assert list((workspace.folder / "evaluations").iterdir())


async def test_stop_words_switch(user: User, workspace, unfiltered):
    workspace.put(unfiltered)
    create_app(services_for(workspace))
    await user.open("/?collection=unfiltered")
    query = user.find(marker="query")
    query.type("#1(foxes are quick)").trigger("keydown.enter")
    await user.should_see("No result")
    switch = user.find(marker="stop-words").elements.pop()
    with user:
        switch.set_value(False)
    await user.should_see(marker="doc-d2")
    await user.should_see("is a stop word of this index")


async def test_topic_versions(user: User, workspace):
    from impact_explorer.store import QuerySource

    store = QueryStore(workspace.saved_queries)
    version = store.save(
        SavedQuery(
            collection="test",
            text="hunts at night",
            name="night",
            source=QuerySource("test.topics", "q1"),
        )
    )
    services = services_for(workspace)
    services.store = store
    create_app(services)
    await user.open("/")
    table = await topics_loaded(user)
    assert "versions" in [c["name"] for c in table.columns]
    assert next(r for r in table.rows if r["id"] == "q1")["versions"] == 1

    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q1"}, 0])
    await user.should_see(marker=f"version-{version.id}")
    user.find(marker=f"version-{version.id}").click()
    await user.should_see(marker="vs-original")
    assert user.find(marker="query").elements.pop().value == "hunts at night"


async def until(condition, timeout=3.0):
    for _ in range(int(timeout / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    assert condition()


async def test_rewrite_query(user: User, workspace):
    from impact_explorer.rewriters import RewriterConfig, Rewriters

    from .test_rewriters import FakeBackend

    workspace.put_rewriter(RewriterConfig(name="fake/rewriter"))
    services = Services(
        workspace,
        topic_sets=TopicSets(loader=lambda _: TOPICS),
        rewriters=Rewriters(factory=FakeBackend),
    )
    create_app(services)
    await user.open("/")
    await topics_loaded(user)
    user.find(marker="topics").trigger("rowClick", [{}, {"id": "q1"}, 0])
    await user.should_see(marker="doc-d0")

    await user.should_see(marker="query-time")
    user.find(marker="rewrite-fake/rewriter").click()
    await user.should_see(marker="rewrite-panel")
    await user.should_see(marker="rewrite-time")
    query = user.find(marker="query").elements.pop()
    assert query.value == "quick fox fox Vixen den burrow"
    # Compared with the topic's original query
    await user.should_see(marker="vs-original")

    # Combined differently without generating again
    calls = FakeBackend.calls
    user.find(marker="rewrite-combine-weight").elements.pop().value = 2
    await until(lambda: query.value == "quick fox quick fox fox Vixen den burrow")
    user.find(marker="rewrite-combine-mode").elements.pop().value = "Generated only"
    await until(lambda: query.value == "fox, Vixen, den. vixen, burrow")
    assert FakeBackend.calls == calls

    user.find(marker="rewrite-restore").click()
    await user.should_not_see(marker="rewrite-panel")
    assert query.value == "quick fox"


async def test_register_rewriter(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="settings-new-rewriter").click()
    user.find(marker="rewriter-name").type("Arthur-75/storm-qwen3-8B").trigger("blur")
    user.find(marker="rewriter-save").click()
    await user.should_see(marker="rewrite")
    assert workspace.rewriters["Arthur-75/storm-qwen3-8B"].combine == "{outputs}"


async def test_preset_for_other_storm_sizes(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="settings-new-rewriter").click()
    user.find(marker="rewriter-name").type(
        "https://huggingface.co/Arthur-75/storm-qwen3-0.6B"
    ).trigger("blur")
    user.find(marker="rewriter-save").click()
    await user.should_see(marker="rewrite")
    rewriter = workspace.rewriters["https://huggingface.co/Arthur-75/storm-qwen3-0.6B"]
    assert rewriter.combine == "{outputs}"


async def test_rewriter_model_settings(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="settings-new-rewriter").click()
    user.find(marker="rewriter-name").type("Arthur-75/storm-qwen3-0.6B").trigger("blur")
    user.find(marker="rewriter-model-settings").click()
    user.find(marker="rewriter-save").click()
    await user.should_see(marker="rewrite")
    rewriter = workspace.rewriters["Arthur-75/storm-qwen3-0.6B"]
    assert (rewriter.system_prompt, rewriter.user_template, rewriter.generation) == (
        "",
        "{query}",
        {},
    )


async def test_preset_uses_the_model_field(user: User, workspace):
    from impact_explorer.rewriters import RewriterConfig

    workspace.put_rewriter(
        RewriterConfig(name="Storm (Qwen3-0.6B)", model="Arthur-75/storm-qwen3-0.6B")
    )
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="settings-rewriter-Storm (Qwen3-0.6B)").click()
    user.find(marker="rewriter-preset").click()
    await user.should_see("Preset of Arthur-75/storm-qwen3-0.6B applied")
    user.find(marker="rewriter-save").click()
    await user.should_see(marker="rewrite")
    saved = workspace.rewriters["Storm (Qwen3-0.6B)"]
    assert (saved.user_template, saved.combine) == ("{query}", "{outputs}")


async def test_allow_group_beam_search_checkbox(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="settings-new-rewriter").click()
    user.find(marker="rewriter-name").type("Arthur-75/storm-qwen3-0.6B").trigger("blur")
    checkbox = user.find(marker="rewriter-allow-remote").elements.pop()
    assert checkbox.value is False
    with user:
        checkbox.set_value(True)
    user.find(marker="rewriter-save").click()
    await user.should_see(marker="rewrite")
    generation = workspace.rewriters["Arthur-75/storm-qwen3-0.6B"].generation
    assert generation["trust_remote_code"] is True
    assert generation["num_beam_groups"] == 3
    assert generation["diversity_penalty"] == 1.0


async def test_index_info(user: User, workspace):
    create_app(services_for(workspace))
    await user.open("/")
    user.find(marker="index-info").click()
    await user.should_see(marker="index-details")
    rows = user.find(marker="index-details").elements.pop().rows
    values = {row["name"]: row["value"] for row in rows}
    assert values["Pipeline"] == "pyserini"
    assert values["Codecs"] == "none (raw postings)"
