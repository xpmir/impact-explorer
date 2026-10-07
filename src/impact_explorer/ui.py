"""NiceGUI web interface."""

import html
import logging
from dataclasses import dataclass, replace

from nicegui import run, ui

from .batch import BatchRun, BatchRuns, RunKey, SavedResult, fingerprint
from .builds import Builds
from .config import BM25Params, ConfigError, Workspace
from .documents import Document
from .engine import Engines, SearchEngine, SearchResult
from .evaluation import Evaluation, Label, evaluate
from .highlight import (
    QUERY_CSS,
    error_html,
    query_html,
    segments,
    snippet,
    term_colors,
    term_counts,
    to_html,
)
from .index_info import Detail, index_details
from .query_syntax import HELP, QuerySyntaxError, is_structured
from .rewriters import Rewrite, Rewriters, recombine
from .settings_ui import CombinationControls, SettingsDialog
from .store import QuerySource, QueryStore, SavedQuery
from .topics import DEFAULT_FIELD, TopicSet, TopicSets

logger = logging.getLogger(__name__)

LABEL_STYLE = {
    Label.RELEVANT: ("positive", "check_circle"),
    Label.NON_RELEVANT: ("negative", "cancel"),
    Label.UNJUDGED: ("grey-6", "help_outline"),
}

CSS = (
    QUERY_CSS
    + """
mark.qt { padding: 0 2px; border-radius: 3px; color: inherit; }
.doc-text { line-height: 1.6; white-space: normal; }
.result-card { width: 100%; }
"""
)


TOPIC_COLUMNS = [
    {"name": "id", "label": "ID", "field": "id", "sortable": True, "align": "left"},
    {
        "name": "text",
        "label": "Topic",
        "field": "text",
        "align": "left",
        "classes": "whitespace-normal",
    },
    {"name": "rel", "label": "#rel", "field": "rel", "sortable": True},
]


@dataclass
class Baseline:
    """Results of the original query of a topic (to compare an edited one)"""

    metrics: dict[str, float]
    ranks: dict[str, int]
    """Rank of each document retrieved by the original query (within depth)"""

    depth: int


class Services:
    """State shared by all clients"""

    def __init__(
        self,
        workspace: Workspace,
        engines=None,
        topic_sets=None,
        store=None,
        rewriters=None,
    ):
        self.workspace = workspace
        self.engines = engines or Engines(workspace)
        self.topic_sets = topic_sets or TopicSets()
        self.store = store or QueryStore(workspace.saved_queries)
        self.batch_runs = BatchRuns(workspace.folder / "evaluations")
        workspace.on_change(self.batch_runs.invalidate)
        self.rewriters = rewriters or Rewriters()
        workspace.on_rewriter_change(self.rewriters.invalidate)
        self.builds = Builds(workspace)


def ms(seconds: float) -> str:
    milliseconds = seconds * 1000
    return f"{milliseconds:.1f} ms" if milliseconds < 100 else f"{milliseconds:,.0f} ms"


class ExplorerPage:
    """Per-client page state and widgets"""

    def __init__(self, services: Services, collection: str | None = None):
        self.services = services
        names = list(services.workspace.collections)
        if collection not in names:
            collection = names[0] if names else None
        self.collection = collection
        self.source: QuerySource | None = None
        self.topic_set: TopicSet | None = None
        self.result: SearchResult | None = None
        self.evaluation: Evaluation | None = None
        self.documents: dict[str, Document | None] = {}
        self.filter = "all"
        self.requested_dataset: str | None = None
        self.query_field = DEFAULT_FIELD
        self.rewrite_result: Rewrite | None = None
        self.keep_stop_words = False
        """Disables the index's stop list (query switch)"""
        """Topic field used as query (TREC topics: text, description, ...)"""
        self.baseline: Baseline | None = None
        """The original topic query's results, when the query was edited"""
        self._baselines: dict[tuple, Baseline] = {}
        self.top_k = 10
        self.depth = 1000
        self.bm25 = (
            replace(services.workspace.collection(collection).bm25)
            if collection
            else BM25Params()
        )

    @property
    def engine(self) -> SearchEngine:
        return self.services.engines.get(self.collection)

    @property
    def qrels(self) -> dict[str, float]:
        if self.source is None or self.topic_set is None:
            return {}
        return self.topic_set.qrels.get(self.source.topic_id, {})

    # --- Layout

    def build(self):
        ui.add_css(CSS)
        workspace = self.services.workspace
        with ui.header().classes("items-center gap-4 bg-slate-800"):
            ui.label("impact-explorer").classes("text-lg font-bold")
            if self.collection:
                ui.select(
                    list(workspace.collections),
                    value=self.collection,
                    label="Collection",
                    on_change=lambda e: self.set_collection(e.value),
                ).props("dark dense options-dense").classes("w-56").mark("collection")
                ui.button(icon="info", on_click=self.show_index_info).props(
                    "flat round dense color=white"
                ).tooltip("Index details: pipeline, storage").mark("index-info")
            ui.space()
            ui.button(icon="settings", on_click=self.open_settings).props(
                "flat round color=white"
            ).tooltip("Workspace: collections and datasets").mark("settings")

        if self.collection is None:
            with ui.column().classes("w-full items-center mt-24 gap-2"):
                ui.label("No collection in this workspace yet").classes("text-xl")
                ui.label(str(workspace.folder)).classes("text-sm text-grey-7 font-mono")
                ui.button(
                    "Add a collection", icon="add", on_click=self.open_settings
                ).mark("add-collection")
            return

        with ui.left_drawer(value=True).classes("bg-slate-50").props("width=440"):
            with ui.tabs().classes("w-full") as tabs:
                topics_tab = ui.tab("Topics", icon="list")
                saved_tab = ui.tab("Saved", icon="bookmark")
            with ui.tab_panels(tabs, value=topics_tab).classes("w-full bg-transparent"):
                with ui.tab_panel(topics_tab).classes("p-0 gap-2"):
                    self.build_topics_panel()
                with ui.tab_panel(saved_tab).classes("p-0"):
                    self.saved_panel = ui.column().classes("w-full gap-1")
        self.refresh_saved()

        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                self.query_input = (
                    ui.input("Query", placeholder="Type a query or pick a topic")
                    .props("outlined clearable")
                    .classes("grow")
                    .on("keydown.enter", self.search)
                    .on_value_change(self.update_feedback)
                    .mark("query")
                )
                ui.button("Search", icon="search", on_click=self.search).mark("search")
                ui.button("Save", icon="bookmark_add", on_click=self.save_dialog).props(
                    "outline"
                ).mark("save")
                self.build_rewrite_menu()
                ui.button(icon="help_outline", on_click=self.show_help).props(
                    "flat round"
                ).tooltip("Query syntax").mark("help")
            with ui.row().classes("w-full items-start gap-2 -mt-2"):
                self.feedback_kind = ui.badge().props("outline").classes("mt-1")
                with ui.column().classes("grow gap-0"):
                    self.feedback = (
                        ui.html(sanitize=False).classes("qs").mark("feedback")
                    )
                    self.feedback_message = (
                        ui.label().classes("text-xs").mark("feedback-message")
                    )
            self.rewrite_panel = ui.column().classes("w-full gap-1")
            with ui.row().classes("items-center gap-3"):
                ui.number("Top-K", min=1, max=1000, step=5, format="%d").bind_value(
                    self, "top_k", forward=lambda v: int(v or 10)
                ).props("dense outlined").classes("w-24").on(
                    "keydown.enter", self.search
                )
                ui.number("Depth", min=1, max=10000, step=100, format="%d").bind_value(
                    self, "depth", forward=lambda v: int(v or 1000)
                ).props("dense outlined").classes("w-24").tooltip(
                    "Search depth used to locate the rank of non-retrieved documents"
                )
                self.k1_input = (
                    ui.number("BM25 k1", step=0.1)
                    .bind_value(self.bm25, "k1")
                    .props("dense outlined")
                    .classes("w-24")
                )
                self.b_input = (
                    ui.number("BM25 b", step=0.05, min=0, max=1)
                    .bind_value(self.bm25, "b")
                    .props("dense outlined")
                    .classes("w-24")
                )
                ui.switch(
                    "Remove stop words",
                    value=True,
                    on_change=lambda e: self.set_stop_words(e.value),
                ).props("dense").tooltip(
                    "Off: stop words of the index are searched too (they only "
                    "match if the index was built without removing them)"
                ).mark("stop-words")
                self.source_label = ui.label().classes("text-sm text-grey-7")
                # Edited topic queries keep their assessments (to compare
                # reformulations); this detaches the query from the topic
                self.detach_button = (
                    ui.button(icon="link_off", on_click=self.detach)
                    .props("flat dense round size=sm")
                    .tooltip("Detach from topic (drop assessments)")
                )
            self.topic_panel = ui.column().classes("w-full gap-0")
            self.terms_row = ui.row().classes("gap-1 items-center")
            self.metrics_row = ui.row().classes("gap-2 items-center")
            with ui.tabs().classes("w-full").props("align=left") as self.result_tabs:
                self.top_tab = ui.tab("top", label="Top-K", icon="format_list_numbered")
                self.missed_tab = ui.tab(
                    "missed", label="Not retrieved", icon="search_off"
                )
            with ui.tab_panels(self.result_tabs, value=self.top_tab).classes("w-full"):
                with ui.tab_panel(self.top_tab).classes("p-0 gap-2"):
                    ui.toggle(
                        {
                            "all": "All",
                            Label.RELEVANT.value: "Relevant",
                            Label.NON_RELEVANT.value: "Non-relevant",
                            Label.UNJUDGED.value: "Unjudged",
                        },
                        value="all",
                        on_change=lambda e: self.set_filter(e.value),
                    ).props("dense no-caps").mark("filter")
                    self.top_list = ui.column().classes("w-full gap-2")
                with ui.tab_panel(self.missed_tab).classes("p-0"):
                    self.missed_list = ui.column().classes("w-full gap-2")
        self.update_source_label()
        self.update_feedback()
        ui.timer(0, self.preload_engine, once=True)

    # --- Query rewriting

    def build_rewrite_menu(self):
        rewriters = list(self.services.workspace.rewriters)
        with (
            ui.dropdown_button("Rewrite", icon="auto_fix_high", auto_close=True)
            .props("outline no-caps")
            .mark("rewrite")
        ):
            if not rewriters:
                ui.item("No rewriter: register one in the settings (⚙)").props(
                    "disable"
                )
            for name in rewriters:
                ui.item(name, on_click=lambda n=name: self.rewrite(n)).mark(
                    f"rewrite-{name}"
                )

    async def rewrite(self, name: str):
        text = (self.query_input.value or "").strip()
        if not text:
            ui.notify("Nothing to rewrite: the query is empty", type="warning")
            return
        config = self.services.workspace.rewriters.get(name)
        if config is None:
            return
        notification = ui.notification(
            f"Rewriting with {name}… (the first time, the model is loaded)",
            spinner=True,
            timeout=None,
        )
        try:
            result = await run.io_bound(self.services.rewriters.rewrite, config, text)
        except Exception as e:
            logger.warning("Rewriting with %s failed: %s", name, e)
            ui.notify(f"Rewriting failed: {e}", type="negative", multi_line=True)
            return
        finally:
            notification.dismiss()
        self.rewrite_result = result
        self.query_input.value = result.query
        self.render_rewrite_panel()
        await self.search()

    def render_rewrite_panel(self):
        self.rewrite_panel.clear()
        result = self.rewrite_result
        if result is None:
            return
        with (
            self.rewrite_panel,
            ui.card()
            .classes("w-full p-2")
            .props("flat bordered")
            .mark("rewrite-panel"),
        ):
            with ui.row().classes("w-full items-center gap-1"):
                ui.icon("auto_fix_high", color="primary")
                ui.label(f"Rewritten by {result.rewriter}").classes("font-medium")
                ui.label(f"from “{result.original}”").classes("text-sm text-grey-8")
                ui.space()
                ui.button("Restore", icon="undo", on_click=self.restore_query).props(
                    "flat dense no-caps"
                ).mark("rewrite-restore")
                ui.button(icon="close", on_click=self.close_rewrite).props(
                    "flat dense round size=sm"
                )
            self.combination = CombinationControls(
                result.combine,
                result.query_weight,
                "rewrite-combine",
                on_change=self.recombine,
            )
            self.render_rewrite_timings(result)
            with ui.row().classes("gap-1"):
                for keyword in result.keywords:
                    ui.chip(keyword).props("dense outline square").classes("text-xs")
            with (
                ui.expansion(f"Raw outputs ({len(result.outputs)})")
                .props("dense")
                .classes("w-full text-xs")
            ):
                for output in result.outputs:
                    ui.label(output).classes("font-mono whitespace-pre-wrap")

    def render_rewrite_timings(self, result: Rewrite):
        generation = result.timings.get("generation")
        if generation is None:
            return
        text = f"⏱ inference {ms(generation.wall)} (CPU {ms(generation.cpu)})"
        if loading := result.timings.get("loading"):
            text += f" · model loading {ms(loading.wall)}"
        if result.cached:
            text += " · cached (time of the first rewrite)"
        ui.label(text).classes("text-xs text-grey-8").tooltip(
            "Wall-clock and CPU time of the whole process (inference uses "
            "several threads; on a GPU, CPU time is mostly waiting)"
        ).mark("rewrite-time")

    async def recombine(self):
        """Searches the same outputs combined differently (no generation)"""
        result = self.rewrite_result
        if result is None:
            return
        try:
            result = recombine(
                result, self.combination.combine, self.combination.query_weight
            )
        except (KeyError, IndexError, ValueError) as e:
            ui.notify(f"Invalid template: {e}", type="warning")
            return
        self.rewrite_result = result
        if self.query_input.value != result.query:
            self.query_input.value = result.query
            await self.search()

    async def restore_query(self):
        if self.rewrite_result is None:
            return
        self.query_input.value = self.rewrite_result.original
        self.close_rewrite()
        await self.search()

    def close_rewrite(self):
        self.rewrite_result = None
        self.rewrite_panel.clear()

    async def preload_engine(self):
        try:
            await run.io_bound(self.services.engines.get, self.collection)
        except Exception as e:
            # Usually a settings problem, reported in the page
            logger.warning("Could not open %s: %s", self.collection, e)
            ui.notify(f"Could not open {self.collection}: {e}", type="negative")
            return
        self.update_feedback()

    async def show_index_info(self):
        config = self.services.workspace.collection(self.collection)
        details = await run.io_bound(index_details, config.index_path)
        engine = self.services.engines.loaded(self.collection)
        if engine is not None:
            details.insert(
                0,
                Detail("Index", "Vocabulary", f"{engine.index.num_postings():,} terms"),
            )
        details.append(
            Detail(
                "Search",
                "Scoring",
                f"BM25 at query time (k1={self.bm25.k1}, b={self.bm25.b})",
            )
        )
        details.append(
            Detail(
                "Search",
                "Loaded",
                "in memory" if config.in_memory else "memory-mapped",
            )
        )
        with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full"):
            ui.label(f"Index of {self.collection}").classes("text-lg font-bold")
            ui.label(str(config.index_path)).classes(
                "text-xs font-mono text-grey-7 break-all"
            )
            ui.table(
                columns=[
                    {
                        "name": "section",
                        "label": "",
                        "field": "section",
                        "align": "left",
                    },
                    {"name": "name", "label": "", "field": "name", "align": "left"},
                    {"name": "value", "label": "", "field": "value", "align": "left"},
                ],
                rows=[
                    {"id": i, "section": d.section, "name": d.name, "value": d.value}
                    for i, d in enumerate(details)
                ],
                row_key="id",
            ).props("dense flat hide-header wrap-cells").classes("w-full").mark(
                "index-details"
            )
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    def show_help(self):
        examples = [
            "quick brown fox",
            "#combine:0=2(fox dog)",
            "#syn(car automobile) price",
            "#band(fox #syn(dog hound))",
            "#1(new york) hotel",
            "#uw8(fox dog)",
        ]
        with ui.dialog() as dialog, ui.card().classes("w-[720px] max-w-full"):
            ui.label("Query syntax").classes("text-lg font-bold")
            ui.markdown(HELP)
            ui.label("Examples (click to use)").classes("font-medium")
            with ui.row().classes("gap-1"):
                for example in examples:

                    def use(example=example):
                        self.query_input.value = example
                        dialog.close()

                    ui.chip(example, on_click=use).props("outline square").classes(
                        "font-mono"
                    )
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    async def set_stop_words(self, remove: bool):
        self.keep_stop_words = not remove
        self.update_feedback()
        if self.result is not None:
            await self.search()

    def stop_word_notes(self, engine, analyzed) -> list[str]:
        """Explains what searching stop words can find in this index"""
        if not self.keep_stop_words:
            return []
        notes = []
        for term in analyzed.terms:
            if all(w in engine.stop_words for w in term.words):
                notes.append(
                    f"{'/'.join(term.words)} is a stop word of this index: "
                    f"only {term.df:,} documents contain it"
                )
        return notes

    def update_feedback(self, *_):
        """Live parse of the query: syntax errors, operators, word resolution"""
        text = self.query_input.value or ""
        structured = is_structured(text)
        self.feedback_kind.text = "structured" if structured else "text"
        self.feedback_kind.props(f"color={'purple' if structured else 'grey-7'}")
        self.feedback_kind.set_visibility(bool(text.strip()))
        self.feedback_message.text = ""
        self.feedback_message.classes(replace="text-xs text-grey-7")
        engine = self.services.engines.loaded(self.collection)
        if engine is None or not text.strip():
            self.feedback.content = ""
            return
        try:
            analyzed = engine.analyze(text, keep_stop_words=self.keep_stop_words)
            colors = term_colors(analyzed)
            if structured:
                parsed = analyzed.parsed
                self.feedback.content = query_html(parsed.nodes, colors)
                notes = list(parsed.warnings)
                notes += [
                    f"{n.label} is dropped: {n.dropped}"
                    for n in _operators(parsed.nodes)
                    if n.dropped
                ]
                if parsed.uses_positions() and not engine.has_positions:
                    notes.append(
                        "#1/#uwN need an index built with positions "
                        "(impact-explorer index --positions)"
                    )
                if not parsed.term_ids():
                    notes.insert(0, "No word of the query is in the index")
            else:
                words = engine.plain_words(text, self.keep_stop_words)
                self.feedback.content = query_html(words, colors)
                notes = (
                    []
                    if any(w.term_id is not None for w in words)
                    else ["No word of the query is in the index"]
                )
                if "(" in text or ")" in text:
                    notes.append(
                        "Parentheses are ignored in text queries; "
                        "use an operator such as #combine(…)"
                    )
        except QuerySyntaxError as e:
            self.feedback.content = error_html(text, e.start, e.end)
            self.feedback_message.text = f"Syntax error: {e.message}"
            self.feedback_message.classes(replace="text-xs text-negative")
            return
        notes += self.stop_word_notes(engine, analyzed)
        if notes:
            self.feedback_message.text = " · ".join(notes)
            self.feedback_message.classes(replace="text-xs text-warning")

    def build_topics_panel(self):
        datasets = list(self.services.workspace.collection(self.collection).datasets)
        with ui.row().classes("w-full items-center no-wrap gap-1"):
            self.dataset_select = (
                ui.select(
                    datasets,
                    value=datasets[0] if datasets else None,
                    label="datamaestro IR dataset",
                    new_value_mode="add-unique",
                    on_change=lambda e: self.on_dataset(e.value),
                )
                .props("dense outlined use-input input-debounce=0")
                .classes("grow")
                .tooltip(
                    "Pick a dataset, or type a datamaestro id and press Enter "
                    "to add it to the collection"
                )
                .mark("dataset")
            )
            ui.button(icon="playlist_remove", on_click=self.remove_dataset).props(
                "flat dense round"
            ).tooltip("Remove this dataset from the collection")
        self.field_select = (
            ui.select(
                {DEFAULT_FIELD: DEFAULT_FIELD},
                value=DEFAULT_FIELD,
                label="Query from",
                on_change=lambda e: self.set_query_field(e.value),
            )
            .props("dense outlined options-dense")
            .classes("w-full")
            .tooltip("Topic field used as query (and for Evaluate all)")
            .mark("query-field")
        )
        self.field_select.set_visibility(False)
        self.topics_filter = (
            ui.input("Filter topics").props("dense clearable").classes("w-full")
        )
        with ui.row().classes("w-full items-center no-wrap gap-1"):
            self.evaluate_button = (
                ui.button("Evaluate all", icon="analytics", on_click=self.evaluate_all)
                .props("dense flat no-caps")
                .tooltip(
                    "Runs every assessed topic with the current parameters "
                    "(top-K, depth, BM25) to sort topics by performance"
                )
                .mark("evaluate")
            )
            self.evaluate_progress = ui.linear_progress(
                value=0, show_value=False
            ).classes("grow")
            self.evaluate_cancel = ui.button(
                icon="close", on_click=self.cancel_evaluation
            ).props("flat dense round size=sm")
            self.evaluate_progress.set_visibility(False)
            self.evaluate_cancel.set_visibility(False)
        self.evaluate_summary = (
            ui.label().classes("text-xs text-grey-8").mark("evaluate-summary")
        )
        self.batch_run: BatchRun | None = None
        self.saved_results: dict[str, SavedResult] = {}
        self.batch_timer = None
        self.topics_table = (
            ui.table(
                columns=TOPIC_COLUMNS,
                rows=[],
                row_key="id",
                pagination=15,
            )
            .props("dense flat wrap-cells")
            .classes("w-full")
            .mark("topics")
        )
        self.topics_table.bind_filter_from(self.topics_filter, "value")
        self.topics_table.on("rowClick", lambda e: self.select_topic(e.args[1]["id"]))
        if datasets:
            ui.timer(0, lambda: self.load_topics(datasets[0]), once=True)

    # --- Topics and saved queries

    async def on_dataset(self, dataset_id: str | None):
        if dataset_id == self.requested_dataset:
            return
        if await self.load_topics(dataset_id):
            collection = self.services.workspace.collection(self.collection)
            if dataset_id not in collection.datasets:
                self.services.workspace.add_dataset(self.collection, dataset_id)
                ui.notify(f"Added {dataset_id} to {self.collection}", type="positive")

    def remove_dataset(self):
        dataset_id = self.dataset_select.value
        if not dataset_id:
            return
        self.services.workspace.remove_dataset(self.collection, dataset_id)
        datasets = list(self.services.workspace.collection(self.collection).datasets)
        self.dataset_select.set_options(
            datasets, value=datasets[0] if datasets else None
        )
        ui.notify(f"Removed {dataset_id} from {self.collection}")

    def open_settings(self):
        def saved(name: str | None):
            ui.navigate.to(f"/?collection={name}" if name else "/")

        SettingsDialog(self.services, self.collection, saved).open()

    async def load_topics(self, dataset_id: str | None) -> bool:
        self.requested_dataset = dataset_id
        self.topics_table.rows = []
        self.topic_set = None
        if not dataset_id:
            return False
        self.topics_table.props("loading")
        try:
            topic_set = await run.io_bound(self.services.topic_sets.get, dataset_id)
        except Exception as e:
            logger.exception("Could not load %s", dataset_id)
            ui.notify(f"Could not load {dataset_id}: {e}", type="negative")
            return False
        finally:
            self.topics_table.props(remove="loading")
        self.topic_set = topic_set
        names = topic_set.field_names()
        if self.query_field not in names:
            self.query_field = DEFAULT_FIELD
        self.rewrite_result: Rewrite | None = None
        self.keep_stop_words = False
        """Disables the index's stop list (query switch)"""
        self.field_select.set_options(names, value=self.query_field)
        self.field_select.set_visibility(len(names) > 1)
        await self.refresh_topic_rows()
        return True

    def version_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.services.store.list(self.collection):
            if q.source is not None and q.source.dataset == self.topic_set.dataset_id:
                counts[q.source.topic_id] = counts.get(q.source.topic_id, 0) + 1
        return counts

    def base_columns(self) -> list[dict]:
        """Topic columns (with the number of saved versions, if any)"""
        if not self.version_counts():
            return TOPIC_COLUMNS
        return TOPIC_COLUMNS + [
            {
                "name": "versions",
                "label": "saved",
                "field": "versions",
                "sortable": True,
            }
        ]

    async def refresh_topic_rows(self):
        topic_set = self.topic_set
        counts = self.version_counts()
        self.topics_table.columns = self.base_columns()
        self.topics_table.rows = [
            {
                "id": t.topic_id,
                "text": t.query(self.query_field),
                "rel": topic_set.relevant_count(t.topic_id),
                "versions": counts.get(t.topic_id) or None,
            }
            for t in topic_set.topics.values()
        ]
        self.topics_table.pagination = {"rowsPerPage": 15}
        self.evaluate_summary.text = ""
        # Results computed earlier (by any client) with the same parameters
        batch = self.services.batch_runs.get(self.batch_key())
        if batch is not None:
            await self.follow(batch)
        else:
            self.batch_run = None
            if self.saved_results:
                self.saved_results = {}
                self.refresh_saved()

    async def set_query_field(self, name: str | None):
        if not name or name == self.query_field or self.topic_set is None:
            return
        self.query_field = name
        await self.refresh_topic_rows()

    # --- Topic-level evaluation

    def batch_key(self) -> RunKey | None:
        if self.topic_set is None:
            return None
        params = BM25Params(**self.bm25.__dict__)
        return RunKey.of(
            self.collection,
            self.topic_set.dataset_id,
            self.top_k,
            self.depth,
            params,
            self.query_field,
            fingerprint(self.services.workspace.collection(self.collection)),
            not self.keep_stop_words,
        )

    async def evaluate_all(self):
        if self.topic_set is None:
            return
        try:
            engine = await run.io_bound(self.services.engines.get, self.collection)
        except Exception as e:
            ui.notify(f"Cannot open {self.collection}: {e}", type="negative")
            return
        batch = self.services.batch_runs.start(engine, self.topic_set, self.batch_key())
        if batch.total == 0:
            ui.notify("No topic of this dataset has relevant documents")
            return
        await self.follow(batch)

    def cancel_evaluation(self):
        if self.batch_run is not None:
            self.batch_run.cancel.set()

    async def follow(self, batch: BatchRun):
        """Shows the progress of a batch evaluation, then its results"""
        self.batch_run = batch
        if self.batch_timer is not None:
            self.batch_timer.cancel()
            self.batch_timer = None
        if batch.finished:
            await self.show_batch(batch)
            return
        self.evaluate_progress.set_visibility(True)
        self.evaluate_cancel.set_visibility(True)
        self.evaluate_button.disable()

        async def poll():
            if self.batch_run is not batch:
                return
            self.evaluate_progress.value = batch.done / max(batch.total, 1)
            self.evaluate_summary.text = f"Evaluating {batch.done:,}/{batch.total:,}…"
            if batch.finished:
                self.batch_timer.cancel()
                self.batch_timer = None
                await self.show_batch(batch)

        self.batch_timer = ui.timer(0.3, poll)

    async def show_batch(self, batch: BatchRun):
        self.evaluate_progress.set_visibility(False)
        self.evaluate_cancel.set_visibility(False)
        self.evaluate_button.enable()
        if batch.error:
            self.evaluate_summary.text = f"Evaluation failed: {batch.error}"
            return
        if not batch.per_topic or self.topic_set is None:
            self.evaluate_summary.text = "Evaluation cancelled"
            return
        if batch.key != self.batch_key():
            return
        means = batch.means()
        names = list(means)
        bm25 = "k1={:g} b={:g}".format(*batch.key.bm25[:2])
        status = "" if batch.complete else " (cancelled: partial)"
        field = "" if batch.key.field == DEFAULT_FIELD else f" ({batch.key.field})"
        if not batch.key.stop_words:
            field += " (stop words kept)"
        self.evaluate_summary.text = (
            f"Mean over {len(batch.per_topic):,} topics{field}{status}: "
            + " · ".join(f"{n} {v:.3f}" for n, v in means.items())
            + f" — {bm25}, depth {batch.key.depth}"
        )
        saved = await self.evaluate_saved(batch)
        main = names[0]
        best: dict[str, float] = {}
        for result in saved.values():
            value = result.metrics[main]
            best[result.topic_id] = max(value, best.get(result.topic_id, value))
        columns = [{"name": n, "label": n, "field": n, "sortable": True} for n in names]
        if best:
            columns[1:1] = [
                {"name": "saved", "label": "saved", "field": "saved", "sortable": True},
                {"name": "delta", "label": "Δ", "field": "delta", "sortable": True},
            ]
            gains = [
                best[t] - batch.per_topic[t][main] for t in best if t in batch.per_topic
            ]
            self.evaluate_summary.text += (
                f" · saved queries: {len(gains)} topics, mean Δ{main} "
                f"{sum(gains) / max(len(gains), 1):+.3f}"
            )
        self.topics_table.columns = self.base_columns() + columns
        for row in self.topics_table.rows:
            metrics = batch.per_topic.get(row["id"], {})
            for n in names:
                row[n] = round(metrics[n], 4) if n in metrics else None
            if best:
                saved_value = best.get(row["id"])
                row["saved"] = None if saved_value is None else round(saved_value, 4)
                row["delta"] = (
                    round(saved_value - metrics[main], 4)
                    if saved_value is not None and main in metrics
                    else None
                )
        # Worst topics first: the interesting ones to look at
        self.topics_table.pagination = {
            "rowsPerPage": 15,
            "sortBy": names[0],
            "descending": False,
        }
        self.topics_table.update()

    def modified_saved_queries(self, batch: BatchRun) -> list[SavedQuery]:
        """Saved queries of this dataset's topics that differ from the topic"""
        topics = self.topic_set.topics
        return [
            q
            for q in self.services.store.list(self.collection)
            if q.source is not None
            and q.source.dataset == batch.key.dataset
            and q.source.topic_id in topics
            and q.text.strip() != topics[q.source.topic_id].query(batch.key.field)
        ]

    async def evaluate_saved(self, batch: BatchRun) -> dict[str, SavedResult]:
        queries = self.modified_saved_queries(batch)
        if not queries:
            self.saved_results = {}
            return {}
        try:
            engine = await run.io_bound(self.services.engines.get, self.collection)
            results = await run.io_bound(
                self.services.batch_runs.evaluate_saved,
                engine,
                self.topic_set,
                batch,
                queries,
            )
        except Exception as e:
            logger.warning("Could not evaluate saved queries: %s", e)
            results = {}
        self.saved_results = results
        self.refresh_saved()
        return results

    async def select_topic(self, topic_id: str):
        topic = self.topic_set.topics[topic_id]
        self.source = QuerySource(self.topic_set.dataset_id, topic_id, self.query_field)
        self.query_input.value = topic.query(self.query_field)
        self.close_rewrite()
        await self.search()

    async def use_topic_field(self, name: str):
        """Searches with another field of the current topic"""
        topic = self.current_topic()
        if topic is None:
            return
        self.source = QuerySource(self.source.dataset, self.source.topic_id, name)
        self.query_input.value = topic.query(name)
        await self.search()

    def current_topic(self):
        if self.source is None or self.topic_set is None:
            return None
        if self.topic_set.dataset_id != self.source.dataset:
            return None
        return self.topic_set.topics.get(self.source.topic_id)

    def render_topic_panel(self):
        """All the fields of the current topic (title, description, ...)"""
        self.topic_panel.clear()
        topic = self.current_topic()
        if topic is None:
            return
        text = (self.query_input.value or "").strip()
        with self.topic_panel:
            with (
                ui.expansion(
                    f"Topic {topic.topic_id}",
                    icon="topic",
                    value=len(topic.fields) > 1,
                )
                .props("dense")
                .classes("w-full text-sm")
                .mark("topic-panel")
            ):
                for name, value in topic.fields.items():
                    original = name == self.source.field
                    with ui.row().classes("w-full items-start no-wrap gap-2"):
                        ui.label(name).classes(
                            "w-24 shrink-0 font-medium"
                            + (" text-primary" if original else " text-grey-8")
                        ).tooltip("Original query of this search" if original else "")
                        ui.label(value).classes("grow whitespace-pre-wrap")
                        ui.button(
                            icon="search",
                            on_click=lambda n=name: self.use_topic_field(n),
                        ).props(
                            "flat dense round size=sm"
                            + (" color=primary" if original and value == text else "")
                        ).tooltip(f"Search with the {name}").mark(f"use-field-{name}")
                self.render_versions(topic, text)

    def topic_versions(self, dataset_id: str, topic_id: str) -> list[SavedQuery]:
        """Saved queries linked to a topic (its modified versions)"""
        return [
            q
            for q in self.services.store.list(self.collection)
            if q.source is not None
            and q.source.dataset == dataset_id
            and q.source.topic_id == topic_id
        ]

    def render_versions(self, topic, text: str):
        versions = self.topic_versions(self.source.dataset, topic.topic_id)
        if not versions:
            return
        ui.label(f"Saved versions ({len(versions)})").classes(
            "font-medium text-grey-8 mt-2"
        )
        for query in versions:
            current = query.text == text
            with (
                ui.row()
                .classes(
                    "w-full items-center no-wrap gap-2 cursor-pointer rounded px-1"
                    + (" bg-blue-1" if current else "")
                )
                .on("click", lambda q=query: self.load_saved(q))
                .mark(f"version-{query.id}")
            ):
                ui.icon("bookmark", size="xs", color="primary" if current else "grey-6")
                with ui.column().classes("grow gap-0"):
                    if query.name:
                        ui.label(query.name).classes("font-medium")
                    ui.label(query.text).classes("whitespace-pre-wrap")
                self.saved_metrics(query)

    def refresh_saved(self):
        self.saved_panel.clear()
        queries = self.services.store.list(self.collection)
        with self.saved_panel:
            if not queries:
                ui.label("No saved query for this collection").classes(
                    "text-grey-6 p-2"
                )
            for query in queries:
                with ui.card().classes("w-full p-2").props("flat bordered"):
                    with ui.row().classes("w-full items-start no-wrap"):
                        with (
                            ui.column()
                            .classes("grow gap-0 cursor-pointer")
                            .on("click", lambda q=query: self.load_saved(q))
                        ):
                            ui.label(query.name or query.text).classes("font-medium")
                            if query.name:
                                ui.label(query.text).classes("text-sm text-grey-8")
                            details = query.created[:10]
                            if query.source:
                                details += (
                                    f" · {query.source.dataset}#{query.source.topic_id}"
                                )
                            ui.label(details).classes("text-xs text-grey-6")
                            self.saved_metrics(query)
                            if query.note:
                                ui.label(query.note).classes("text-xs italic")
                        ui.button(
                            icon="delete", on_click=lambda q=query: self.delete_saved(q)
                        ).props("flat dense round size=sm color=grey-7")

    def saved_metrics(self, query: SavedQuery):
        """Score of a saved query vs. its topic (after an Evaluate all)"""
        result = self.saved_results.get(query.id)
        batch = self.batch_run
        if result is None or batch is None:
            return
        name = next(iter(result.metrics))
        line = f"{name} {result.metrics[name]:.3f}"
        topic = batch.per_topic.get(result.topic_id)
        color = "text-grey-8"
        if topic is not None:
            delta = result.metrics[name] - topic[name]
            line += f" ({delta:+.3f} vs. topic {batch.key.field})"
            color = (
                "text-positive"
                if delta > 0
                else "text-negative"
                if delta < 0
                else color
            )
        ui.label(line).classes(f"text-xs {color}").mark(f"saved-metrics-{query.id}")

    async def load_saved(self, query: SavedQuery):
        self.source = query.source
        if query.source is not None:
            dataset_id = query.source.dataset
            if self.topic_set is None or self.topic_set.dataset_id != dataset_id:
                await self.load_topics(dataset_id)
            options = list(self.dataset_select.options or [])
            if dataset_id not in options:
                options.append(dataset_id)
            # requested_dataset == dataset_id, so this does not reload
            self.dataset_select.set_options(options, value=dataset_id)
            if query.source.field != self.query_field and self.topic_set is not None:
                if query.source.field in self.topic_set.field_names():
                    self.field_select.set_value(query.source.field)
        self.query_input.value = query.text
        await self.search()

    def delete_saved(self, query: SavedQuery):
        self.services.store.delete(query.id)
        self.refresh_saved()
        self.after_saved_change()

    def after_saved_change(self):
        """Saved queries changed: topic panel and version counts"""
        self.render_topic_panel()
        if self.topic_set is not None:
            counts = self.version_counts()
            for row in self.topics_table.rows:
                row["versions"] = counts.get(row["id"]) or None
            names = {c["name"] for c in self.topics_table.columns}
            if counts and "versions" not in names:
                self.topics_table.columns = self.base_columns() + [
                    c for c in self.topics_table.columns if c not in TOPIC_COLUMNS
                ]
            self.topics_table.update()

    def save_dialog(self):
        text = (self.query_input.value or "").strip()
        if not text:
            ui.notify("Nothing to save: the query is empty", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Save query for collection “{self.collection}”").classes(
                "font-bold"
            )
            ui.label(text).classes("text-sm text-grey-8")
            name = ui.input("Name").classes("w-full").mark("save-name")
            note = ui.textarea("Note").classes("w-full").props("autogrow")
            link = None
            if self.source is not None:
                link = ui.checkbox(
                    f"Link to topic {self.source.dataset}#{self.source.topic_id}",
                    value=True,
                )

            def save():
                source = self.source if link is not None and link.value else None
                self.services.store.save(
                    SavedQuery(
                        collection=self.collection,
                        text=text,
                        name=name.value or "",
                        note=note.value or "",
                        source=source,
                    )
                )
                dialog.close()
                self.refresh_saved()
                self.after_saved_change()
                ui.notify("Query saved", type="positive")

            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", on_click=save).mark("save-confirm")
        dialog.open()

    def set_collection(self, name: str):
        # Everything (topics, saved queries, cached documents) depends on the
        # collection: start from a fresh page
        ui.navigate.to(f"/?collection={name}")

    # --- Search and rendering

    async def search(self):
        text = (self.query_input.value or "").strip()
        if not text:
            return
        params = BM25Params(**self.bm25.__dict__)
        depth = max(self.depth, self.top_k)
        try:
            engine = await run.io_bound(self.services.engines.get, self.collection)
            self.result = await run.io_bound(
                engine.search, text, depth, params, None, self.keep_stop_words
            )
        except QuerySyntaxError as e:
            self.update_feedback()
            if not self.feedback_message.text.startswith("Syntax error"):
                # Reported by impact-index only (e.g. no positions)
                self.feedback_message.text = f"Error: {e.message}"
                self.feedback_message.classes(replace="text-xs text-negative")
            ui.notify(f"Invalid query: {e.message}", type="negative")
            return
        except ConfigError as e:
            logger.warning("Cannot search %s: %s", self.collection, e)
            ui.notify(f"Collection settings: {e}", type="negative", multi_line=True)
            return
        except Exception as e:
            logger.exception("Search failed")
            ui.notify(f"Search failed: {e}", type="negative")
            return
        self.evaluation = evaluate(self.result, self.qrels, self.top_k)
        self.baseline = await self.original_results(engine, text, depth, params)
        wanted = [h.docid for h in self.result.top(self.top_k)] + [
            m.docid for m in self.evaluation.missed
        ]
        missing = [d for d in dict.fromkeys(wanted) if d not in self.documents]
        if missing:
            fetched = await run.io_bound(engine.documents.get, missing)
            self.documents.update(zip(missing, fetched, strict=True))
        self.render()

    def render_timings(self, result: SearchResult):
        retrieval = result.timings.get("retrieval")
        if retrieval is None:
            return
        analysis = result.timings.get("analysis")
        text = f"⏱ {ms(retrieval.wall)} (CPU {ms(retrieval.cpu)})"
        tooltip = (
            f"Retrieval (impact-index): {ms(retrieval.wall)} wall-clock, "
            f"{ms(retrieval.cpu)} CPU"
        )
        if analysis is not None:
            tooltip += (
                f" · query analysis (explorer, incl. df and stems): "
                f"{ms(analysis.wall)} wall-clock, {ms(analysis.cpu)} CPU"
            )
        ui.label(text).classes("text-sm text-grey-8").tooltip(tooltip).mark(
            "query-time"
        )

    async def original_results(self, engine, text, depth, params):
        """Results of the original topic query, if the query was edited"""
        topic = self.current_topic()
        if topic is None:
            return None
        original = topic.query(self.source.field)
        # Also compared when only the stop list was switched off
        if original == text and not self.keep_stop_words:
            return None
        key = (
            self.source.dataset,
            self.source.topic_id,
            self.source.field,
            self.top_k,
            depth,
            tuple(params.__dict__.values()),
        )
        if key not in self._baselines:
            try:
                result = await run.io_bound(
                    engine.search, original, depth, params, False
                )
            except Exception as e:
                logger.warning("Could not run the original query: %s", e)
                return None
            self._baselines[key] = Baseline(
                metrics=evaluate(result, self.qrels, self.top_k).metrics,
                ranks=result.ranks,
                depth=depth,
            )
        return self._baselines[key]

    def set_filter(self, value: str):
        self.filter = value
        self.render_top()

    def detach(self):
        self.source = None
        self.baseline = None
        if self.result is not None:
            self.evaluation = evaluate(self.result, self.qrels, self.top_k)
        self.render()

    def update_source_label(self):
        self.detach_button.set_visibility(self.source is not None)
        if self.source is None:
            self.source_label.text = "Free-text query (no assessments)"
        else:
            n = sum(1 for rel in self.qrels.values() if rel > 0)
            self.source_label.text = (
                f"Topic {self.source.dataset}#{self.source.topic_id} · {n} relevant"
            )

    def render(self):
        self.update_source_label()
        self.render_topic_panel()
        self.terms_row.clear()
        self.metrics_row.clear()
        if self.result is None:
            self.top_list.clear()
            self.missed_list.clear()
            return
        colors = term_colors(self.result.query)
        with self.terms_row:
            if not self.result.query.terms:
                ui.label("No query term is in the index vocabulary").classes(
                    "text-negative"
                )
            for term in self.result.query.terms:
                with (
                    ui.element("span")
                    .classes("px-2 py-0.5 rounded text-sm")
                    .style(f"background:{colors[term.term_id]}")
                ):
                    words = [w for w in term.words if w != term.stem]
                    label = html.escape(term.label)
                    if term.stem and not term.stem_verified:
                        label = f"<i>{label}</i>"
                    ui.html(
                        f"<b>{label}</b> "
                        f"<span class='text-grey-8'>df={term.df:,}</span>"
                        + (f" ×{term.weight:g}" if term.weight != 1 else "")
                        + (
                            f" <span class='text-xs text-grey-7'>← "
                            f"{html.escape(', '.join(words))}</span>"
                            if term.stem and words
                            else ""
                        ),
                        sanitize=False,
                    ).mark(f"term-{term.term_id}")
                    ui.tooltip(
                        f"term #{term.term_id}"
                        + (
                            f" (index form: {term.stem})"
                            if term.stem and term.stem_verified
                            else f" (index form: probably {term.stem}, not verified)"
                            if term.stem
                            else " (index form not recovered)"
                        )
                        + f", query words: {', '.join(term.words)}"
                        + f", weight {term.weight:g}"
                    )
        with self.metrics_row:
            ui.label(f"{len(self.result.hits):,} hits (depth {self.depth})").classes(
                "text-sm"
            )
            self.render_timings(self.result)
            if self.qrels:
                for name, value in self.evaluation.metrics.items():
                    with ui.badge(f"{name} {value:.3f}").props("outline color=primary"):
                        original = (
                            self.baseline.metrics.get(name) if self.baseline else None
                        )
                        if original is not None:
                            delta = value - original
                            color = (
                                "text-positive"
                                if delta > 1e-9
                                else "text-negative"
                                if delta < -1e-9
                                else "text-grey-7"
                            )
                            ui.label(f"{delta:+.3f}").classes(f"ml-1 {color}")
                            ui.tooltip(f"original query: {original:.3f}")
                if self.baseline is not None:
                    ui.label(f"vs. original {self.source.field}").classes(
                        "text-xs text-grey-7"
                    ).mark("vs-original")
        self.top_tab.props(f'label="Top-{self.top_k}"')
        self.missed_tab.props(f'label="Not retrieved ({len(self.evaluation.missed)})"')
        self.render_top()
        self.render_missed()

    def render_top(self):
        self.top_list.clear()
        if self.result is None:
            return
        colors = term_colors(self.result.query)
        with self.top_list:
            for hit, lbl in zip(
                self.result.top(self.top_k), self.evaluation.labels, strict=True
            ):
                if self.filter != "all" and lbl.value != self.filter:
                    continue
                self.document_card(
                    hit.docid,
                    colors,
                    lbl,
                    header=f"#{hit.rank}",
                    score=hit.score,
                    relevance=self.qrels.get(hit.docid),
                    rank=hit.rank,
                )
            if not self.result.hits:
                ui.label("No result").classes("text-grey-6")

    def render_missed(self):
        self.missed_list.clear()
        colors = term_colors(self.result.query)
        with self.missed_list:
            if not self.qrels:
                ui.label(
                    "No assessments: pick a topic to see missed documents"
                ).classes("text-grey-6")
            elif not self.evaluation.missed:
                ui.label(f"All relevant documents are in the top-{self.top_k}").classes(
                    "text-positive"
                )
            for missed in self.evaluation.missed:
                self.document_card(
                    missed.docid,
                    colors,
                    Label.RELEVANT,
                    header=f"#{missed.rank}" if missed.rank else f"> {self.depth}",
                    score=missed.score,
                    relevance=missed.relevance,
                    rank=missed.rank,
                )

    def original_rank(self, docid: str, rank: int | None):
        """Rank of a document for the original query (when it was edited)"""
        if self.baseline is None:
            return
        before = self.baseline.ranks.get(docid)
        if before is None:
            if rank is not None:
                ui.label("new").classes(
                    "text-xs px-1 rounded bg-blue-1 text-blue-9"
                ).tooltip(
                    f"Not retrieved by the original query (depth {self.baseline.depth})"
                ).mark(f"orig-{docid}")
            return
        if rank is None or before == rank:
            icon, color = "drag_handle", "grey-7"
        elif rank < before:
            icon, color = "arrow_upward", "positive"
        else:
            icon, color = "arrow_downward", "negative"
        with (
            ui.row()
            .classes("items-center gap-0 text-xs")
            .tooltip("Rank for the original query")
            .mark(f"orig-{docid}")
        ):
            ui.icon(icon, color=color, size="xs")
            ui.label(f"was #{before}").classes(f"text-{color}")

    def document_card(self, docid, colors, lbl, *, header, score, relevance, rank=None):
        document = self.documents.get(docid)
        color, icon = LABEL_STYLE[lbl]
        query_terms = set(colors)
        with (
            ui.card()
            .classes("result-card p-3")
            .props("flat bordered")
            .mark(f"doc-{docid}")
        ):
            segs = (
                segments(
                    document.text,
                    query_terms,
                    self.engine.matcher_for(self.keep_stop_words),
                )
                if document
                else []
            )
            counts = term_counts(segs)
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(header).classes("font-mono font-bold")
                ui.icon(icon, color=color).tooltip(
                    lbl.value
                    + (f" (rel={relevance:g})" if relevance is not None else "")
                )
                ui.label(docid).classes("font-mono text-sm")
                self.original_rank(docid, rank)
                if score is not None:
                    ui.label(f"score {score:.4f}").classes("text-sm text-grey-7")
                ui.space()
                for term in self.result.query.terms:
                    if counts.get(term.term_id):
                        ui.label(f"{term.label}×{counts[term.term_id]}").classes(
                            "text-xs px-1 rounded"
                        ).style(f"background:{colors[term.term_id]}")
            if document is None:
                ui.label("Document not found in the document store").classes(
                    "text-negative text-sm"
                )
                return
            if document.title:
                ui.label(document.title).classes("font-medium")
            ui.html(to_html(snippet(segs), colors), sanitize=False).classes(
                "doc-text text-sm"
            )
            if len(document.text) > 400:
                expansion = (
                    ui.expansion("Full document")
                    .props("dense")
                    .classes("text-sm w-full")
                )

                def show_full(e, expansion=expansion, segs=segs):
                    if e.value and not expansion.default_slot.children:
                        with expansion:
                            ui.html(to_html(segs, colors), sanitize=False).classes(
                                "doc-text"
                            )

                expansion.on_value_change(show_full)


def _operators(nodes):
    from .query_syntax import Operator

    for node in nodes:
        if isinstance(node, Operator):
            yield node
            yield from _operators(node.children)


def create_app(services: Services):
    @ui.page("/")
    async def index(collection: str | None = None, saved: str | None = None):
        query = services.store.get(saved) if saved else None
        if query is not None:
            collection = query.collection
        page = ExplorerPage(services, collection)
        page.build()
        if query is not None and query.collection == page.collection:
            await page.load_saved(query)
