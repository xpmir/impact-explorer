"""Workspace settings dialog: collections, index paths and datasets,
rewriters, and index builds."""

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from nicegui import run, ui

from .builds import (
    PIPELINES,
    STAGE_LABELS,
    STAGES,
    STOP_WORDS,
    BuildSpec,
    BuildState,
    iter_states,
)
from .checks import (
    Check,
    Status,
    check_dataset,
    check_documents,
    check_index,
    check_name,
)
from .config import BM25Params, CollectionConfig, ConfigError

if TYPE_CHECKING:
    from .ui import Services

logger = logging.getLogger(__name__)

BUILD_STATUS_STYLE = {
    "pending": ("schedule", "grey"),
    "running": ("sync", "primary"),
    "done": ("check_circle", "positive"),
    "failed": ("error", "negative"),
    "cancelled": ("pause_circle", "warning"),
    "interrupted": ("pause_circle", "warning"),
}

STATUS_STYLE = {
    Status.OK: ("check_circle", "positive"),
    Status.WARNING: ("warning", "warning"),
    Status.ERROR: ("error", "negative"),
}


CUSTOM = "Custom template"


class CombinationControls:
    """How a rewrite becomes the searched query: a predefined combination
    (or a custom template) and the weight of the original query"""

    def __init__(self, combine: str, query_weight: int, marker: str, on_change=None):
        from .rewriters import COMBINATIONS, combination_name

        self.on_change = on_change
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            self.mode = (
                ui.select(
                    [*COMBINATIONS, CUSTOM],
                    value=combination_name(combine) or CUSTOM,
                    label="Searched query",
                )
                .props("dense")
                .classes("w-56")
                .tooltip(
                    "Generated only: the outputs alone (as STORM is trained "
                    "and evaluated); original + generated: the query followed "
                    "by the outputs (repeated words weigh more); unique "
                    "keywords: each keyword once"
                )
                .mark(f"{marker}-mode")
            )
            self.weight = (
                ui.number("Original query weight", value=query_weight, min=1, step=1)
                .props("dense")
                .classes("w-40")
                .tooltip("The original query is repeated this many times")
                .mark(f"{marker}-weight")
            )
        self.template = (
            ui.input("Template ({query}, {keywords}, {outputs})", value=combine)
            .props("dense")
            .classes("w-full font-mono")
            .mark(f"{marker}-template")
        )
        self._update_visibility()
        self.mode.on_value_change(self._on_mode)
        self.weight.on_value_change(self._changed)
        self.template.on("blur", self._changed)

    @property
    def combine(self) -> str:
        return self.template.value or "{query} {keywords}"

    @property
    def query_weight(self) -> int:
        try:
            return max(1, int(self.weight.value or 1))
        except (TypeError, ValueError):
            return 1

    def set(self, combine: str, query_weight: int | None = None):
        from .rewriters import combination_name

        self.template.value = combine
        self.mode.value = combination_name(combine) or CUSTOM
        if query_weight is not None:
            self.weight.value = query_weight
        self._update_visibility()

    def _update_visibility(self):
        self.template.set_visibility(self.mode.value == CUSTOM)
        self.weight.set_visibility("{query}" in (self.template.value or ""))

    async def _on_mode(self, e):
        from .rewriters import COMBINATIONS

        if e.value in COMBINATIONS:
            self.template.value = COMBINATIONS[e.value]
        self._update_visibility()
        await self._changed()

    async def _changed(self, *_):
        self._update_visibility()
        if self.on_change is not None:
            result = self.on_change()
            if hasattr(result, "__await__"):
                await result


class CheckLabel:
    """Shows the result of a (background) check below a field"""

    def __init__(self, marker: str):
        self.check: Check | None = None
        self._generation = 0
        with ui.row().classes("items-center gap-1 -mt-2 no-wrap") as self.row:
            self.spinner = ui.spinner(size="xs")
            self.icon = ui.icon("check_circle", size="xs")
            self.label = ui.label().classes("text-xs").mark(marker)
        self.row.set_visibility(False)

    def show(self, check: Check):
        self.check = check
        icon, color = STATUS_STYLE[check.status]
        self.spinner.set_visibility(False)
        self.icon.set_visibility(True)
        self.icon.name = icon
        self.icon.props(f"color={color}")
        self.label.text = check.message
        self.label.classes(replace=f"text-xs text-{color}")
        self.row.set_visibility(bool(check.message) or check.status != Status.OK)

    async def run(self, fn: Callable[..., Check], *args):
        """Runs a check in a thread; only the latest run is displayed"""
        self._generation += 1
        generation = self._generation
        self.row.set_visibility(True)
        self.spinner.set_visibility(True)
        self.icon.set_visibility(False)
        self.label.text = "Checking…"
        self.label.classes(replace="text-xs text-grey-7")
        try:
            check = await run.io_bound(fn, *args)
        except Exception as e:
            logger.exception("Check failed")
            check = Check.error(str(e))
        if generation == self._generation:
            self.show(check)


class SettingsDialog:
    def __init__(self, services: "Services", current: str | None, on_saved):
        self.editing_collection_hint = current
        self.services = services
        self.on_saved = on_saved
        self.workspace = services.workspace
        self.dialog = ui.dialog().props("persistent")
        with self.dialog, ui.card().classes("w-[860px] max-w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label("Workspace").classes("text-lg font-bold")
                ui.label(str(self.workspace.folder)).classes(
                    "text-xs text-grey-7 font-mono"
                )
                ui.space()
                ui.button(icon="close", on_click=self.dialog.close).props(
                    "flat round dense"
                )
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                with ui.column().classes("w-48 gap-1"):
                    ui.label("Collections").classes("text-sm font-medium")
                    self.collection_list = ui.column().classes("w-full gap-0")
                    ui.button(
                        "New", icon="add", on_click=lambda: self.edit(None)
                    ).props("flat dense no-caps").mark("settings-new")
                    ui.label("Rewriters").classes("text-sm font-medium mt-4")
                    self.rewriter_list = ui.column().classes("w-full gap-0")
                    ui.button(
                        "New", icon="add", on_click=lambda: self.edit_rewriter(None)
                    ).props("flat dense no-caps").mark("settings-new-rewriter")
                    ui.label("Index builds").classes("text-sm font-medium mt-4")
                    self.build_list = ui.column().classes("w-full gap-0")
                    ui.button(
                        "New", icon="add", on_click=lambda: self.edit_build(None)
                    ).props("flat dense no-caps").mark("settings-new-build")
                self.form = ui.column().classes("grow gap-2 min-w-0")
        self.editing_rewriter: str | None = None
        self.editing_build: str | None = None
        self.edit(current if current in self.workspace.collections else None)

    def open(self):
        self.dialog.open()

    def refresh_list(self):
        self.collection_list.clear()
        with self.collection_list:
            for name in self.workspace.collections:
                ui.button(name, on_click=lambda n=name: self.edit(n)).props(
                    "flat dense no-caps align=left"
                    + (" color=primary" if name == self.editing else " color=grey-9")
                ).classes("w-full")
        self.rewriter_list.clear()
        with self.rewriter_list:
            for name in self.workspace.rewriters:
                selected = name == self.editing_rewriter
                ui.button(name, on_click=lambda n=name: self.edit_rewriter(n)).props(
                    "flat dense no-caps align=left"
                    + (" color=primary" if selected else " color=grey-9")
                ).classes("w-full text-left break-all").mark(
                    f"settings-rewriter-{name}"
                )
        self.build_list.clear()
        with self.build_list:
            for name, state, alive in iter_states(self.services.builds):
                status = state.status(alive)
                icon, color = BUILD_STATUS_STYLE[status]
                selected = name == self.editing_build
                ui.button(
                    name, icon=icon, on_click=lambda n=name: self.edit_build(n)
                ).props(
                    "flat dense no-caps align=left"
                    + (" color=primary" if selected else " color=grey-9")
                ).classes("w-full text-left break-all").mark(
                    f"settings-build-{name}"
                ).tooltip(status)

    def edit_rewriter(self, name: str | None):
        from .rewriters import (
            BACKENDS,
            GROUP_BEAM_SEARCH,
            PRESETS,
            RewriterConfig,
            check,
            find_preset,
            group_beam_search_enabled,
            set_group_beam_search,
        )

        self.editing, self.editing_rewriter, self.editing_build = None, name, None
        self.refresh_list()
        existing = self.workspace.rewriters.get(name) if name else None
        r = existing or RewriterConfig.preset("")
        self.form.clear()
        with self.form:
            ui.label("Edit rewriter" if existing else "New rewriter").classes(
                "font-medium"
            )
            ui.label(
                "A language model that rewrites (e.g. expands) queries. "
                f"Known models get preset prompts: {', '.join(PRESETS)}"
            ).classes("text-xs text-grey-7")
            name_input = (
                ui.input(
                    "Name", value=r.name, placeholder="e.g. Arthur-75/storm-qwen3-8B"
                )
                .classes("w-full font-mono")
                .mark("rewriter-name")
            )
            if existing:
                name_input.props("readonly")
            with ui.row().classes("w-full gap-2 no-wrap"):
                backend = ui.select(list(BACKENDS), value=r.backend, label="Backend")
                backend.classes("w-40")
                model = ui.input(
                    "Model (default: the name)", value=r.model or ""
                ).classes("grow font-mono")
            url = ui.input(
                "Server URL (OpenAI-compatible)",
                value=r.url or "",
                placeholder="http://localhost:8000/v1",
            ).classes("w-full font-mono")
            url.bind_visibility_from(backend, "value", value="openai")
            system_prompt = (
                ui.textarea("System prompt", value=r.system_prompt)
                .props("autogrow")
                .classes("w-full font-mono")
                .mark("rewriter-system")
            )
            user_template = (
                ui.textarea("User prompt ({query})", value=r.user_template)
                .props("autogrow")
                .classes("w-full font-mono")
            )
            combination = CombinationControls(
                r.combine, r.query_weight, "rewriter-combine"
            )
            generation = (
                ui.textarea(
                    "Generation parameters (JSON)",
                    value=json.dumps(r.generation, indent=1),
                )
                .props("autogrow")
                .classes("w-full font-mono")
            )
            allow_remote = (
                ui.checkbox(
                    f"Group beam search (runs code from hf.co/{GROUP_BEAM_SEARCH})",
                    value=group_beam_search_enabled(r.generation),
                )
                .classes("text-sm")
                .tooltip(
                    "STORM's best setting (num_beam_groups=3, "
                    "diversity_penalty=1.0); transformers moved group beam "
                    "search to a Hub repository: generating with it downloads "
                    "and executes that code. Unchecked: the model's decoding "
                    "(plain beam search for STORM)"
                )
                .mark("rewriter-allow-remote")
            )

            def on_allow(e):
                try:
                    params = json.loads(generation.value or "{}")
                except ValueError:
                    return
                generation.value = json.dumps(
                    set_group_beam_search(params, e.value), indent=1
                )

            allow_remote.on_value_change(on_allow)
            with ui.row().classes("gap-2"):
                device = ui.input("Device", value=r.device).props("dense")
                dtype = ui.select(
                    ["auto", "float32", "bfloat16", "float16"],
                    value=r.dtype,
                    label="dtype",
                ).props("dense")
            status = CheckLabel("check-rewriter")

            def model_name() -> str:
                """The model (the model field, else the name)"""
                return (model.value or "").strip() or (name_input.value or "").strip()

            def apply_preset():
                if find_preset(model_name()) is None:
                    status.show(
                        Check.warning(
                            f"No preset for {model_name()!r} (known: "
                            + ", ".join(PRESETS)
                            + ")"
                        )
                    )
                    return
                preset = RewriterConfig.preset("", model_name())
                status.show(Check.ok(f"Preset of {preset.model_id} applied"))
                system_prompt.value = preset.system_prompt
                user_template.value = preset.user_template
                combination.set(preset.combine, preset.query_weight)
                generation.value = json.dumps(
                    set_group_beam_search(preset.generation, allow_remote.value),
                    indent=1,
                )

            def pristine() -> bool:
                """The form still holds the generic defaults"""
                default = RewriterConfig(name="")
                try:
                    params = json.loads(generation.value or "{}")
                except ValueError:
                    return False
                return (
                    (system_prompt.value or "") == default.system_prompt
                    and (user_template.value or "{query}") == default.user_template
                    and combination.combine == default.combine
                    and set_group_beam_search(params, False) == default.generation
                )

            def auto_preset():
                # Only fills an empty form: never overwrites edits
                if find_preset(model_name()) is not None and pristine():
                    apply_preset()

            if not existing:
                name_input.on("blur", auto_preset)
            model.on("blur", auto_preset)

            def use_model_settings():
                """Only the query is sent: the model's chat template and
                generation_config.json do the rest"""
                system_prompt.value = ""
                user_template.value = "{query}"
                generation.value = "{}"

            def collect() -> RewriterConfig:
                try:
                    params = json.loads(generation.value or "{}")
                except ValueError as e:
                    raise ValueError(f"Generation parameters: {e}") from None
                return RewriterConfig(
                    name=(name_input.value or "").strip(),
                    backend=backend.value,
                    model=(model.value or "").strip() or None,
                    url=(url.value or "").strip() or None,
                    system_prompt=system_prompt.value or "",
                    user_template=user_template.value or "{query}",
                    combine=combination.combine,
                    query_weight=combination.query_weight,
                    generation=params,
                    device=device.value or "auto",
                    dtype=dtype.value,
                )

            def run_check(rewriter: RewriterConfig) -> Check:
                ok, message = check(rewriter)
                return Check.ok(message) if ok else Check.error(message)

            async def do_check():
                try:
                    rewriter = collect()
                except ValueError as e:
                    status.show(Check.error(str(e)))
                    return
                await status.run(run_check, rewriter)

            def save():
                try:
                    rewriter = collect()
                    rewriter.validate()
                except ValueError as e:
                    status.show(Check.error(f"Cannot save: {e}"))
                    return
                if not existing and rewriter.name in self.workspace.rewriters:
                    status.show(Check.error(f"{rewriter.name!r} already exists"))
                    return
                self.workspace.put_rewriter(rewriter)
                self.dialog.close()
                self.on_saved(self.editing_collection_hint)

            def delete():
                self.workspace.remove_rewriter(name)
                self.dialog.close()
                self.on_saved(self.editing_collection_hint)

            if existing:
                ui.timer(0, do_check, once=True)
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                if existing:
                    ui.button("Remove", icon="delete", on_click=delete).props(
                        "flat color=negative"
                    )
                ui.space()
                ui.button("Preset", icon="auto_fix_high", on_click=apply_preset).props(
                    "flat"
                ).tooltip("Prompts and parameters of a known model").mark(
                    "rewriter-preset"
                )
                ui.button(
                    "Model's own", icon="settings_suggest", on_click=use_model_settings
                ).props("flat").tooltip(
                    "Send only the query: the model's chat template and "
                    "generation_config.json provide the prompt and decoding"
                ).mark("rewriter-model-settings")
                ui.button("Check", icon="fact_check", on_click=do_check).props(
                    "outline"
                )
                ui.button("Save", icon="save", on_click=save).mark("rewriter-save")

    def edit(self, name: str | None):
        self.editing = name
        self.editing_rewriter = self.editing_build = None
        self.refresh_list()
        existing = self.workspace.collections.get(name) if name else None
        c = (
            replace(
                existing, bm25=replace(existing.bm25), datasets=list(existing.datasets)
            )
            if existing
            else CollectionConfig(name="", index="")
        )
        self.form.clear()
        with self.form:
            ui.label("Edit collection" if existing else "New collection").classes(
                "font-medium"
            )
            name_input = (
                ui.input("Name", value=c.name, placeholder="e.g. msmarco-passage")
                .classes("w-full")
                .mark("settings-name")
            )
            if existing:
                name_input.props("readonly").tooltip(
                    "Saved queries refer to the name, so it cannot be changed"
                )
            name_check = CheckLabel("check-name")

            index_input = (
                ui.input(
                    "Index path",
                    value=c.index,
                    placeholder="impact-index BOW index folder",
                )
                .classes("w-full font-mono")
                .mark("settings-index")
            )
            index_check = CheckLabel("check-index")
            ui.label(
                f"Relative paths are resolved against {self.workspace.folder}"
            ).classes("text-xs text-grey-6")

            docstore_input = ui.input(
                "Document store path (optional)",
                value=c.docstore or "",
                placeholder="default: <index>/docstore or a sibling docstore folder",
            ).classes("w-full font-mono")
            documents_input = ui.input(
                "datamaestro documents (if no document store)",
                value=c.documents or "",
                placeholder="a datamaestro document store dataset id",
            ).classes("w-full font-mono")
            documents_check = CheckLabel("check-documents")

            ui.label(
                "datamaestro IR datasets (topics + assessments; one or more)"
            ).classes("text-sm font-medium mt-2")
            datasets = (
                ui.input_chips(value=list(c.datasets), new_value_mode="add-unique")
                .props("dense outlined")
                .classes("w-full")
                .mark("settings-datasets")
            )
            ui.label(
                "Type a dataset id (e.g. com.microsoft.msmarco.passage.dev) "
                "and press Enter"
            ).classes("text-xs text-grey-6 -mt-2")
            dataset_checks_column = ui.column().classes("w-full gap-1")
            dataset_checks: dict[str, CheckLabel] = {}

            with ui.expansion("Advanced").classes("w-full").props("dense"):
                with ui.row().classes("gap-2"):
                    k1 = ui.number("Default BM25 k1", value=c.bm25.k1, step=0.1).props(
                        "dense"
                    )
                    b = ui.number("Default BM25 b", value=c.bm25.b, step=0.05).props(
                        "dense"
                    )
                    variant = ui.select(
                        ["bm25", "lucene"], value=c.bm25.variant, label="IDF variant"
                    ).props("dense")
                with ui.row().classes("gap-2"):
                    content_format = ui.select(
                        ["json", "text"], value=c.content_format, label="Content format"
                    ).props("dense")
                    id_key = (
                        ui.input("Id key", value=c.id_key or "")
                        .props("dense")
                        .tooltip("Empty: ids are document numbers")
                    )
                    text_fields = ui.input(
                        "JSON text fields", value=",".join(c.text_fields)
                    ).props("dense")
                    title_field = ui.input(
                        "Title field", value=c.title_field or ""
                    ).props("dense")

            status = ui.label().classes("text-sm").mark("settings-status")

            def collect() -> CollectionConfig:
                return CollectionConfig(
                    name=(name_input.value or "").strip(),
                    index=(index_input.value or "").strip(),
                    docstore=(docstore_input.value or "").strip() or None,
                    documents=(documents_input.value or "").strip() or None,
                    datasets=[d.strip() for d in datasets.value or [] if d.strip()],
                    id_key=id_key.value or None,
                    content_format=content_format.value,
                    text_fields=[
                        f.strip()
                        for f in (text_fields.value or "").split(",")
                        if f.strip()
                    ]
                    or ["text"],
                    title_field=title_field.value or None,
                    bm25=BM25Params(
                        k1=float(k1.value),
                        b=float(b.value),
                        variant=variant.value,
                        k3=c.bm25.k3,
                    ),
                    in_memory=c.in_memory,
                    base=self.workspace.folder,
                )

            def run_name_check():
                value = (name_input.value or "").strip()
                taken = not existing and value in self.workspace.collections
                name_check.show(check_name(value, taken))

            async def run_documents_check():
                await documents_check.run(check_documents, collect())
                # Dataset compatibility depends on the documents
                await recheck_datasets()

            async def run_index_check():
                # The document check compares document counts with the index
                await index_check.run(check_index, collect())
                await run_documents_check()

            async def run_dataset_checks():
                ids = collect().datasets
                for dataset_id in list(dataset_checks):
                    if dataset_id not in ids:
                        dataset_checks.pop(dataset_id).row.parent_slot.parent.delete()
                pending = []
                for dataset_id in ids:
                    if dataset_id in dataset_checks:
                        continue
                    with dataset_checks_column:
                        with ui.column().classes("w-full gap-0"):
                            ui.label(dataset_id).classes("text-xs font-mono")
                            label = CheckLabel(f"check-dataset-{dataset_id}")
                            label.row.classes(remove="-mt-2")
                    dataset_checks[dataset_id] = label
                    pending.append((label, dataset_id))
                collection = collect()
                for label, dataset_id in pending:
                    await label.run(
                        check_dataset, dataset_id, self.services.topic_sets, collection
                    )

            name_input.on_value_change(run_name_check)
            for element in (index_input,):
                element.on("blur", run_index_check)
                element.on("keydown.enter", run_index_check)
            for element in (docstore_input, documents_input, id_key):
                element.on("blur", run_documents_check)
                element.on("keydown.enter", run_documents_check)
            datasets.on_value_change(run_dataset_checks)

            async def recheck_datasets():
                for dataset_id in list(dataset_checks):
                    dataset_checks.pop(dataset_id).row.parent_slot.parent.delete()
                await run_dataset_checks()

            async def check_all():
                if existing or name_input.value:
                    run_name_check()
                if index_input.value:
                    # Also checks the documents, then the datasets
                    await run_index_check()
                else:
                    await run_dataset_checks()

            # Check the current settings right away
            ui.timer(0, check_all, once=True)

            async def save():
                run_name_check()
                if name_check.check.status is Status.ERROR:
                    status.text = f"Cannot save: {name_check.check.message}"
                    status.classes(replace="text-sm text-negative")
                    return
                try:
                    collection = collect()
                    collection.validate()
                except ConfigError as e:
                    status.text = f"Cannot save: {e}"
                    status.classes(replace="text-sm text-negative")
                    return
                self.workspace.put(collection)
                checks = [index_check.check, documents_check.check] + [
                    label.check for label in dataset_checks.values()
                ]
                if any(ch is not None and ch.status is Status.ERROR for ch in checks):
                    ui.notify("Saved, but some settings have problems", type="warning")
                self.dialog.close()
                self.on_saved(collection.name)

            def delete():
                n = len(self.services.store.list(name))
                with ui.dialog() as confirm, ui.card():
                    ui.label(f"Remove collection {name!r} from the workspace?")
                    ui.label(
                        "The index files are not deleted."
                        + (
                            f" Its {n} saved queries are kept and come back if a "
                            "collection with the same name is added."
                            if n
                            else ""
                        )
                    ).classes("text-sm text-grey-8")
                    with ui.row().classes("w-full justify-end"):
                        ui.button("Cancel", on_click=confirm.close).props("flat")

                        def do_delete():
                            confirm.close()
                            self.workspace.remove(name)
                            self.dialog.close()
                            self.on_saved(None)

                        ui.button("Remove", on_click=do_delete).props("color=negative")
                confirm.open()

            async def recheck():
                for dataset_id in list(dataset_checks):
                    dataset_checks.pop(dataset_id).row.parent_slot.parent.delete()
                await check_all()

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                if existing:
                    ui.button("Remove", icon="delete", on_click=delete).props(
                        "flat color=negative"
                    )
                ui.space()
                ui.button("Re-check", icon="refresh", on_click=recheck).props(
                    "outline"
                ).tooltip("Checks are run when the dialog opens and fields change")
                ui.button("Save", icon="save", on_click=save).mark("settings-save")

    # --- Index builds

    def edit_build(self, name: str | None):
        self.editing, self.editing_rewriter, self.editing_build = None, None, name
        self.refresh_list()
        self.form.clear()
        with self.form:
            if name is None:
                self.new_build_form()
            else:
                BuildView(self, name)

    def new_build_form(self):
        builds = self.services.builds
        ui.label("New index build").classes("font-medium")
        ui.label(
            "Downloads datamaestro documents, stores them in a document store "
            "and builds a BOW index, then adds the collection to the "
            "workspace. The build runs in its own process: it goes on if the "
            "interface stops, and can be resumed after a failure."
        ).classes("text-xs text-grey-7")
        documents = (
            ui.select(
                {},
                label="datamaestro documents",
                with_input=True,
                new_value_mode="add-unique",
                clearable=True,
            )
            .props("dense input-debounce=0")
            .classes("w-full font-mono")
            .mark("build-documents")
            .tooltip(
                "Document collections of the installed datamaestro repositories "
                "(type to filter); any dataset id can be typed"
            )
        )
        catalog_status = ui.label("Loading the datamaestro catalog…").classes(
            "text-xs text-grey-6 -mt-2"
        )
        name = (
            ui.input("Name (also the collection's)", placeholder="e.g. msmarco-passage")
            .classes("w-full")
            .mark("build-name")
        )
        output = (
            ui.input("Output folder", placeholder="default: indexes/<name>")
            .classes("w-full font-mono")
            .mark("build-output")
        )
        ui.label(
            f"Relative paths are resolved against {self.workspace.folder}"
        ).classes("text-xs text-grey-6 -mt-2")
        with ui.row().classes("gap-2 items-center"):
            pipeline = ui.select(list(PIPELINES), value="pyserini", label="Pipeline")
            pipeline.props("dense").classes("w-40")
            stop_words = ui.select(
                list(STOP_WORDS), value="default", label="Stop words"
            )
            stop_words.props("dense").classes("w-40").tooltip(
                "default: the pipeline's own list (terrier-pisa filters queries only)"
            )
            positions = ui.checkbox("Positions").tooltip(
                "Stores token positions (needed for #1 and #uwN queries)"
            )
        datasets = (
            ui.select(
                [],
                label="datamaestro IR datasets (topics + assessments)",
                multiple=True,
                with_input=True,
                new_value_mode="add-unique",
                value=[],
            )
            .props("dense use-chips input-debounce=0")
            .classes("w-full")
            .mark("build-datasets")
            .tooltip("Those of the selected documents come first")
        )
        catalog = None
        # What was filled automatically (user edits are never overwritten)
        auto = {"name": "", "datasets": []}

        def on_documents(e):
            documents_id = (e.value or "").strip()
            if catalog is not None:
                datasets.set_options(
                    catalog.topic_options(documents_id), value=datasets.value
                )
                if list(datasets.value or []) == auto["datasets"]:
                    auto["datasets"] = catalog.topics_for(documents_id)
                    datasets.value = list(auto["datasets"])
            if (name.value or "") == auto["name"]:
                auto["name"] = suggested_name(documents_id)
                name.value = auto["name"]

        documents.on_value_change(on_documents)

        async def load():
            nonlocal catalog
            catalog = await run.io_bound(self.services.catalog)
            if catalog.error:
                catalog_status.text = f"{catalog.error}: type a dataset id"
                return
            documents.set_options(
                {c.id: c.label for c in catalog.collections.values()},
                value=documents.value,
            )
            datasets.set_options(
                catalog.topic_options(documents.value), value=datasets.value
            )
            with_topics = sum(1 for c in catalog.collections.values() if c.topics)
            catalog_status.text = (
                f"{len(catalog.collections)} document collections "
                f"({with_topics} with topics) in the installed datamaestro "
                "repositories"
            )

        ui.timer(0, load, once=True)
        status = ui.label().classes("text-sm").mark("build-status")

        def start():
            spec = BuildSpec(
                name=(name.value or "").strip(),
                documents=(documents.value or "").strip(),
                output=(output.value or "").strip(),
                pipeline=pipeline.value,
                stop_words=stop_words.value,
                positions=bool(positions.value),
                datasets=[d.strip() for d in datasets.value or [] if d.strip()],
            )
            try:
                builds.create(spec)
            except (ConfigError, OSError) as e:
                status.text = f"Cannot start: {e}"
                status.classes(replace="text-sm text-negative")
                return
            self.edit_build(spec.name)

        with ui.row().classes("w-full justify-end mt-2"):
            ui.button("Start", icon="play_arrow", on_click=start).mark("build-start")


DOCUMENTS_SUFFIXES = (".documents", ".collection", ".docs", ".corpus")


def suggested_name(documents_id: str) -> str:
    """A collection name for a documents dataset, e.g. msmarco-passage for
    com.microsoft.msmarco.passage.documents"""
    for suffix in DOCUMENTS_SUFFIXES:
        documents_id = documents_id.removesuffix(suffix)
    return "-".join(documents_id.split(".")[-2:])


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def stage_progress(state: BuildState, stage: str, running: bool) -> str:
    current = state.stages[stage]
    if current.status == "pending" and current.done == 0:
        return ""
    parts = []
    if current.total:
        parts.append(f"{current.done:,} / {current.total:,} documents")
    elif current.done:
        parts.append(f"{current.done:,} documents")
    rate = current.rate()
    if rate:
        parts.append(f"{rate:,.0f}/s")
    if running and current.status == "running":
        if current.total and current.done >= current.total and stage == "index":
            parts.append("writing the index")
        elif rate and current.total:
            parts.append(f"{duration((current.total - current.done) / rate)} left")
    elif current.started and current.finished:
        parts.append(f"in {duration(current.finished - current.started)}")
    return " · ".join(parts)


class BuildView:
    """State of a build, refreshed while it is displayed"""

    def __init__(self, dialog: SettingsDialog, name: str):
        self.dialog = dialog
        self.builds = dialog.services.builds
        self.name = name
        self.last: tuple | None = None
        with ui.row().classes("w-full items-center"):
            ui.label(f"Index build {name}").classes("font-medium")
            ui.space()
            self.status = ui.label().classes("text-sm").mark("build-state")
        self.summary = ui.label().classes("text-xs text-grey-7 font-mono break-all")
        self.stages = ui.column().classes("w-full gap-1")
        self.error = ui.column().classes("w-full gap-1")
        with ui.expansion("Log").classes("w-full").props("dense"):
            self.log = (
                ui.code("")
                .classes("w-full text-xs max-h-64 overflow-auto")
                .mark("build-log")
            )
        self.buttons = ui.row().classes("w-full justify-end gap-2 mt-2")
        self.refresh()
        ui.timer(1.0, self.refresh)

    def refresh(self):
        try:
            state, alive = self.builds.get(self.name)
        except (OSError, ValueError, TypeError):
            self.status.text = "Unknown build (removed?)"
            return
        self.log.content = self.builds.files(self.name).log_tail()
        key = (json.dumps(state.to_dict(), sort_keys=True), alive)
        if key == self.last:
            return
        previous_status = None
        if self.last is not None:
            previous_status = BuildState.from_dict(json.loads(self.last[0])).status(
                self.last[1]
            )
        self.last = key
        status = state.status(alive)
        if previous_status is not None and previous_status != status:
            self.dialog.refresh_list()
        self.render(state, alive, status)

    def render(self, state: BuildState, alive: bool, status: str):
        icon, color = BUILD_STATUS_STYLE[status]
        self.status.text = status
        self.status.classes(replace=f"text-sm text-{color}")
        spec = state.spec
        output = spec.output_path(self.dialog.workspace.folder)
        self.summary.text = (
            f"{spec.documents} → {output} · {spec.pipeline}, stop words "
            f"{spec.stop_words}{', positions' if spec.positions else ''}"
        )
        self.stages.clear()
        with self.stages:
            for stage in STAGES:
                current = state.stages[stage]
                stage_status = current.status
                if stage_status == "running" and not alive:
                    stage_status = "interrupted"
                stage_icon, stage_color = BUILD_STATUS_STYLE.get(
                    stage_status, BUILD_STATUS_STYLE["pending"]
                )
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    if stage_status == "running":
                        ui.spinner(size="xs")
                    else:
                        ui.icon(stage_icon, size="xs").props(f"color={stage_color}")
                    ui.label(STAGE_LABELS[stage]).classes("text-sm w-36")
                    ui.label(stage_progress(state, stage, alive)).classes(
                        "text-xs text-grey-7"
                    ).mark(f"build-stage-{stage}")
                if current.total and stage_status in ("running", "interrupted"):
                    ui.linear_progress(
                        value=min(1.0, current.done / current.total),
                        show_value=False,
                    ).props("rounded")
        self.error.clear()
        if state.error and not alive:
            with self.error:
                ui.label(state.error).classes("text-sm text-negative break-all").mark(
                    "build-error"
                )
                if state.traceback:
                    with ui.expansion("Traceback").classes("w-full").props("dense"):
                        ui.code(state.traceback).classes("w-full text-xs")
        if status == "interrupted":
            with self.error:
                ui.label(
                    "The build process stopped (interface or machine restart, "
                    "killed process): resume it to go on where it stopped."
                ).classes("text-sm text-warning")
        if state.registered:
            with self.error:
                ui.label(f"Collection {spec.name!r} added to the workspace").classes(
                    "text-sm text-positive"
                )
        self.render_buttons(state, status)

    def render_buttons(self, state: BuildState, status: str):
        self.buttons.clear()
        with self.buttons:
            if status != "running":
                ui.button("Remove", icon="delete", on_click=self.remove).props(
                    "flat color=negative"
                ).tooltip("Forgets the build; its index and document store are kept")
            ui.space()
            if status == "running":
                ui.button("Cancel", icon="stop", on_click=self.cancel).props(
                    "outline color=negative"
                ).mark("build-cancel")
            else:
                if status != "pending":
                    ui.button(
                        "Restart from scratch", icon="restart_alt", on_click=self.reset
                    ).props("flat").mark("build-reset")
                if status != "done":
                    ui.button(
                        "Resume" if status != "pending" else "Start",
                        icon="play_arrow",
                        on_click=self.resume,
                    ).mark("build-resume").tooltip(
                        "Completed stages are skipped; the document store "
                        "resumes from its last checkpoint"
                    )
                elif state.registered:
                    ui.button(
                        "Open collection",
                        icon="open_in_new",
                        on_click=lambda: self.dialog.on_saved(state.spec.name),
                    ).mark("build-open")

    def act(self, action, message: str | None = None):
        try:
            action(self.name)
        except (ConfigError, OSError) as e:
            ui.notify(str(e), type="negative")
            return
        if message:
            ui.notify(message)
        self.last = None
        # Let the process start before refreshing
        ui.timer(0.5, self.refresh, once=True)

    def resume(self):
        self.act(self.builds.start)

    def cancel(self):
        self.act(self.builds.cancel, "Cancelling…")

    def reset(self):
        with ui.dialog() as confirm, ui.card():
            ui.label(f"Restart build {self.name!r} from scratch?")
            ui.label("Its document store and index are deleted and rebuilt.").classes(
                "text-sm text-grey-8"
            )
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=confirm.close).props("flat")

                def do_reset():
                    confirm.close()
                    self.act(self.builds.reset)
                    self.act(self.builds.start)

                ui.button("Restart", on_click=do_reset).props("color=negative")
        confirm.open()

    def remove(self):
        try:
            self.builds.remove(self.name)
        except (ConfigError, OSError) as e:
            ui.notify(str(e), type="negative")
            return
        self.dialog.edit_build(None)
