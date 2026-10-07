"""Workspace settings dialog: collections, index paths and datasets."""

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from nicegui import run, ui

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
                self.form = ui.column().classes("grow gap-2")
        self.editing_rewriter: str | None = None
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

        self.editing, self.editing_rewriter = None, name
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
        self.editing_rewriter = None
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
