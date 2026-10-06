# impact-explorer

A web interface (NiceGUI) to explore search results of an
[impact-index](https://github.com/xpmir/impact-index) BOW index (BM25)
against [datamaestro-ir](https://github.com/xpmir/datamaestro_ir)
collections:

- pick topics from datamaestro IR (Adhoc) datasets, or type any query; for
  topics with several fields (TREC: title, description, narrative), choose
  the field used as query and look at all of them;
- see the top-K documents with matching terms highlighted, each marked
  relevant / non-relevant / unjudged (filterable), plus nDCG@K, RR@K, AP,
  P@K, R@K; query terms are shown in their index (stemmed) form with their
  document frequency;
- edit a topic's query and compare with the original: metric differences,
  and the original rank of every document ("was #3", "new");
- switch the index's stop list on or off for a query ("Remove stop
  words"): stop words only match if the index kept them (e.g. built with
  `--pipeline terrier-pisa` or `--stop-words none`);
- rewrite queries with language models (e.g. the STORM query expander
  `Arthur-75/storm-qwen3-8B`) registered in the settings, and compare the
  rewritten query with the original;
- see and select the saved versions of a topic;
- evaluate every topic of a dataset at once, and sort topics by any metric
  (worst first); results are stored in the workspace, and saved
  reformulations of the topics are evaluated alongside;
- list the relevant documents that were *not* retrieved in the top-K, with
  their rank within the search depth when retrieved deeper;
- save queries, each linked to a collection (and optionally to its topic);
- see what an index is made of (ⓘ next to the collection): kind and
  codecs, positions, value type, size, and text analysis — the build
  pipeline (`pyserini`, `terrier`, `terrier-pisa`) is recognized from the
  analyzer, with what was overridden (e.g. the stemmer);
- write structured queries (`#combine`, `#syn`, `#band`, `#1`, `#uwN`) with
  a live parser that points at syntax errors and shows how every word
  resolves (indexed term, stop word, unknown) and which operators are
  dropped. The help button (?) next to the query documents the syntax.

## Install

```bash
uv sync
uv sync --extra rewriters    # local query rewriters (torch, transformers)
uv run pre-commit install    # ruff + conventional commit messages
```

## Usage

Everything lives in a **workspace folder**:

```
<workspace>/
    workspace.json        collections: index path, documents, datamaestro datasets
    saved-queries.json    saved queries
    evaluations/          stored "Evaluate all" results
```

The settings dialog checks everything as it is edited: index, documents
(and that they match the index), and each dataset (topics, assessments,
and whether its assessed documents belong to the collection). Checks never
download anything.

Both files are edited from the interface (gear icon): add a collection by
giving its index path, then add the datamaestro IR datasets that provide
topics and assessments. A dataset id typed in the topic panel is also
added to the collection.

```bash
# Index a datamaestro collection (BOW index + document store) and register
# it in a workspace; --positions enables #1 and #uwN queries, --stop-words
# lucene|terrier|none|<file> sets the stop list (default: the pipeline's)
uv run impact-explorer index com.microsoft.msmarco.passage.documents \
    data/msmarco-passage --positions \
    --workspace ws --dataset com.microsoft.msmarco.passage.dev.small

# Start the interface
uv run impact-explorer serve ws --port 8080
```

### Rewriters

Rewriters are registered in the settings (gear icon, "Rewriters"). A
rewriter has a prompt (system prompt and user template with `{query}`),
generation parameters, and how the searched query is built from the
outputs:

- *Generated only* (`{outputs}`): the raw outputs, concatenated — STORM's
  setting (it is trained with its outputs alone as the BM25 query, and
  repeats the query terms itself);
- *Original + generated* (`{query} {outputs}`);
- *Original + unique keywords* (`{query} {keywords}`): each keyword once;
- a custom template with `{query}`, `{keywords}` and `{outputs}`.

Repeated words add up in BM25 (as in Lucene), so repetitions in the outputs
weigh more; the *original query weight* repeats `{query}`. The rewrite
panel can change both without generating again. Naming a rewriter after a
known model (e.g. `Arthur-75/storm-qwen3-8B`) fills in its preset.

- `transformers` backend: the model runs locally (cuda, mps or cpu); it is
  downloaded and loaded the first time it is used.
- `openai` backend: any OpenAI-compatible server, e.g.
  `vllm serve Arthur-75/storm-qwen3-8B`, with its URL
  (`http://host:8000/v1`).

Group beam search (`num_beam_groups`, used by STORM) is no longer part of
transformers: it runs code from the Hub
(`transformers-community/group-beam-search`), which you have to allow
explicitly ("Allow group beam search" in the rewriter's settings) — or use
plain beam search instead.

A saved query can be opened directly with `/?saved=<id>`, and a collection
with `/?collection=<name>`.

### Collections

A collection needs an impact-index BOW index (built with
`BOWIndexBuilder`, so that it has an analyzer) and a source of document
texts:

- a datamaestro-ir document store dataset, indexed in its iteration order
  (e.g. `com.microsoft.msmarco.passage.documents`), or
- an impact-index `DocumentStore` whose document numbers are the index
  docids (default: a `docstore` folder inside or next to the index), with
  JSON (`{"title": …, "text": …}`) or plain-text contents; a store without
  keys uses document numbers as ids.

Index docids are mapped to external ids through the document store; qrels
use external ids.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
```

Notes:

- The structured query parser (`query_syntax.py`) mirrors impact-index's
  own (`src/query.rs`) to give positioned feedback; searching always goes
  through impact-index, whose errors are reported as is.
- impact-index does not expose its vocabulary: index forms of query terms
  are recovered with the index's stemmer and checked against the analyzer;
  forms that cannot be checked (Porter is not idempotent) are shown in
  italics.
