# TODO

## Nested positional operators (needs impact-index)

Support queries such as

    #uw10(#1(salvation army) #syn(founded established began))

impact-index rejects them today: `#1`, `#uwN` and `#syn` only take plain
words (`QueryNode::Phrase`/`Window`/`Syn` hold `Vec<TermIndex>`, and the
matchop parser raises "operator '#1' not allowed inside uw10"). The
explorer's parser (`query_syntax.py`) mirrors that restriction and must be
updated alongside.

What has to change (see `positions-plan.md` §3.2–3.3 in impact-index):

- [ ] **AST**: let `Phrase`/`Window`/`Syn` hold child nodes instead of
  term ids, restricted to *positional* children: terms, `#syn` (union of
  the children's positions) and `#1` (a phrase match is a span). `#band`
  and `#combine` have no positions and stay forbidden inside them.
- [ ] **Positional cursors**: today `PhraseCursor` merges the position
  lists of raw term cursors. Nesting needs a common interface that yields,
  per document, a sorted list of *spans* `(start, end)`:
  - term: `(p, p + 1)` for each position;
  - `#syn`: merge of the children's spans (and tf = number of spans, which
    keeps the current "sum of tfs" semantics);
  - `#1`: children's spans chained with adjacency
    (`next.start == previous.end`), yielding the covering span;
  - `#uwN`: minimal windows covering one span of every child, with
    `end - start <= N`.
- [ ] **Semantics to decide** (check against Terrier/Indri, which allow
  nesting in `#uwN`/`#odN`):
  - window width counted in tokens over the covering span (so a 2-token
    phrase uses 2 of the N tokens)?
  - overlapping spans (a word matching two children) — allowed or not?
  - virtual tf of `#uwN` over nested children (today: occurrences of the
    rarest term) and df (today: N / 100 heuristic).
- [ ] **Bounds**: max tf of a nested positional node ≤ min over children of
  their max tf (still safe for WAND/MaxScore).
- [ ] **Parser**: accept the nested forms in `parse_matchop_with`, keeping
  the stop word / unknown term rules (an unknown word inside a positional
  child drops the whole positional operator).
- [ ] **Python**: nested dict form (`{"window": {"children": [...],
  "width": 10}}`), stubs regenerated.
- [ ] **Tests**: brute-force reference over a synthetic positional corpus,
  pruning-safety identity tests, stop word gaps (`position_gaps`).
- [ ] **Explorer**: relax `query_syntax.py`, update the help (`HELP`) and
  the feedback tests.

Workarounds until then:

- `#band(#1(salvation army) #syn(founded established began))` (same
  document, no window);
- `#combine(#1(salvation army) #uw10(salvation army founded)
  #uw10(salvation army established) #uw10(salvation army began))`
  (one window per synonym).

## Rewriters

- [ ] "Evaluate all" with a rewriter (rewrite every topic, cache rewrites
  in the workspace, compare with the original topics).
- [ ] Per-keyword weights in the searched query (e.g. original terms
  weighted higher than expansion terms, as `#combine:0=W(...)`).
