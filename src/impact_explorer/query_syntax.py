"""Matchop query syntax: parsing with positions, for user feedback.

This mirrors the grammar of impact-index's own parser (``src/query.rs``),
which remains the one used for searching. The mirror keeps character
offsets, so that the interface can point at errors, and records how each
word resolves (term, stop word, unknown) and which operators are dropped.

Grammar::

    query    := node*
    node     := WORD | operator
    operator := '#combine' (':' IDX '=' WEIGHT)* '(' node* ')'
              | '#band' '(' node* ')'
              | '#syn' '(' WORD* ')'
              | '#1' '(' WORD* ')'
              | '#uw' N '(' WORD* ')'          (N >= 2)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

TOKEN_RE = re.compile(r"[()]|[^\s()]+")
SUBWORD_RE = re.compile(r"\w+(?:'\w+)*")

POSITIONAL = ("1", "uw")


def is_structured(text: str) -> bool:
    """A query is structured as soon as it uses an operator"""
    return any(tok.startswith("#") for tok in TOKEN_RE.findall(text))


class Resolution(StrEnum):
    TERM = "term"
    STOPWORD = "stop word"
    UNKNOWN = "unknown"


@dataclass
class Token:
    text: str
    start: int
    end: int


class QuerySyntaxError(ValueError):
    def __init__(self, message: str, start: int, end: int):
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end


@dataclass
class Word:
    text: str
    start: int
    end: int
    resolution: Resolution
    term_id: int | None = None


@dataclass
class Operator:
    name: str
    """combine, band, syn, 1 or uwN"""

    start: int
    end: int
    children: list["Word | Operator"] = field(default_factory=list)
    weights: dict[int, float] = field(default_factory=dict)
    width: int | None = None
    dropped: str | None = None
    """Why the operator can never match (None if it can)"""

    @property
    def label(self) -> str:
        if self.name == "combine" and self.weights:
            spec = "".join(f":{i}={w:g}" for i, w in sorted(self.weights.items()))
            return f"#combine{spec}"
        return f"#{self.name}"


@dataclass
class ParsedQuery:
    text: str
    nodes: list[Word | Operator]
    warnings: list[str]

    def words(self) -> list[Word]:
        result = []

        def visit(node):
            if isinstance(node, Word):
                result.append(node)
            else:
                for child in node.children:
                    visit(child)

        for node in self.nodes:
            visit(node)
        return result

    def term_ids(self) -> list[int]:
        """Resolved terms, in query order (no duplicates)"""
        return list(
            dict.fromkeys(w.term_id for w in self.words() if w.term_id is not None)
        )

    def uses_positions(self) -> bool:
        def visit(node) -> bool:
            if isinstance(node, Word):
                return False
            if node.name == "1" or node.name.startswith("uw"):
                return True
            return any(visit(c) for c in node.children)

        return any(visit(n) for n in self.nodes)


Resolver = Callable[[str], tuple[Resolution, int | None]]


class _Parser:
    def __init__(self, text: str, resolve: Resolver):
        self.text = text
        self.tokens = [
            Token(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)
        ]
        self.pos = 0
        self.resolve = resolve
        self.warnings: list[str] = []

    def peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Token | None:
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def end_of_input(self) -> int:
        return len(self.text)

    def expect(self, expected: str, context: Token):
        tok = self.next()
        if tok is None:
            raise QuerySyntaxError(
                f"expected '{expected}', found end of input"
                f" (to close {context.text} opened here)"
                if expected == ")"
                else f"expected '{expected}', found end of input",
                context.start if expected == ")" else self.end_of_input(),
                context.end if expected == ")" else self.end_of_input(),
            )
        if tok.text != expected:
            raise QuerySyntaxError(
                f"expected '{expected}', found '{tok.text}'", tok.start, tok.end
            )
        return tok

    def at_close(self) -> bool:
        tok = self.peek()
        return tok is None or tok.text == ")"

    def parse(self) -> ParsedQuery:
        nodes = []
        while not self.at_close():
            nodes.append(self.node())
        tok = self.peek()
        if tok is not None:
            raise QuerySyntaxError(
                f"unexpected trailing token '{tok.text}' (unbalanced parenthesis)",
                tok.start,
                tok.end,
            )
        return ParsedQuery(self.text, nodes, self.warnings)

    def word(self, tok: Token) -> Word:
        resolution, term_id = self.resolve(tok.text)
        return Word(tok.text, tok.start, tok.end, resolution, term_id)

    def node(self) -> Word | Operator:
        tok = self.next()
        if not tok.text.startswith("#"):
            return self.word(tok)
        return self.operator(tok)

    def operator(self, tok: Token) -> Operator:
        name, *specs = tok.text[1:].split(":")
        if name == "combine":
            op = Operator("combine", tok.start, tok.end)
            for spec in specs:
                if "=" not in spec:
                    raise QuerySyntaxError(
                        f"malformed combine weight spec '{spec}' in '{tok.text}'"
                        " (expected :INDEX=WEIGHT)",
                        tok.start,
                        tok.end,
                    )
                idx, weight = spec.split("=", 1)
                try:
                    index = int(idx)
                    if index < 0:
                        raise ValueError
                except ValueError:
                    raise QuerySyntaxError(
                        f"bad combine index '{idx}' in '{tok.text}'", tok.start, tok.end
                    ) from None
                try:
                    op.weights[index] = float(weight)
                except ValueError:
                    raise QuerySyntaxError(
                        f"bad combine weight '{weight}' in '{tok.text}'",
                        tok.start,
                        tok.end,
                    ) from None
            self.expect("(", tok)
            while not self.at_close():
                op.children.append(self.node())
            op.end = self.expect(")", tok).end
            for index in op.weights:
                if index >= len(op.children):
                    self.warnings.append(
                        f"{tok.text}: weight index {index} has no matching child "
                        f"(children are numbered from 0 to {len(op.children) - 1})"
                    )
            if not any(self.matchable(c) for c in op.children):
                op.dropped = "no child can match"
            return op

        if specs:
            # impact-index only reads weights for #combine
            self.warnings.append(f"{tok.text}: the ':…' suffix is ignored")

        if name == "band":
            op = Operator("band", tok.start, tok.end)
            self.expect("(", tok)
            while not self.at_close():
                op.children.append(self.node())
            op.end = self.expect(")", tok).end
            failing = [c for c in op.children if self.fails(c)]
            if failing:
                op.dropped = "a child can never match: " + ", ".join(
                    self.describe(c) for c in failing
                )
            elif not any(self.matchable(c) for c in op.children):
                op.dropped = "no child left"
            return op

        if name == "syn" or name == "1" or self._is_window(name):
            op = Operator(name, tok.start, tok.end)
            if name.startswith("uw"):
                op.width = int(name[2:])
                if op.width < 2:
                    raise QuerySyntaxError(
                        f"window width must be >= 2, got {op.width} (in '{tok.text}')",
                        tok.start,
                        tok.end,
                    )
            self.expect("(", tok)
            while not self.at_close():
                child = self.next()
                if child.text.startswith("#"):
                    raise QuerySyntaxError(
                        f"operator '{child.text}' not allowed inside #{name}"
                        " (only words are allowed)",
                        child.start,
                        child.end,
                    )
                op.children.append(self.word(child))
            op.end = self.expect(")", tok).end
            terms = [c for c in op.children if c.resolution is Resolution.TERM]
            unknown = [c for c in op.children if c.resolution is Resolution.UNKNOWN]
            if name != "syn" and unknown:
                op.dropped = "unknown word(s): " + ", ".join(c.text for c in unknown)
            elif not terms:
                op.dropped = "no indexed word"
            return op

        raise QuerySyntaxError(
            f"unknown operator '{tok.text}' (use #combine, #syn, #band, #1 or #uwN)",
            tok.start,
            tok.end,
        )

    @staticmethod
    def _is_window(name: str) -> bool:
        return len(name) > 2 and name.startswith("uw") and name[2:].isdigit()

    @staticmethod
    def matchable(node) -> bool:
        if isinstance(node, Word):
            return node.resolution is Resolution.TERM
        return node.dropped is None

    @staticmethod
    def fails(node) -> bool:
        if isinstance(node, Word):
            return node.resolution is Resolution.UNKNOWN
        return node.dropped is not None

    @staticmethod
    def describe(node) -> str:
        return node.text if isinstance(node, Word) else node.label


def parse(text: str, resolve: Resolver) -> ParsedQuery:
    """Parses a matchop query (raises QuerySyntaxError)"""
    return _Parser(text, resolve).parse()


def load_stop_words(index_folder: Path) -> set[str]:
    """Stop words of a BOW index (best effort: empty if unavailable)"""
    try:
        import cbor2

        with (Path(index_folder) / "analyzer.cbor").open("rb") as fp:
            config = cbor2.load(fp)
        words = set(config.get("stop_words_list") or [])
        words.update(config.get("query_stop_words_list") or [])
        return {w.lower() for w in words}
    except Exception:
        return set()


def make_resolver(
    term_lookup: Callable[[str], tuple[int, ...]],
    stop_words: set[str],
    stop_lookup: Callable[[str], tuple[int, ...]] | None = None,
):
    """Resolves a query token like impact-index does

    The first indexed sub-word wins; a token whose sub-words are all stop
    words is skipped, otherwise it is unknown. With ``stop_lookup``, stop
    words are kept and resolved with it (an analyzer without stop list).
    """

    def resolve(token: str) -> tuple[Resolution, int | None]:
        subwords = SUBWORD_RE.findall(token)
        if not subwords:
            return Resolution.STOPWORD, None
        unknown = False
        for sub in subwords:
            terms = term_lookup(sub)
            if terms:
                return Resolution.TERM, terms[0]
            if sub.lower() not in stop_words:
                unknown = True
            elif stop_lookup is not None:
                terms = stop_lookup(sub)
                if terms:
                    return Resolution.TERM, terms[0]
                unknown = True
        return (Resolution.UNKNOWN if unknown else Resolution.STOPWORD), None

    return resolve


def to_query_tree(parsed: ParsedQuery) -> dict | int | None:
    """impact-index's nested query form, with the words resolved here

    Applies impact-index's rules: stop words are skipped, an unknown word is
    dropped from ``#combine``/``#syn`` but makes ``#band``/``#1``/``#uwN``
    unmatchable. Returns None if nothing can match.
    """

    def term_ids(op: Operator) -> list[int]:
        return [c.term_id for c in op.children if c.resolution is Resolution.TERM]

    def convert(node):
        if isinstance(node, Word):
            return {"term": node.term_id} if node.term_id is not None else None
        if node.dropped:
            return None
        if node.name == "combine":
            children = [
                [node.weights.get(i, 1.0), tree]
                for i, child in enumerate(node.children)
                if (tree := convert(child)) is not None
            ]
            return {"combine": children} if children else None
        if node.name == "syn":
            return {"syn": term_ids(node)}
        if node.name == "band":
            children = [tree for c in node.children if (tree := convert(c)) is not None]
            return {"band": children}
        if node.name == "1":
            return {"phrase": term_ids(node)}
        return {"window": {"terms": term_ids(node), "width": node.width}}

    trees = [tree for node in parsed.nodes if (tree := convert(node)) is not None]
    if not trees:
        return None
    if len(trees) == 1:
        return trees[0]
    return {"combine": [[1.0, tree] for tree in trees]}


HELP = """
Plain text is analyzed like documents (tokenizer, stop words, stemmer) and
scored with BM25. As soon as the query uses an operator, it is parsed as
a **structured query** (Terrier matchop syntax):

| Operator | Meaning | Example |
|---|---|---|
| `#combine(…)` | weighted sum of its children (the default) | `#combine(quick fox)` |
| `#combine:I=W(…)` | child `I` (from 0) gets weight `W` | `#combine:0=2(fox dog)` |
| `#syn(w1 w2 …)` | synonyms, scored as one term | `#syn(car automobile)` |
| `#band(…)` | documents containing *every* child | `#band(fox #1(lazy dog))` |
| `#1(w1 w2 …)` | exact phrase (adjacent words) | `#1(new york)` |
| `#uwN(w1 w2 …)` | all words within a window of `N` tokens | `#uw8(fox dog)` |

* Words outside an operator are combined: `fox #1(lazy dog)` is
  `#combine(fox #1(lazy dog))`.
* `#syn`, `#1` and `#uwN` only take words; `#combine` and `#band` take
  words or operators.
* Stop words are removed everywhere (`#1(bank of america)` is
  `#1(bank america)`).
* A word not in the index is dropped from `#combine`/`#syn`, but makes the
  whole `#1`, `#uwN` or `#band` unmatchable.
* `#1` and `#uwN` need an index built with positions.
* Phrases and windows are scored as one virtual term (with a heuristic
  document frequency), as Terrier does.
"""
