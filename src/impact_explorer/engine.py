"""Search over an impact-index BOW index with query-time BM25 scoring."""

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import BM25Params, CollectionConfig, Workspace
from .documents import DocumentSource, document_source
from .query_syntax import (
    ParsedQuery,
    QuerySyntaxError,
    Word,
    is_structured,
    load_stop_words,
    make_resolver,
    parse,
    to_query_tree,
)

WORD_RE = re.compile(r"\w+(?:'\w+)*")
"""Words as seen by the highlighter (apostrophes kept so that the analyzer
applies its own possessive/contraction rule)"""


@dataclass
class QueryTerm:
    term_id: int
    weight: float
    words: list[str]
    """Query words mapped to this term"""

    df: int
    """Number of documents containing the term"""

    stem: str | None = None
    """Index form of the term (e.g. stemmed), when it could be recovered"""

    stem_verified: bool = False
    """Whether the analyzer maps ``stem`` back to this term"""

    @property
    def label(self) -> str:
        return self.stem or self.words[0]


@dataclass
class AnalyzedQuery:
    text: str
    terms: list[QueryTerm]
    parsed: ParsedQuery | None = None
    """Parse tree of a structured (matchop) query"""

    @property
    def vector(self) -> dict[int, float]:
        return {t.term_id: t.weight for t in self.terms}


@dataclass
class Hit:
    rank: int
    docid: str
    score: float


@dataclass
class SearchResult:
    query: AnalyzedQuery
    hits: list[Hit]
    """All hits up to the search depth"""

    ranks: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.ranks = {hit.docid: hit.rank for hit in self.hits}

    def top(self, k: int) -> list[Hit]:
        return self.hits[:k]


class TermMatcher:
    """Maps document words to index term ids using the index's analyzer

    Each distinct word is analyzed once (and cached), which applies exactly
    the same pipeline (tokenizer, stop words, stemmer) as query analysis.
    """

    def __init__(self, analyzer, cache_size: int = 200_000):
        self.analyzer = analyzer
        self.cache: dict[str, tuple[int, ...]] = {}
        self.cache_size = cache_size
        self.lock = threading.Lock()

    def terms(self, word: str) -> tuple[int, ...]:
        key = word.lower()
        try:
            return self.cache[key]
        except KeyError:
            pass
        terms = tuple(self.analyzer.analyze_query(key).keys())
        with self.lock:
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[key] = terms
        return terms


def index_features(folder: Path) -> list[str]:
    """Features recorded in an index manifest (e.g. ``positions``)"""
    try:
        manifest = json.loads((Path(folder) / "manifest.json").read_text())
    except (OSError, ValueError):
        return []
    return list(manifest.get("features", []))


def make_stemmer(folder: Path):
    """The stemmer of a BOW index (from its analyzer settings), or None"""
    try:
        import cbor2
        import snowballstemmer

        with (Path(folder) / "analyzer.cbor").open("rb") as fp:
            config = cbor2.load(fp)
        name = config.get("stemmer")
        if not name:
            return lambda word: word
        algorithm = "porter" if name == "porter" else config.get("language", "english")
        return snowballstemmer.stemmer(algorithm).stemWord
    except Exception:
        return None


class KeepStopWordsMatcher:
    """Word -> terms with the index analyzer, then without stop list for
    stop words (so that highlighting follows the query)"""

    def __init__(self, engine: "SearchEngine"):
        self.engine = engine

    def terms(self, word: str) -> tuple[int, ...]:
        terms = self.engine.matcher.terms(word)
        if terms or word.lower() not in self.engine.stop_words:
            return terms
        return self.engine.stop_matcher.terms(word)


class SearchEngine:
    def __init__(self, config: CollectionConfig, documents: DocumentSource):
        import impact_index

        self.config = config
        self.documents = documents
        self.index = impact_index.Index.load(str(config.index_path), config.in_memory)
        self.analyzer = self.index.analyzer()
        self.matcher = TermMatcher(self.analyzer)
        self.has_positions = "positions" in index_features(config.index_path)
        self.stemmer = make_stemmer(config.index_path)
        self._stems: dict[int, tuple[str | None, bool]] = {}
        self.stop_words = load_stop_words(config.index_path)
        self.resolve = make_resolver(self.matcher.terms, self.stop_words)
        self._stop_matcher: TermMatcher | None = None
        self._keep_matcher: KeepStopWordsMatcher | None = None
        self._scored: dict[tuple, object] = {}
        self._lock = threading.Lock()

    @staticmethod
    def open(config: CollectionConfig) -> "SearchEngine":
        return SearchEngine(config, document_source(config))

    def scored(self, params: BM25Params):
        import impact_index

        key = (params.k1, params.b, params.variant, params.k3)
        with self._lock:
            if key not in self._scored:
                self._scored[key] = self.index.with_scoring(
                    impact_index.BM25Scoring(
                        k1=params.k1, b=params.b, variant=params.variant, k3=params.k3
                    )
                )
            return self._scored[key]

    def stem(self, term_id: int, words: list[str]) -> tuple[str | None, bool]:
        """Index form of a term, recovered by stemming one of its words

        impact-index does not expose its vocabulary, so the index's stemmer
        is re-applied. Returns the form and whether it is verified, i.e. the
        analyzer maps it back to the same term; stemmers are not idempotent
        (Porter: university -> univers -> univ), so an unverified form is
        still the most likely one.
        """
        if term_id in self._stems:
            return self._stems[term_id]
        found: tuple[str | None, bool] = (None, False)
        # Stop words too (they are searched when the stop list is disabled)
        matcher = self.matcher_for(True)
        if self.stemmer is not None:
            for word in words:
                base = re.sub(r"'s$", "", word.lower())
                if term_id not in matcher.terms(base):
                    continue
                candidate = self.stemmer(base)
                verified = matcher.terms(candidate) == (term_id,)
                if verified:
                    found = (candidate, True)
                    break
                if found[0] is None:
                    found = (candidate, False)
        self._stems[term_id] = found
        return found

    @property
    def stop_matcher(self) -> "TermMatcher":
        """Maps words to terms without any stop list (for stop words)"""
        if self._stop_matcher is None:
            import cbor2
            import impact_index

            try:
                with (self.config.index_path / "analyzer.cbor").open("rb") as fp:
                    settings = cbor2.load(fp)
            except (OSError, ValueError):
                settings = {}
            analyzer = impact_index.TextAnalyzer.load(
                str(self.config.index_path),
                stemmer=settings.get("stemmer"),
                language=settings.get("language"),
                stop_words=None,
            )
            self._stop_matcher = TermMatcher(analyzer)
        return self._stop_matcher

    def matcher_for(self, keep_stop_words: bool):
        """Word -> terms, keeping stop words or not"""
        if not keep_stop_words:
            return self.matcher
        if self._keep_matcher is None:
            self._keep_matcher = KeepStopWordsMatcher(self)
        return self._keep_matcher

    def resolver(self, keep_stop_words: bool):
        if not keep_stop_words:
            return self.resolve
        return make_resolver(
            self.matcher.terms, self.stop_words, self.stop_matcher.terms
        )

    def parse(self, text: str, keep_stop_words: bool = False) -> ParsedQuery:
        """Parses a query (plain text is parsed as a sequence of words)

        Raises QuerySyntaxError for malformed structured queries.
        """
        return parse(text, self.resolver(keep_stop_words))

    def plain_words(self, text: str, keep_stop_words: bool = False) -> list[Word]:
        """Words of a plain-text query, with their resolution"""
        resolve = self.resolver(keep_stop_words)
        words = []
        for match in WORD_RE.finditer(text):
            resolution, term_id = resolve(match.group())
            words.append(
                Word(match.group(), match.start(), match.end(), resolution, term_id)
            )
        return words

    def analyze(
        self,
        text: str,
        structured: bool | None = None,
        keep_stop_words: bool = False,
    ) -> AnalyzedQuery:
        """Analyzes a query; ``structured`` forces (or prevents) matchop parsing
        (default: structured as soon as the query uses an operator), and
        ``keep_stop_words`` disables the index's stop list"""
        if structured is None:
            structured = is_structured(text)
        if structured:
            parsed = self.parse(text, keep_stop_words)
            words: dict[int, list[str]] = {}
            for word in parsed.words():
                if word.term_id is not None:
                    seen = words.setdefault(word.term_id, [])
                    if word.text.lower() not in seen:
                        seen.append(word.text.lower())
            terms = [
                QueryTerm(
                    term_id=term,
                    weight=1.0,
                    words=words[term],
                    df=self.index.postings(term).length(),
                )
                for term in parsed.term_ids()
            ]
            for term in terms:
                term.stem, term.stem_verified = self.stem(term.term_id, term.words)
            return AnalyzedQuery(text=text, terms=terms, parsed=parsed)

        vector = self.analyzer.analyze_query(text)
        if keep_stop_words:
            for word in WORD_RE.findall(text):
                if word.lower() in self.stop_words and not self.matcher.terms(word):
                    for term in self.stop_matcher.terms(word):
                        vector[term] = vector.get(term, 0.0) + 1.0
        matcher = self.matcher_for(keep_stop_words)
        words: dict[int, list[str]] = {term: [] for term in vector}
        for word in dict.fromkeys(w.lower() for w in WORD_RE.findall(text)):
            for term in matcher.terms(word):
                if term in words and word not in words[term]:
                    words[term].append(word)
        terms = [
            QueryTerm(
                term_id=term,
                weight=weight,
                words=words[term] or [f"#{term}"],
                df=self.index.postings(term).length(),
            )
            for term, weight in vector.items()
        ]
        # Order terms as they appear in the query
        order = {w: i for i, w in enumerate(WORD_RE.findall(text.lower()))}
        terms.sort(key=lambda t: min(order.get(w, len(order)) for w in t.words))
        for term in terms:
            term.stem, term.stem_verified = self.stem(term.term_id, term.words)
        return AnalyzedQuery(text=text, terms=terms)

    def search(
        self,
        text: str,
        depth: int = 1000,
        params: BM25Params | None = None,
        structured: bool | None = None,
        keep_stop_words: bool = False,
    ) -> SearchResult:
        query = self.analyze(text, structured, keep_stop_words)
        if not query.terms:
            return SearchResult(query=query, hits=[])
        scored = self.scored(params or self.config.bm25)
        if query.parsed is not None:
            # With the index's stop list, impact-index parses the string
            # itself (the reference); otherwise, the tree resolved here
            tree = to_query_tree(query.parsed) if keep_stop_words else text
            if tree is None:
                return SearchResult(query=query, hits=[])
            try:
                results = scored.search_maxscore_query(tree, depth)
            except ValueError as e:
                # impact-index's parser is the reference: report its errors
                raise QuerySyntaxError(str(e), 0, len(text)) from e
        else:
            results = scored.search_maxscore(query.vector, depth)
        ext_ids = self.documents.external_ids([r.docid for r in results])
        hits = [
            Hit(rank=rank, docid=ext_id, score=r.score)
            for rank, (ext_id, r) in enumerate(
                zip(ext_ids, results, strict=True), start=1
            )
        ]
        return SearchResult(query=query, hits=hits)


class Engines:
    """Lazily opened search engines, shared by all clients"""

    def __init__(self, workspace: Workspace, opener=SearchEngine.open):
        self.workspace = workspace
        self.opener = opener
        self._engines: dict[str, SearchEngine] = {}
        self._lock = threading.Lock()
        workspace.on_change(self.invalidate)

    def get(self, name: str) -> SearchEngine:
        with self._lock:
            if name not in self._engines:
                self._engines[name] = self.opener(self.workspace.collection(name))
            return self._engines[name]

    def loaded(self, name: str) -> SearchEngine | None:
        """The engine if it is already open (never blocks on loading)"""
        return self._engines.get(name)

    def invalidate(self, name: str):
        """Drops the engine of a collection whose settings changed"""
        with self._lock:
            self._engines.pop(name, None)
