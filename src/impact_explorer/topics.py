"""Topics and assessments from datamaestro IR (Adhoc) datasets."""

import threading
from dataclasses import dataclass, field

DEFAULT_FIELD = "text"


@dataclass
class Topic:
    topic_id: str
    text: str
    fields: dict[str, str] = field(default_factory=dict)
    """All the textual fields (e.g. text/title, description, narrative)"""

    def query(self, name: str = DEFAULT_FIELD) -> str:
        """The text of a field (the default text if the topic lacks it)"""
        return self.fields.get(name) or self.text


def text_fields(item) -> dict[str, str]:
    """Non-empty string fields of a datamaestro text item, ``text`` first"""
    import attrs

    fields = {DEFAULT_FIELD: (item.text or "").strip()}
    if attrs.has(type(item)):
        for attribute in attrs.fields(type(item)):
            value = getattr(item, attribute.name, None)
            if isinstance(value, str) and value.strip():
                fields.setdefault(attribute.name, value.strip())
    return {name: value for name, value in fields.items() if value}


@dataclass
class TopicSet:
    dataset_id: str
    topics: dict[str, Topic]
    qrels: dict[str, dict[str, float]] = field(default_factory=dict)
    """topic id -> document id -> relevance"""

    def field_names(self) -> list[str]:
        """Fields present in the topics (in order of appearance)"""
        names = {DEFAULT_FIELD: None}
        for topic in self.topics.values():
            names.update(dict.fromkeys(topic.fields))
        return list(names)

    def relevant_count(self, topic_id: str) -> int:
        return sum(1 for rel in self.qrels.get(topic_id, {}).values() if rel > 0)


def topic_set_from_adhoc(dataset_id: str, adhoc) -> TopicSet:
    """Builds a topic set from a datamaestro ``Adhoc`` dataset"""
    topics = {}
    for record in adhoc.topics.iter():
        # datamaestro-ir records (IDTextRecord)
        topic_id = record["id"]
        item = record["text_item"]
        topics[topic_id] = Topic(topic_id, item.text.strip(), text_fields(item))

    qrels: dict[str, dict[str, float]] = {}
    assessments = getattr(adhoc, "assessments", None)
    if assessments is not None:
        for assessed in assessments.iter():
            judged = qrels.setdefault(assessed.topic_id, {})
            for assessment in assessed.assessments:
                judged[assessment.doc_id] = float(getattr(assessment, "rel", 1.0))
    return TopicSet(dataset_id=dataset_id, topics=topics, qrels=qrels)


class TopicSets:
    """Lazily prepared datamaestro topic sets (shared by all clients)"""

    def __init__(self, loader=None):
        self._loader = loader or self._prepare
        self._sets: dict[str, TopicSet] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _prepare(dataset_id: str) -> TopicSet:
        from datamaestro import prepare_dataset

        return topic_set_from_adhoc(dataset_id, prepare_dataset(dataset_id))

    def put(self, topic_set: TopicSet):
        with self._lock:
            self._sets[topic_set.dataset_id] = topic_set

    def get(self, dataset_id: str) -> TopicSet:
        with self._lock:
            if dataset_id not in self._sets:
                self._sets[dataset_id] = self._loader(dataset_id)
            return self._sets[dataset_id]
