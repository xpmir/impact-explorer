from impact_explorer.index_info import index_details, recognize_pipeline, summary


def test_recognize_pipeline():
    terrier = {
        "tokenizer": "pisa-english",
        "stemmer": "porter",
        "stop_words_list": ["a", "the"],
        "stop_words_family": "terrier",
        "query_stop_words_list": [],
        "position_gaps": False,
    }
    assert recognize_pipeline(terrier) == (
        "terrier",
        ["stemmer porter (default snowball)"],
    )
    pisa = {**terrier, "stemmer": "snowball", "stop_words_list": []}
    pisa["query_stop_words_list"] = ["the"]
    assert recognize_pipeline(pisa) == ("terrier-pisa", [])
    pyserini = {
        "tokenizer": "lucene-english",
        "stemmer": "porter",
        "stop_words_list": ["the"],
        "stop_words_family": "lucene",
        "position_gaps": True,
    }
    assert recognize_pipeline(pyserini) == ("pyserini", [])
    assert recognize_pipeline({"tokenizer": "standard"}) == (None, [])


def test_index_details(collection_folder):
    details = {
        (d.section, d.name): d.value for d in index_details(collection_folder / "index")
    }
    assert details[("Index", "Kind")].startswith("forward")
    assert details[("Index", "Documents")] == "5"
    assert details[("Storage", "Values")].startswith("int32")
    assert details[("Text analysis", "Pipeline")].startswith("pyserini")
    assert "pipeline pyserini" in summary(collection_folder / "index")
