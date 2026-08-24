from agent.task import LeafRegistry
from method.retrieval.corpus import build_registry_documents


def test_registry_documents_use_descriptions_and_audit_fallback():
    registry = LeafRegistry.from_mapping({"categories": [
        {"category_id": "A", "description": " alpha   data "},
        {"category_id": "B", "description": ""},
        "C", "D", "E",
    ]})

    documents = build_registry_documents(registry)

    assert documents[0].text == "标签名称：A\n标签描述：alpha data"
    assert documents[0].provenance == "registry_description"
    assert documents[1].text == "标签名称：B"
    assert documents[1].provenance == "label_name_fallback"
    assert len(documents[0].checksum) == 64
    assert tuple(item.category_id for item in documents) == registry.ids
