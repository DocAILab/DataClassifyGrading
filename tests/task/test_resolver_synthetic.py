from agent.task import ClassificationTargetResolver, DatasetConfig, ResolutionStatus


def test_path_target_resolution_uses_explicit_synthetic_config() -> None:
    config = DatasetConfig(
        dataset="demo",
        leaf_level="category",
        path_fields=("group", "category"),
    )
    resolver = ClassificationTargetResolver(config)

    result = resolver.resolve_detailed(
        {"classification": {"group": "Synthetic", "category": "Alpha"}}
    )

    assert result.status is ResolutionStatus.RESOLVED
    assert result.target is not None
    assert result.target.category_id == "demo:Synthetic.Alpha"
    assert result.target.category_path == ("Synthetic", "Alpha")


def test_code_target_resolution_never_falls_back_to_path() -> None:
    config = DatasetConfig(
        dataset="demo",
        leaf_level="category",
        id_strategy="code",
        path_fields=("group", "category"),
    )
    resolver = ClassificationTargetResolver(config, code_leaf_map={})

    result = resolver.resolve_detailed(
        {"classification": {"group": "Synthetic", "category": "Alpha"}}
    )

    assert result.status is ResolutionStatus.CODE_UNRESOLVED
    assert result.target is None
