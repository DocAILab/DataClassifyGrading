import argparse
import json
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")
CLASSIFICATION_FIELDS = ("level_1", "level_2", "level_3", "level_4")
METADATA_FIELDS = (
    "database_name",
    "database_description",
    "table_name",
    "table_description",
    "field_name",
    "field_description",
    "field_type",
    "value",
)

DEVELOPER_PROMPT = """你是数据库字段数据分类分级智能体。
根据用户提供的字段元数据，预测四级数据分类和数据等级。
只能输出一个合法、紧凑的 JSON 对象，不得输出解释、Markdown 或其他文字。
输出结构必须严格为：
{"classification":{"level_1":"","level_2":"","level_3":"","level_4":""},"data_level":""}
如果训练数据中某个分类层级为空，输出中对应层级也保持空字符串。"""


def load_json_list(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Input split file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Input split must be a JSON array: {path}")

    return data


def validate_item(item, index, source_path):
    if not isinstance(item, dict):
        raise ValueError(f"Item {index} in {source_path} must be an object")

    metadata = item.get("metadata")
    classification = item.get("classification")

    if not isinstance(metadata, dict):
        raise ValueError(f"Item {index} in {source_path} has invalid metadata")
    if not isinstance(classification, dict):
        raise ValueError(
            f"Item {index} in {source_path} has invalid classification"
        )

    missing_metadata = [field for field in METADATA_FIELDS if field not in metadata]
    missing_classification = [
        field for field in CLASSIFICATION_FIELDS if field not in classification
    ]

    if missing_metadata:
        raise ValueError(
            f"Item {index} in {source_path} is missing metadata fields: "
            + ", ".join(missing_metadata)
        )
    if missing_classification:
        raise ValueError(
            f"Item {index} in {source_path} is missing classification fields: "
            + ", ".join(missing_classification)
        )


def is_labeled(item):
    classification = item["classification"]
    inferred = any(
        str(classification.get(field, "")).strip()
        for field in CLASSIFICATION_FIELDS
    )
    status = item.get("label_status")

    if status == "unlabeled" and inferred:
        raise ValueError(
            f"Item {item.get('id', '<unknown>')} has label_status=unlabeled "
            "but contains a classification label"
        )
    if status == "labeled" and not inferred:
        raise ValueError(
            f"Item {item.get('id', '<unknown>')} has label_status=labeled "
            "but all classification levels are empty"
        )

    return inferred


def build_user_content(metadata):
    labels = {
        "database_name": "数据库名称",
        "database_description": "数据库描述",
        "table_name": "表名称",
        "table_description": "表描述",
        "field_name": "字段名称",
        "field_description": "字段描述",
        "field_type": "字段类型",
        "value": "字段样例值",
    }
    lines = ["请对以下数据库字段进行数据分类分级："]

    for field in METADATA_FIELDS:
        value = str(metadata.get(field, "")).strip()

        if value:
            lines.append(f"{labels[field]}：{value}")

    return "\n".join(lines)


def build_reference_answer(item):
    classification = item["classification"]

    return {
        "classification": {
            field: str(classification.get(field, ""))
            for field in CLASSIFICATION_FIELDS
        },
        "data_level": str(item.get("data_level", "")),
    }


def convert_item(item):
    reference_answer = build_reference_answer(item)
    reference_answer_json = json.dumps(
        reference_answer,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {
                "role": "user",
                "content": build_user_content(item["metadata"]),
            },
        ],
        "reference_answer": reference_answer,
        "reference_answer_json": reference_answer_json,
        "source_id": str(item.get("id", "")),
    }


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    temporary_path.replace(path)


def export_split(input_path, output_path):
    data = load_json_list(input_path)
    records = []
    skipped_unlabeled = 0

    for index, item in enumerate(data):
        validate_item(item, index, input_path)

        if not is_labeled(item):
            skipped_unlabeled += 1
            continue

        records.append(convert_item(item))

    if not records:
        raise ValueError(f"No labeled records available in {input_path}")

    write_jsonl(records, output_path)

    return {
        "input_records": len(data),
        "exported_records": len(records),
        "skipped_unlabeled": skipped_unlabeled,
        "output_file": str(output_path),
    }


def export_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report = {"format": "openai_rft_jsonl", "splits": {}}

    for split_name in SPLIT_NAMES:
        report["splits"][split_name] = export_split(
            input_dir / f"{split_name}.json",
            output_dir / f"{split_name}.jsonl",
        )

    report_path = output_dir / "export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert processed train/val/test JSON files to OpenAI RFT JSONL."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed/pers_info",
        help=(
            "Prepared dataset directory containing train.json, val.json "
            "and test.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/rft/pers_info",
        help="Directory for train.jsonl, val.jsonl and test.jsonl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = export_dataset(arguments.input_dir, arguments.output_dir)

    for name, details in result["splits"].items():
        print(
            f"{name}: exported={details['exported_records']}, "
            f"skipped_unlabeled={details['skipped_unlabeled']}"
        )
