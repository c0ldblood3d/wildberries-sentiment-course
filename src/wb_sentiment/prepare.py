"""Create two hashed, reproducible dataset versions from the supplied CSV."""

from wb_sentiment.config import load_config, resolve_path
from wb_sentiment.data import save_versions


def main() -> None:
    config = load_config()["train"]
    source = resolve_path(config["source_path"])
    if not source.exists():
        raise FileNotFoundError("Поместите исходный файл в data/raw/filtered.csv (см. README).")
    manifest = save_versions(source, resolve_path(config["versions_dir"]))
    for version, details in manifest["versions"].items():
        print(f"{version}: {details['rows']} строк; sha256={details['sha256']}")


if __name__ == "__main__":
    main()
