"""Create two hashed, reproducible dataset versions from the supplied CSV."""

from pathlib import Path

from wb_sentiment.data import save_versions

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "data/raw/filtered.csv"
    if not source.exists():
        raise FileNotFoundError("Поместите исходный файл в data/raw/filtered.csv (см. README).")
    manifest = save_versions(source, ROOT / "data/versions")
    for version, details in manifest["versions"].items():
        print(f"{version}: {details['rows']} строк; sha256={details['sha256']}")


if __name__ == "__main__":
    main()
