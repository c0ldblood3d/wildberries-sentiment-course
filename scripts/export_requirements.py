"""Export the inference dependency closure with exact versions from poetry.lock."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = tomllib.loads((ROOT / "poetry.lock").read_text())
packages = {item["name"].lower().replace("_", "-"): item for item in lock["package"]}
required = {"fastapi", "uvicorn", "joblib", "scikit-learn"}
pending = list(required)
markers: dict[str, str] = {}
while pending:
    name = pending.pop()
    package = packages[name]
    for dependency, constraint in (package.get("dependencies") or {}).items():
        if isinstance(constraint, dict):
            if constraint.get("optional"):
                continue
            marker = constraint.get("markers")
            dependency = dependency.lower().replace("_", "-")
        else:
            marker = None
            dependency = dependency.lower().replace("_", "-")
        if dependency not in packages:
            continue
        if dependency not in required:
            required.add(dependency)
            pending.append(dependency)
        if marker:
            markers[dependency] = marker
lines = []
for name in sorted(required):
    package = packages[name]
    pin = f"{package['name']}=={package['version']}"
    marker = markers.get(name) or package.get("markers")
    if marker:
        pin += f"; {marker}"
    lines.append(pin)
(ROOT / "requirements.txt").write_text("\n".join(lines) + "\n")
print(f"Exported {len(lines)} inference pins from poetry.lock")
