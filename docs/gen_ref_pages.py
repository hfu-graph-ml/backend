"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

for path in sorted(Path("backend").rglob("*.py")):
  module_path = path.relative_to("backend").with_suffix("")
  doc_path = path.relative_to("backend").with_suffix(".md")
  full_doc_path = Path("reference", doc_path)

  parts = list(module_path.parts)

  if parts[-1] == "__init__":
    parts = parts[:-1]
  elif parts[-1] == "__main__":
    continue

  if path.name == "main.py" or path.name == "__init__.py":
    continue

  with mkdocs_gen_files.open(doc_path, "w") as fd:
    identifier = "backend." + ".".join(parts)
    print("::: " + identifier, file=fd)

  mkdocs_gen_files.set_edit_path(doc_path, path)