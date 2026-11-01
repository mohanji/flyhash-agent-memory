"""Builds docs/demo.html (self-contained) from docs/demo_template.html + docs/demo_data.json."""
import os, json
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
tpl = open(os.path.join(d, "demo_template.html"), encoding="utf-8").read()
data = open(os.path.join(d, "demo_data.json"), encoding="utf-8").read()
json.loads(data)  # validate
open(os.path.join(d, "demo.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", data))
print("wrote docs/demo.html")
