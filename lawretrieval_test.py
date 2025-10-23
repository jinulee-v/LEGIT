import json
import re

with open("data/deduplicated_relevant_laws.json") as file:
    laws = json.load(file)

supreme_court_pattern = re.compile(r"[0-9]+[다두][0-9]+")

max_len = 0
max_law = None
for law in laws:
    patterns = set()
    for var in law["variants"]:
        for match in supreme_court_pattern.finditer(var):
            patterns.add(match.group(0))
    
    if len(patterns) > max_len:
        max_len = len(patterns)
        max_law = law

print(max_len)
print(json.dumps(max_law, ensure_ascii=False, indent=4))