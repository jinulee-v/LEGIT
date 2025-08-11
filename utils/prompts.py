import os

prompts = {}
for filename in os.listdir("prompts"):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        prompts[filename.replace(".txt", "")] = f.read()