import json

with open("data/legit.jsonl", "r", encoding="utf-8") as f:
    data_corpus = [json.loads(line) for line in f]

index = input(f"인덱스를 입력하세요 (0-{len(data_corpus) - 1}): ")
try:
    index = int(index)
    if 0 <= index < len(data_corpus):
        datum = data_corpus[index]
        print(datum["precedent"])
        print("----")
        print(datum["event_summary"])
        print("----")
        print(json.dumps(datum["issues"], ensure_ascii=False, indent=2))
    else:
        print("유효하지 않은 인덱스입니다.")
except ValueError:
    print("유효한 숫자를 입력하세요.")