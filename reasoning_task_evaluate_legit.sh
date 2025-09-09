#!/bin/bash

PACKAGE=$1
MODEL=$2

python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_EXAONE-3.0-7.8B-Instruct.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_exaone3.5:7.8b.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_exaone3.5:32b.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemini-2.0-flash-001.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemini-2.5-flash-001.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemini-2.5-pro.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemma-3-4b-it.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemma-3-12b-it.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gemma-3-27b-it.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gpt-4.1-mini.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_gpt-4.1.jsonl --package ${PACKAGE} --model ${MODEL}
python reasoning_task_evaluate_legit.py --response_path results/reasoning_tasks_o3.jsonl --package ${PACKAGE} --model ${MODEL}