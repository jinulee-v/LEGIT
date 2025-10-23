import os
import asyncio
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()

default_generation_config = {
    "temperature": 0,
    "top_p": 0.95,
}
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
request_id_cnt = 0
lock = asyncio.Lock()
client = AsyncOpenAI(
    base_url=VLLM_URL
)

async def generate(model: str, prompt: str, system_prompt: str, response_schema: BaseModel = None, vllm_url=VLLM_URL):
    """Send async HTTP request to vLLM server instead of using SDK."""
    global request_id_cnt
    async with lock:
        request_id = f"vllm_async_request_{request_id_cnt}"
        request_id_cnt += 1

    # Merge prompts
    full_prompt = system_prompt + "\n\n" + prompt

    # Guided decoding schema if provided
    # guided_decoding_params = None
    # if response_schema is not None:
    #     guided_decoding_params = GuidedDecodingParams(
    #         json=
    #     )

    # Prepare payload (align with vLLM REST API)
    if response_schema:
        chat_completion = client.chat.completions.parse(
            messages = [{"role": "user", "content": full_prompt}],
            model = model,
            temperature = default_generation_config["temperature"],
            top_p = default_generation_config["top_p"],
            # guided_decoding = response_schema.model_json_schema(),
            response_format = response_schema,
            timeout=3000
        )
    else:
        chat_completion = client.chat.completions.create(
            messages = [{"role": "user", "content": full_prompt}],
            model = model,
            temperature = default_generation_config["temperature"],
            top_p = default_generation_config["top_p"],
            timeout=3000
        )
    try:
        chat_completion = await chat_completion
    except:
        # Any OpenAI.*Exception (length, timeout, ...)
        return None
    data = chat_completion.choices[0].message.content

    if response_schema:
        return json.loads(data)
    else:
        return data


class LegitEvaluation(BaseModel):
    rationales: str
    contains_issue: bool
    correct_conclusion: bool
class FluentEvaluation(BaseModel):
    rationales: str
    is_fluent: bool

# Set model ID
model = os.environ.get("VERL_JUDGE_VLLM_MODEL", "google/gemma-3-27b-it")

async def get_fluency_reward(response: str):
    system_prompt = "다음의 글이 자연스러운 한국어로 작성되었는지 여부를 판단하라. 이상한 단어가 계속 반복되거나, 문법 오류가 심각하거나, 문맥에 맞지 않는 단어가 자주 등장하는 등의 문제가 없을 경우에는 자연스럽다고 본다. 답변은 다음의 JSON 형식으로 작성할 것. {\"rationales\": str, \"is_fluent\": bool}\n\n"
    result = generate(model, response, system_prompt=system_prompt, response_schema=FluentEvaluation)
    return await result

async def get_legit_issue_reward(response: str, issue_rubric: str):
    prompt = issue_rubric.replace("{response}", response)
    result = generate(model, prompt, system_prompt="", response_schema=LegitEvaluation)
    return await result

def reward_fn_batch_full_issues(data_sources, solution_strs, ground_truths, extra_infos):
    batch_size = len(data_sources)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Submit coroutines
    fluency_reward_coros = [loop.create_task(get_fluency_reward(solution_str)) for solution_str in solution_strs]
    issue_reward_coros = [[loop.create_task(get_legit_issue_reward(solution_str, rubric)) for rubric in extra_info["rubrics"]] for solution_str, extra_info in zip(solution_strs, extra_infos)]
    all_tasks = fluency_reward_coros + [coro for sublist in issue_reward_coros for coro in sublist]
    print("reward_fn: batch size:", batch_size)
    print("reward_fn: number of async vllm calls:", len(all_tasks))
    loop.run_until_complete(asyncio.wait(all_tasks))
    print("reward_fn: Finish!")

    fluency_rewards = [t.result() for t in fluency_reward_coros]
    issue_rewards = [[t.result() for t in sublist] for sublist in issue_reward_coros]

    # Block until each coroutine is finished
    rewards = []
    for i in range(batch_size):
        # Compute fluency reward
        fluency_reward = fluency_rewards[i]
        if fluency_reward is None:
            rewards.append(-5.0)
            continue
        fluency_reward = fluency_reward["is_fluent"]
        if not fluency_reward:
            rewards.append(-5.0)
            continue

        # Compute legit reward
        reward = 0.0

        # Compute root reward
        root_reward = issue_rewards[i][0]
        if root_reward is not None and root_reward["correct_conclusion"]:
            reward += 5.0
        for rubric_idx in range(1, len(issue_rewards[i])):
            issue_reward = issue_rewards[i][rubric_idx]
            if issue_reward is not None and issue_reward["contains_issue"]:
                if issue_reward["correct_conclusion"]:
                    reward += 5.0 / (len(issue_reward_coros[i]) - 1)
                else:
                    reward += 2.0 / (len(issue_reward_coros[i]) - 1)
            else:
                reward += 0.0
        rewards.append(reward)
    return rewards

def reward_fn_batch_root_only(data_sources, solution_strs, ground_truths, extra_infos):
    batch_size = len(data_sources)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Submit coroutines
    fluency_reward_coros = [loop.create_task(get_fluency_reward(solution_str)) for solution_str in solution_strs]
    issue_reward_coros = [[loop.create_task(get_legit_issue_reward(solution_str, rubric)) for rubric in extra_info["rubrics"][:1]] for solution_str, extra_info in zip(solution_strs, extra_infos)]
    all_tasks = fluency_reward_coros + [coro for sublist in issue_reward_coros for coro in sublist]
    print("reward_fn: batch size:", batch_size)
    print("reward_fn: number of async vllm calls:", len(all_tasks))
    loop.run_until_complete(asyncio.wait(all_tasks))
    print("reward_fn: Finish!")

    fluency_rewards = [t.result() for t in fluency_reward_coros]
    issue_rewards = [[t.result() for t in sublist] for sublist in issue_reward_coros]

    # Block until each coroutine is finished
    rewards = []
    for i in range(batch_size):
        # Compute fluency reward
        fluency_reward = fluency_rewards[i]
        if fluency_reward is None:
            rewards.append(-5.0)
            continue
        fluency_reward = fluency_reward["is_fluent"]
        if not fluency_reward:
            rewards.append(-5.0)
            continue

        # Compute legit reward
        reward = 0.0

        # Compute root reward
        root_reward = issue_rewards[i][0]
        if root_reward is not None and root_reward["correct_conclusion"]:
            reward += 10.0
        rewards.append(reward)
    return rewards


# TEST CODE

async def main_async():
    # Test the reward functions
    test_response = '주어진 사실관계와 청구취지를 토대로 재판의 결과를 예측해 보겠습니다.\n\n---\n\n## 1. 사실관계의 법적 구조\n\n* **명의신탁 관계**:\n  1995.3.16. 피고 명의로 소유권이전등기가 되었으나, 실제로는 원고 A, B, C의 모 G도 공동 매수인으로서 참여. 즉, 당사자들 사이에는 **계약명의신탁** 관계가 형성됨.\n\n* **명의신탁 유형**:\n  매도인 I는 원고들의 실질적인 매수인 지위를 알지 못하였으므로, 이는 \\*\\*“매도인 선의의 계약명의신탁”\\*\\*에 해당.\n\n* **부동산 실명법 적용**:\n  「부동산 실권리자명의 등기에 관한 법률」(부동산실명법) 제11조에 따라, 1996.7.1.까지 한시적으로 유예기간이 있었고, 그 이후에는 명의신탁약정 및 그에 따른 등기는 **무효**가 되고, 명의수탁자인 피고가 완전한 소유권을 취득하게 됨.\n  → 따라서 원고들(G, A, B, C)은 자신들의 명의신탁 지분에 대해 더 이상 물권적 권리를 주장할 수 없음.\n\n* **확인서, 합의서, 사실확인서 작성**:\n  피고가 1996년, 1999년, 2000년에 확인서와 합의서를 작성한 사실은 있으나, 이는 **실명법의 강행규정**을 무력화할 수 없음. 사적 합의로 다시 원고들에게 소유권을 인정할 수는 없으며, 법률상 소유권은 이미 피고에게 귀속됨.\n\n* **근저당권 설정**:\n  원고 A, B 및 G(→C) 명의로 근저당권을 설정했으나, 이는 단순히 “무단처분 방지용” 장치였을 뿐이며, 실질적으로는 명의신탁약정을 전제로 한 것이므로 실효적 권리 보장은 어려움.\n\n---\n\n## 2. 쟁점\n\n1. **명의신탁자(원고들)가 피고를 상대로 공유지분에 관한 소유권이전등기청구(부당이득반환청구 포함)를 할 수 있는지 여부**\n   → 부동산실명법 위반 여부에 따라 판단.\n\n2. **피고가 작성한 확인서·합의서가 원고들에게 새로운 권리(채권적 청구권)를 인정하는지 여부**\n   → “소유권이전등기청구”의 근거로는 부족함.\n\n---\n\n## 3. 판례의 태도\n\n* 대법원은 일관되게 **매도인 선의의 계약명의신탁은 무효**이고,\n  유예기간(1996.7.1.) 이후에는 명의수탁자가 단독 소유권자가 되며, 명의신탁자는 소유권을 주장할 수 없다고 판시(대법원 2002다1178 등).\n\n* 다만, 명의수탁자가 신탁자에게 작성해준 확인서 등은 **소유권이전등기청구권이 아니라 손해배상청구(부당이득 반환 포함) 또는 채권적 반환청구의 증거**로 쓰일 수 있을 뿐, 물권적 청구는 허용되지 않음.\n\n---\n\n## 4. 결론 – 예상 판결\n\n* **원고들의 청구(소유권이전등기청구)는 기각될 가능성이 높음.**\n  → 이유: 이 사건은 ‘매도인 선의의 계약명의신탁’으로서 부동산실명법 제4조, 제11조에 따라 명의신탁약정 및 그에 따른 등기는 무효이고, 1996.7.1. 이후 피고가 완전한 소유권을 취득하였기 때문.\n\n* 다만, 원고들은 자신들의 출자금 또는 매수대금 상당액에 대해 \\*\\*피고를 상대로 부당이득반환청구(금전청구)\\*\\*를 할 여지가 있음.\n  그러나 현재 소송의 청구취지가 “소유권이전등기청구”로 특정되어 있으므로, 재판부는 **원고 패소 판결**을 내릴 것으로 보임.\n\n---\n\n✅ **예측 결과**:\n원고들의 청구(각 지분에 관한 소유권이전등기 절차 이행)는 기각되고, 피고가 임야의 단독 소유권자로 확정될 가능성이 큼.\n\n---\n\n원하시면 제가 대법원 판례와 비교하여 “패소 확정 판결문 요약 형식”으로 재구성해드릴까요?\n'
    test_rubrics = [
        "법률 쟁점에 대한 글을 읽고, (1) \"summary/claim\"의 내용이 TEXT에서 다뤄졌는지와 (2) 해당 \"summary/claim\"에 대한 결론이 \"conclusion\"이 TEXT 부분과 일치하는지를 판단하시오.\n\n평가 방법:\n- conclusion의 방향이 TEXT 부분과 일치하는지 판단할 것.\n - conclusion이 \"주장이 옳다/그르다\" 형식인 경우, \"claim\"에 명시된 주장이 본문에서 다루어졌는지를 먼저 판단하고, 그 결론이 옳은지 그른지에 대해서 판단하시오.\n - conclusion이 판결문 주문의 형식인 경우, 지급액, 이행의 내용, 기산일, 기각/각하 여부, ... 등을 모두 정확하게 예측하였는지 판단하시오. 단, 소송비용 분담과 가집행에 대한 내용은 제외한다.\n - 그 외 conclusion의 경우, 글 전체의 취지를 고려하여 동일한 주장을 다루고 같은 결론에 도달하였다고 할 수 있는지를 판단하시오.\n - 각하/기각, 소유권/점유권, 이행이익/신뢰이익 등 법률용어의 일치 여부를 매우 엄격하게 적용할 것.\n\n형식:\n- 답변의 형식은 엄격하게 <OUTPUT>{\"rationales\": str, \"contains_issue\": bool, \"correct_conclusion\": bool}</OUTPUT> 형식을 지킬 것.\n- rationales에는 1문단 정도 길이로 간단하게 평가 이유를 서술하시오.\n- JSON 태그의 최종 형식은 예시와 같이 Dict 형식으로 출력하시오.\n\n<EXAMPLE>\n<INPUT>\n<TEXT>\n민사소송법 제269조 제1항에서 말하는 본소의 청구와의 상호 관련성은 본소와 반소의 양 청구가 소송물 또는 그 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻하고, 본소의 방어 방법과의 상호 관련성은 반소 청구가 본소의 항변 사유와 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻한다.\n\n이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.\n\n따라서 피고의 반소 청구는 적법하다.\n</TEXT>\n<ISSUE>\nsummary: 피고의 반소 청구에 대한 판단\nclaims:\n피고: 원고의 횡령 및 업무방해로 인한 부당이득의 반환 및 손해배상을 구하면서 원고의 횡령 및 업무방해 범행이 이 사건 폭행의 동기가 되었으므로 이 사건 본소 청구와 관련이 있다.\nconclusion:\n피고의 주장은 이유 없다.\n</ISSUE>\n</INPUT>\n<OUTPUT>{\"rationales\": \"'이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.' 부분을 보면, 주어진 TEXT는 피고의 반소청구가 본소 청구와 관련이 있는지 판단하고 있다. 그러나, 정답 conclusion은 이러한 청구가 적법하지 않다고 본 반면 본문에서는 피고의 반소 청구가 적법하다고 평가하고 있다. 따라서 contains_issue는 True, correct_conclusion는 False이다.\", \"contains_issue\": true, \"correct_conclusion\": false}\n</OUTPUT>\n</EXAMPLE>\n\n<INPUT>\n<TEXT>\n{response}\n</TEXT>\n<ISSUE>\nsummary: \nclaims:\n원고들: 피고는 거제시 E 임야 149,520㎡ 중 원고 A에게 149520분의 31405 지분, 원고 B에게 149520분의 34710 지분, 원고 C에게 149520분의 36364 지분에 관하여 각 부당이득반환을 원인으로 한 소유권이전등절차를 이행하라.\nconclusion:\n1. 원고들의 청구를 모두 기각한다.\n2. 소송비용은 원고들이 부담한다.\n</ISSUE>\n</INPUT>\n<OUTPUT>",
        "법률 쟁점에 대한 글을 읽고, (1) \"summary/claim\"의 내용이 TEXT에서 다뤄졌는지와 (2) 해당 \"summary/claim\"에 대한 결론이 \"conclusion\"이 TEXT 부분과 일치하는지를 판단하시오.\n\n평가 방법:\n- conclusion의 방향이 TEXT 부분과 일치하는지 판단할 것.\n - conclusion이 \"주장이 옳다/그르다\" 형식인 경우, \"claim\"에 명시된 주장이 본문에서 다루어졌는지를 먼저 판단하고, 그 결론이 옳은지 그른지에 대해서 판단하시오.\n - conclusion이 판결문 주문의 형식인 경우, 지급액, 이행의 내용, 기산일, 기각/각하 여부, ... 등을 모두 정확하게 예측하였는지 판단하시오. 단, 소송비용 분담과 가집행에 대한 내용은 제외한다.\n - 그 외 conclusion의 경우, 글 전체의 취지를 고려하여 동일한 주장을 다루고 같은 결론에 도달하였다고 할 수 있는지를 판단하시오.\n - 각하/기각, 소유권/점유권, 이행이익/신뢰이익 등 법률용어의 일치 여부를 매우 엄격하게 적용할 것.\n\n형식:\n- 답변의 형식은 엄격하게 <OUTPUT>{\"rationales\": str, \"contains_issue\": bool, \"correct_conclusion\": bool}</OUTPUT> 형식을 지킬 것.\n- rationales에는 1문단 정도 길이로 간단하게 평가 이유를 서술하시오.\n- JSON 태그의 최종 형식은 예시와 같이 Dict 형식으로 출력하시오.\n\n<EXAMPLE>\n<INPUT>\n<TEXT>\n민사소송법 제269조 제1항에서 말하는 본소의 청구와의 상호 관련성은 본소와 반소의 양 청구가 소송물 또는 그 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻하고, 본소의 방어 방법과의 상호 관련성은 반소 청구가 본소의 항변 사유와 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻한다.\n\n이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.\n\n따라서 피고의 반소 청구는 적법하다.\n</TEXT>\n<ISSUE>\nsummary: 피고의 반소 청구에 대한 판단\nclaims:\n피고: 원고의 횡령 및 업무방해로 인한 부당이득의 반환 및 손해배상을 구하면서 원고의 횡령 및 업무방해 범행이 이 사건 폭행의 동기가 되었으므로 이 사건 본소 청구와 관련이 있다.\nconclusion:\n피고의 주장은 이유 없다.\n</ISSUE>\n</INPUT>\n<OUTPUT>{\"rationales\": \"'이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.' 부분을 보면, 주어진 TEXT는 피고의 반소청구가 본소 청구와 관련이 있는지 판단하고 있다. 그러나, 정답 conclusion은 이러한 청구가 적법하지 않다고 본 반면 본문에서는 피고의 반소 청구가 적법하다고 평가하고 있다. 따라서 contains_issue는 True, correct_conclusion는 False이다.\", \"contains_issue\": true, \"correct_conclusion\": false}\n</OUTPUT>\n</EXAMPLE>\n\n<INPUT>\n<TEXT>\n{response}\n</TEXT>\n<ISSUE>\nsummary: 부당이득반환을 원인으로 한 소유권이전등기 청구에 대한 판단\nclaims:\n원고들: 피고는 이 사건 임야의 공동매수인인 원고들과 사이에 이루어진 명의신탁약정에 따라 이러한 명의신탁약정을 알지 못하는 매도인 I으로부터 이 사건 임야를 매수하는 계약을 체결한 뒤, 1995. 3. 16. 피고 단독 명의로 소유권이전등기를 마쳤는데, 이후 부동산 실권리자명의 등기에 관한 법률(이하 '부동산실명법'이라고 한다) 제11조에서 정한 유예기간이 경과함에 따라 피고가 1996. 7. 1. 이 사건 임야에 대한 완전한 소유권을 취득하게 되었으므로, 명의수탁자인 피고는 명의신탁자인 원고들에게 이 사건 임야 중 청구취지 기재 원고들 각 매수지분을 부당이득으로 반환할 의무가 있다.\nconclusion:\n원고들의 이 사건 청구는 모두 이유 없어 이를 기각한다.\n</ISSUE>\n</INPUT>\n<OUTPUT>"
    ]

    print("Testing fluency reward...")
    fluency_result = await get_fluency_reward(test_response)
    print(f"Fluency Result: {fluency_result}")

    print("Testing legit issue reward...")
    legit_result = await get_legit_issue_reward(test_response, test_rubrics[0])
    print(f"Legit Issue Result: {legit_result}")

def main():
    # Test the reward functions
    test_response = '주어진 사실관계와 청구취지를 토대로 재판의 결과를 예측해 보겠습니다.\n\n---\n\n## 1. 사실관계의 법적 구조\n\n* **명의신탁 관계**:\n  1995.3.16. 피고 명의로 소유권이전등기가 되었으나, 실제로는 원고 A, B, C의 모 G도 공동 매수인으로서 참여. 즉, 당사자들 사이에는 **계약명의신탁** 관계가 형성됨.\n\n* **명의신탁 유형**:\n  매도인 I는 원고들의 실질적인 매수인 지위를 알지 못하였으므로, 이는 \\*\\*“매도인 선의의 계약명의신탁”\\*\\*에 해당.\n\n* **부동산 실명법 적용**:\n  「부동산 실권리자명의 등기에 관한 법률」(부동산실명법) 제11조에 따라, 1996.7.1.까지 한시적으로 유예기간이 있었고, 그 이후에는 명의신탁약정 및 그에 따른 등기는 **무효**가 되고, 명의수탁자인 피고가 완전한 소유권을 취득하게 됨.\n  → 따라서 원고들(G, A, B, C)은 자신들의 명의신탁 지분에 대해 더 이상 물권적 권리를 주장할 수 없음.\n\n* **확인서, 합의서, 사실확인서 작성**:\n  피고가 1996년, 1999년, 2000년에 확인서와 합의서를 작성한 사실은 있으나, 이는 **실명법의 강행규정**을 무력화할 수 없음. 사적 합의로 다시 원고들에게 소유권을 인정할 수는 없으며, 법률상 소유권은 이미 피고에게 귀속됨.\n\n* **근저당권 설정**:\n  원고 A, B 및 G(→C) 명의로 근저당권을 설정했으나, 이는 단순히 “무단처분 방지용” 장치였을 뿐이며, 실질적으로는 명의신탁약정을 전제로 한 것이므로 실효적 권리 보장은 어려움.\n\n---\n\n## 2. 쟁점\n\n1. **명의신탁자(원고들)가 피고를 상대로 공유지분에 관한 소유권이전등기청구(부당이득반환청구 포함)를 할 수 있는지 여부**\n   → 부동산실명법 위반 여부에 따라 판단.\n\n2. **피고가 작성한 확인서·합의서가 원고들에게 새로운 권리(채권적 청구권)를 인정하는지 여부**\n   → “소유권이전등기청구”의 근거로는 부족함.\n\n---\n\n## 3. 판례의 태도\n\n* 대법원은 일관되게 **매도인 선의의 계약명의신탁은 무효**이고,\n  유예기간(1996.7.1.) 이후에는 명의수탁자가 단독 소유권자가 되며, 명의신탁자는 소유권을 주장할 수 없다고 판시(대법원 2002다1178 등).\n\n* 다만, 명의수탁자가 신탁자에게 작성해준 확인서 등은 **소유권이전등기청구권이 아니라 손해배상청구(부당이득 반환 포함) 또는 채권적 반환청구의 증거**로 쓰일 수 있을 뿐, 물권적 청구는 허용되지 않음.\n\n---\n\n## 4. 결론 – 예상 판결\n\n* **원고들의 청구(소유권이전등기청구)는 기각될 가능성이 높음.**\n  → 이유: 이 사건은 ‘매도인 선의의 계약명의신탁’으로서 부동산실명법 제4조, 제11조에 따라 명의신탁약정 및 그에 따른 등기는 무효이고, 1996.7.1. 이후 피고가 완전한 소유권을 취득하였기 때문.\n\n* 다만, 원고들은 자신들의 출자금 또는 매수대금 상당액에 대해 \\*\\*피고를 상대로 부당이득반환청구(금전청구)\\*\\*를 할 여지가 있음.\n  그러나 현재 소송의 청구취지가 “소유권이전등기청구”로 특정되어 있으므로, 재판부는 **원고 패소 판결**을 내릴 것으로 보임.\n\n---\n\n✅ **예측 결과**:\n원고들의 청구(각 지분에 관한 소유권이전등기 절차 이행)는 기각되고, 피고가 임야의 단독 소유권자로 확정될 가능성이 큼.\n\n---\n\n원하시면 제가 대법원 판례와 비교하여 “패소 확정 판결문 요약 형식”으로 재구성해드릴까요?\n'
    test_rubrics = [
        "법률 쟁점에 대한 글을 읽고, (1) \"summary/claim\"의 내용이 TEXT에서 다뤄졌는지와 (2) 해당 \"summary/claim\"에 대한 결론이 \"conclusion\"이 TEXT 부분과 일치하는지를 판단하시오.\n\n평가 방법:\n- conclusion의 방향이 TEXT 부분과 일치하는지 판단할 것.\n - conclusion이 \"주장이 옳다/그르다\" 형식인 경우, \"claim\"에 명시된 주장이 본문에서 다루어졌는지를 먼저 판단하고, 그 결론이 옳은지 그른지에 대해서 판단하시오.\n - conclusion이 판결문 주문의 형식인 경우, 지급액, 이행의 내용, 기산일, 기각/각하 여부, ... 등을 모두 정확하게 예측하였는지 판단하시오. 단, 소송비용 분담과 가집행에 대한 내용은 제외한다.\n - 그 외 conclusion의 경우, 글 전체의 취지를 고려하여 동일한 주장을 다루고 같은 결론에 도달하였다고 할 수 있는지를 판단하시오.\n - 각하/기각, 소유권/점유권, 이행이익/신뢰이익 등 법률용어의 일치 여부를 매우 엄격하게 적용할 것.\n\n형식:\n- 답변의 형식은 엄격하게 <OUTPUT>{\"rationales\": str, \"contains_issue\": bool, \"correct_conclusion\": bool}</OUTPUT> 형식을 지킬 것.\n- rationales에는 1문단 정도 길이로 간단하게 평가 이유를 서술하시오.\n- JSON 태그의 최종 형식은 예시와 같이 Dict 형식으로 출력하시오.\n\n<EXAMPLE>\n<INPUT>\n<TEXT>\n민사소송법 제269조 제1항에서 말하는 본소의 청구와의 상호 관련성은 본소와 반소의 양 청구가 소송물 또는 그 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻하고, 본소의 방어 방법과의 상호 관련성은 반소 청구가 본소의 항변 사유와 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻한다.\n\n이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.\n\n따라서 피고의 반소 청구는 적법하다.\n</TEXT>\n<ISSUE>\nsummary: 피고의 반소 청구에 대한 판단\nclaims:\n피고: 원고의 횡령 및 업무방해로 인한 부당이득의 반환 및 손해배상을 구하면서 원고의 횡령 및 업무방해 범행이 이 사건 폭행의 동기가 되었으므로 이 사건 본소 청구와 관련이 있다.\nconclusion:\n피고의 주장은 이유 없다.\n</ISSUE>\n</INPUT>\n<OUTPUT>{\"rationales\": \"'이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.' 부분을 보면, 주어진 TEXT는 피고의 반소청구가 본소 청구와 관련이 있는지 판단하고 있다. 그러나, 정답 conclusion은 이러한 청구가 적법하지 않다고 본 반면 본문에서는 피고의 반소 청구가 적법하다고 평가하고 있다. 따라서 contains_issue는 True, correct_conclusion는 False이다.\", \"contains_issue\": true, \"correct_conclusion\": false}\n</OUTPUT>\n</EXAMPLE>\n\n<INPUT>\n<TEXT>\n{response}\n</TEXT>\n<ISSUE>\nsummary: \nclaims:\n원고들: 피고는 거제시 E 임야 149,520㎡ 중 원고 A에게 149520분의 31405 지분, 원고 B에게 149520분의 34710 지분, 원고 C에게 149520분의 36364 지분에 관하여 각 부당이득반환을 원인으로 한 소유권이전등절차를 이행하라.\nconclusion:\n1. 원고들의 청구를 모두 기각한다.\n2. 소송비용은 원고들이 부담한다.\n</ISSUE>\n</INPUT>\n<OUTPUT>",
        "법률 쟁점에 대한 글을 읽고, (1) \"summary/claim\"의 내용이 TEXT에서 다뤄졌는지와 (2) 해당 \"summary/claim\"에 대한 결론이 \"conclusion\"이 TEXT 부분과 일치하는지를 판단하시오.\n\n평가 방법:\n- conclusion의 방향이 TEXT 부분과 일치하는지 판단할 것.\n - conclusion이 \"주장이 옳다/그르다\" 형식인 경우, \"claim\"에 명시된 주장이 본문에서 다루어졌는지를 먼저 판단하고, 그 결론이 옳은지 그른지에 대해서 판단하시오.\n - conclusion이 판결문 주문의 형식인 경우, 지급액, 이행의 내용, 기산일, 기각/각하 여부, ... 등을 모두 정확하게 예측하였는지 판단하시오. 단, 소송비용 분담과 가집행에 대한 내용은 제외한다.\n - 그 외 conclusion의 경우, 글 전체의 취지를 고려하여 동일한 주장을 다루고 같은 결론에 도달하였다고 할 수 있는지를 판단하시오.\n - 각하/기각, 소유권/점유권, 이행이익/신뢰이익 등 법률용어의 일치 여부를 매우 엄격하게 적용할 것.\n\n형식:\n- 답변의 형식은 엄격하게 <OUTPUT>{\"rationales\": str, \"contains_issue\": bool, \"correct_conclusion\": bool}</OUTPUT> 형식을 지킬 것.\n- rationales에는 1문단 정도 길이로 간단하게 평가 이유를 서술하시오.\n- JSON 태그의 최종 형식은 예시와 같이 Dict 형식으로 출력하시오.\n\n<EXAMPLE>\n<INPUT>\n<TEXT>\n민사소송법 제269조 제1항에서 말하는 본소의 청구와의 상호 관련성은 본소와 반소의 양 청구가 소송물 또는 그 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻하고, 본소의 방어 방법과의 상호 관련성은 반소 청구가 본소의 항변 사유와 대상·발생원인에 있어서 법률상 또는 사실상으로 공통성이 있다는 것을 뜻한다.\n\n이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.\n\n따라서 피고의 반소 청구는 적법하다.\n</TEXT>\n<ISSUE>\nsummary: 피고의 반소 청구에 대한 판단\nclaims:\n피고: 원고의 횡령 및 업무방해로 인한 부당이득의 반환 및 손해배상을 구하면서 원고의 횡령 및 업무방해 범행이 이 사건 폭행의 동기가 되었으므로 이 사건 본소 청구와 관련이 있다.\nconclusion:\n피고의 주장은 이유 없다.\n</ISSUE>\n</INPUT>\n<OUTPUT>{\"rationales\": \"'이 사건 반소 청구원인은 이 사건 폭행의 동기에 해당하여 본소 청구 중 위자료 산정에 있어 참작 사유에 해당한다고 보이므로, 본소의 항변 사유와 사실상으로 공통성이 있음을 추론할 수 있다.' 부분을 보면, 주어진 TEXT는 피고의 반소청구가 본소 청구와 관련이 있는지 판단하고 있다. 그러나, 정답 conclusion은 이러한 청구가 적법하지 않다고 본 반면 본문에서는 피고의 반소 청구가 적법하다고 평가하고 있다. 따라서 contains_issue는 True, correct_conclusion는 False이다.\", \"contains_issue\": true, \"correct_conclusion\": false}\n</OUTPUT>\n</EXAMPLE>\n\n<INPUT>\n<TEXT>\n{response}\n</TEXT>\n<ISSUE>\nsummary: 부당이득반환을 원인으로 한 소유권이전등기 청구에 대한 판단\nclaims:\n원고들: 피고는 이 사건 임야의 공동매수인인 원고들과 사이에 이루어진 명의신탁약정에 따라 이러한 명의신탁약정을 알지 못하는 매도인 I으로부터 이 사건 임야를 매수하는 계약을 체결한 뒤, 1995. 3. 16. 피고 단독 명의로 소유권이전등기를 마쳤는데, 이후 부동산 실권리자명의 등기에 관한 법률(이하 '부동산실명법'이라고 한다) 제11조에서 정한 유예기간이 경과함에 따라 피고가 1996. 7. 1. 이 사건 임야에 대한 완전한 소유권을 취득하게 되었으므로, 명의수탁자인 피고는 명의신탁자인 원고들에게 이 사건 임야 중 청구취지 기재 원고들 각 매수지분을 부당이득으로 반환할 의무가 있다.\nconclusion:\n원고들의 이 사건 청구는 모두 이유 없어 이를 기각한다.\n</ISSUE>\n</INPUT>\n<OUTPUT>"
    ]

    print("Testing full reward function...")
    full_reward = reward_fn_batch_full_issues([None], [test_response], [None], [{"rubrics": test_rubrics}])
    print(f"Full Reward: {full_reward}")

    # print("Testing root reward function...")
    # root_reward = reward_fn_batch_root_only(None, test_response, None, {"rubrics": test_rubrics})
    # print(f"Root Reward: {root_reward}")

if __name__ == "__main__":
    asyncio.run(main_async())
    main()