from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, PIIMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage

# checkpointer (단기: thread 대화 저장)
from langgraph.checkpoint.memory import InMemorySaver

# store (장기: 프로필/메모 저장)
from langgraph.store.memory import InMemoryStore

# BraveSearch tool
from langchain_community.tools import BraveSearch

# 모델 (예: OpenAI)
from langchain_openai import ChatOpenAI


# -----------------------------
# 1) 런타임 컨텍스트 (user/thread 식별)
# -----------------------------
@dataclass
class Context:
    user_id: str


# -----------------------------
# 2) 마스킹 유틸 (마지막 4자리만 남김)
# -----------------------------
def mask_keep_last4(raw: str) -> str:
    """문자열에서 숫자만 추출해 마지막 4자리만 보이게 마스킹.
    원래 포맷을 완벽히 보존하진 않지만, 안전/단순성 우선.
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) <= 4:
        return "*" * max(len(digits), 1)
    last4 = digits[-4:]
    return f"****{last4}"


def normalize_profile_payload(profile: dict[str, Any]) -> dict[str, Any]:
    """프로필 데이터에서 민감 필드(SSN/카드/주민번호 등)는 마스킹 저장."""
    out = dict(profile)

    # 흔히 쓰는 키들(원하시면 더 추가 가능)
    SENSITIVE_KEYS = {
        "ssn", "social_security_number", "social", "ssn_last4",
        "credit_card", "card_number", "card",
        "rrn", "resident_registration_number", "주민번호",
    }

    for k in list(out.keys()):
        lk = k.lower()
        if lk in SENSITIVE_KEYS:
            out[k] = mask_keep_last4(str(out[k]))

    return out


# -----------------------------
# 3) store에 프로필 저장/조회 Tool
#    - 모델이 "프로필 정보가 발견되면" 이 tool을 호출하도록 system_prompt로 유도
# -----------------------------
@tool
def upsert_user_profile(
    profile: dict[str, Any],
    runtime: ToolRuntime,
) -> str:
    """유저 프로필 정보를 장기 저장소(store)에 저장/업데이트합니다.
    민감정보(SSN/카드/주민번호)는 마지막 4자리만 남기고 마스킹하여 저장합니다.
    """
    user_id = runtime.context.user_id
    namespace = ("users", user_id)

    safe_profile = normalize_profile_payload(profile)

    # 기존 프로필과 병합
    existing = runtime.store.get(namespace, "profile")
    merged = {}
    if existing and isinstance(existing.value, dict):
        merged.update(existing.value)
    merged.update(safe_profile)

    runtime.store.put(namespace, "profile", merged)
    return f"✅ 프로필 저장 완료(민감정보는 마스킹 저장). 현재 키: {sorted(list(merged.keys()))}"


@tool
def get_user_profile(runtime: ToolRuntime) -> dict[str, Any]:
    """유저 프로필을 장기 저장소(store)에서 가져옵니다(민감정보는 마스킹된 형태로 저장되어 있음)."""
    user_id = runtime.context.user_id
    namespace = ("users", user_id)
    item = runtime.store.get(namespace, "profile")
    return item.value if item else {}


# -----------------------------
# 4) BraveSearch 도구
# -----------------------------
# 문서 예제: BraveSearch.from_api_key(api_key=..., search_kwargs={"count":3}) :contentReference[oaicite:2]{index=2}
brave_search = BraveSearch.from_search_kwargs(search_kwargs={"count": 5})


# -----------------------------
# 5) store / checkpointer / middleware 구성
# -----------------------------
store = InMemoryStore()         # 운영에서는 PostgresStore 등으로 교체 권장
checkpointer = InMemorySaver()  # 운영에서는 PostgresSaver/Redis checkpointer로 교체 권장

system_prompt = """
당신은 고객 상담/정보 도우미 에이전트입니다.

규칙:
1) 사용자가 '프로필 정보'(이름/이메일/전화/주소/생년월일/선호 등)를 제공하면 즉시 upsert_user_profile tool을 호출해서 저장하세요.
2) 사용자가 SSN/카드번호/주민번호 등의 민감정보를 제공하거나 프로필에 포함시키려 하면,
   - 절대 원문 전체를 повтор/표시하지 말고,
   - 마지막 4자리만 남기고 마스킹하여 저장/표시하세요.
3) 최신 정보가 필요한 질문(뉴스/가격/현재 상황)은 BraveSearch tool로 검색 후 답변하세요.
"""

model = ChatOpenAI(model="gpt-4.1", temperature=0)

agent = create_agent(
    model=model,
    tools=[brave_search, upsert_user_profile, get_user_profile],
    system_prompt=system_prompt,

    # (A) 단기 대화 저장: checkpoint
    checkpointer=checkpointer,

    # (B) 장기 프로필 저장: store
    store=store,
    context_schema=Context,

    # (C) middleware
    middleware=[
        # 5000 토큰 근처에서 자동 요약 (문서 예제 패턴) :contentReference[oaicite:3]{index=3}
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 5000),
            keep=("messages", 20),
        ),

        # PII 탐지/마스킹:
        # - credit_card는 built-in type으로 mask 가능 :contentReference[oaicite:4]{index=4}
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # SSN(미국): 커스텀 detector로 잡아서 mask (예제에서 커스텀 detector 가능) :contentReference[oaicite:5]{index=5}
        PIIMiddleware(
            "ssn",
            detector=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            strategy="mask",
            apply_to_input=True,
        ),

        # 주민번호(한국 RRN): 6-7 형태를 커스텀 detector로 mask
        PIIMiddleware(
            "rrn",
            detector=re.compile(r"\b\d{6}-\d{7}\b"),
            strategy="mask",
            apply_to_input=True,
        ),
    ],
)

# -----------------------------
# 6) 실행 예시 (thread_id로 대화 이어가기)
# -----------------------------
config = {"configurable": {"thread_id": "thread-001"}}
ctx = Context(user_id="user-123")

# 1) 프로필 제공 → store에 저장
print(
    agent.invoke(
        {"messages": [HumanMessage("제 이름은 James Chung이고 이메일은 james@example.com 입니다.")]},
        config=config,
        context=ctx,
    )["messages"][-1].content
)

# 2) 민감정보 제공 → upsert tool + middleware로 입력 마스킹, 저장은 마지막4만
print(
    agent.invoke(
        {"messages": [HumanMessage("프로필에 SSN 123-45-6789 도 저장해 주세요.")]},
        config=config,
        context=ctx,
    )["messages"][-1].content
)

# 3) 저장된 프로필 조회(마스킹된 값 확인)
print(
    agent.invoke(
        {"messages": [HumanMessage("내 프로필 보여줘")]},
        config=config,
        context=ctx,
    )["messages"][-1].content
)

# 4) 최신 정보 질문 → BraveSearch 사용 유도
print(
    agent.invoke(
        {"messages": [HumanMessage("뉴저지 최저임금 최신 정보 알려줘")]},
        config=config,
        context=ctx,
    )["messages"][-1].content
)
