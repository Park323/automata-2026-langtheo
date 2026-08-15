"""번역 호출. spec 5.2.

에이전트 호출과 **별개의 LLM 호출** 1회. L_발신 → L_수신 직역 (pivot 없음).
시스템 메시지는 출력 형식 계약 하나뿐. 번역 방식 지시는 발신 에이전트의
translate_instruction 을 그대로 쓰고, 없으면 중립 기본값 '번역하라' 만 쓴다.
"""
from __future__ import annotations

LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "fr": "French"}

# 출력 형식 계약 — 이것만 시스템이 붙일 수 있다 (spec 5.2). '간결/정확/자연' 금지.
SYSTEM_CONTRACT = (
    "You are a translation engine. Output ONLY the translated text.\n"
    "No explanation, no alternatives, no quotes, no notes."
)

# 중립 기본값. 여기에 '간결하게' 를 넣으면 즉시 위반 (자주 틀리는 곳 2).
# 이 세계에 한국어는 없다 — 개발 언어가 번역 프롬프트에 새면 번역기가 그 영향을 받고,
# 무엇보다 산출물에 없어야 할 언어가 파이프라인에 들어온다. 시스템 계약과 같은
# 영어로 두고, 품질 형용사('간결/정확/자연')는 절대 넣지 않는다.
DEFAULT_DIRECTIVE = ""          # 지시 없음이 가장 중립이다. 대상 언어만 사실로 붙는다


def build_prompt(dst_lang: str, text: str, instruction: str | None) -> str:
    """번역 사용자 프롬프트. 대상 언어(사실)만 붙이고, 방식 지시는 발신자 것/기본값."""
    directive = (instruction or "").strip() or DEFAULT_DIRECTIVE
    head = f"Translate to {LANG_NAME[dst_lang]}."
    if directive:
        head = f"{head} {directive}"
    return f"{head}\n\n{text}"


def translate(client, src_lang: str, dst_lang: str, text: str,
              instruction: str | None = None, meta: dict | None = None) -> dict:
    """직역 1회. 반환: {text, prompt, logprob_mean}.

    client 는 번역 전용 LLM (테스트에선 StubClient). temperature 는 낮게.
    meta 는 raw_calls.jsonl 문맥 (kind="translate" 등) — 지정하면 그대로 sink 에 붙는다.
    """
    prompt = build_prompt(dst_lang, text, instruction)
    resp = client.chat(
        [{"role": "system", "content": SYSTEM_CONTRACT},
         {"role": "user", "content": prompt}],
        temperature=0.2,
        meta=meta or {"kind": "translate", "src": src_lang, "dst": dst_lang},
    )
    msg = resp["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    # logprob 는 지원되면 남기고 아니면 null (백엔드를 이것 때문에 바꾸지 않는다)
    logprob_mean = None
    lp = (resp["choices"][0].get("logprobs") or {}).get("content")
    if lp:
        logprob_mean = sum(t["logprob"] for t in lp) / len(lp)
    return {"text": content, "prompt": prompt, "logprob_mean": logprob_mean}
