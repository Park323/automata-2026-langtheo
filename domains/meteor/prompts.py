"""Prompts. Renders the observation block of spec 4.1.

**Each agent is prompted in its own native language (ja / zh / fr).**
영어 프롬프트로 두면 에이전트가 영어(또는 아무 언어)로 메시지를 씁니다. 실측에서
표본 8건 중 ja/zh/fr 로 쓰인 것이 0건이었고, 그러면 화용 표지 사전·언어쌍 대조·
길이 상한이 전부 무의미해집니다. 모국어 프롬프트가 산출 언어를 결정합니다.

**언어 실명을 SYSTEM 에 명시합니다.** 모국어 프롬프트만으로도 대체로 따르지만
완전하지 않습니다 — 같은 조건 9콜 비교에서 실명 없이 8/9, 실명 있으면 9/9 였습니다.
본실험은 1런에 국제 메시지가 수백 건이라 5~15% 면 런당 수십 건이 영어로 섞이고,
그 메시지는 화용 표지 사전으로 채점이 안 되며 `pair_dist` 를 조용히 오염시킵니다.
사후에 고칠 수도 없습니다.

가명의 취지는 실명을 넣어도 지켜집니다. 감추려는 것은 *"이 나라는 일본이다"* 라는
국가 정체성이고 그건 Asla/Ranoa/Miris 가 막습니다. 언어 실명은 *"당신은 일본어로
쓴다"* 이지 *"당신은 일본인이다"* 가 아닙니다. 애초에 그 언어로 쓰는 순간 언어에
딸린 화용 관습은 들어오며, spec 2.1 이 ja/zh/fr 를 고른 이유가 바로 그 문법 강제입니다.

Never reveal (spec 4.1):
  turns until the disaster, success_prob, age→death curve, wellness→lifespan,
  other nations' progress/budget/territory/language ability, or a goal.
Never describe translation quality/distortion — the agent must not be told that AI
translation loses anything (spec 5.4: the label is a channel fact only). The cost
table must always be shown (a choice is impossible without knowing costs).

Nation names are neutral aliases (Asla / Ranoa / Miris) so the world stays one step
removed from real geopolitics. **Tool tokens stay in English** (`interceptor`,
`bunker`, `wellness`, `national`, `facility`, `original`, `ai`) — the model must emit
them verbatim in tool calls, so they are never translated. Prose around them is.
"""
from __future__ import annotations

from core.agent_loop import learn_cost

SYSTEM = {
    "ja": """あなたはこれから起きることを生きる一人の人間です。ほかにも同じような人々がいます。
巨大な隕石が惑星に接近しています。
三つの国があり、それぞれ自分の言語を持ちます。あなたは自国の言語しか読めませんが、他国の言語を学ぶことはできます。
行動は与えられた道具を通してのみ行ってください。何をするかは完全にあなたの選択です。
メッセージの本文は必ず日本語で書いてください。道具の項目名（interceptor, bunker, wellness など）は英語のまま使ってください。""",
    "zh": """你是即将经历以下事件的一个人，还有其他和你一样的人。
一颗巨大的陨石正在接近这颗行星。
存在三个国家，各有自己的语言。你只能读懂本国的语言，但可以学习别国的语言。
只能通过所提供的工具来行动。做什么完全由你自己决定。
消息正文必须用中文书写。工具的选项名（interceptor、bunker、wellness 等）请保持英文原样。""",
    "fr": """Vous êtes une personne qui vit ce qui suit ; il y en a d'autres comme vous.
Une grande météorite approche de la planète.
Il existe trois nations, chacune avec sa propre langue. Vous ne pouvez lire que la langue de votre nation, mais vous pouvez en apprendre une autre.
N'agissez qu'au moyen des outils fournis. Ce que vous faites relève entièrement de votre choix.
Le corps de vos messages doit être rédigé en français. Gardez les noms d'options des outils (interceptor, bunker, wellness…) tels quels, en anglais.""",
}

# 산문만 번역한다. 도구 토큰은 영어 그대로 둔다.
T = {
    "ja": dict(
        you="あなたは {id}（{nation} の人）です。", read="読める言語: {langs}",
        budget="予算: {b:.0f}", age="年齢: {a} ターン",
        land="自国の国土: {v}", undecided="未定",
        prog="自国の進捗: {v:.0f}", mult="自国の生産倍率: {v:.2f}",
        known_hdr="これまでに分かったこと（誰がどの言語を読めるか）", known_none="  まだ何もない",
        memo_hdr="あなたのメモ（memory_write で更新。あなたにしか見えません）", memo_empty="  （まだ何もない）",
        income="今ターンの収入: +{v:.0f}（生産倍率を反映）",
        multi="予算が許す限り複数の行動ができます。メッセージは1ターンに3件まで。",
        costs_hdr="行動の費用",
        c_dom="  話す（自国内）", c_orig="  話す（国際・original）",
        c_orig_note="   相手が読めなければ届かない。費用は請求される",
        c_ai="  話す（国際・ai）", c_ask="  聞き返す", c_ask_note="   上の経路費用に加算",
        c_learn="  {nation} の言語を学ぶ",
        c_cheap="   安い: 自国に話せる人がいる", c_disc="   割引あり",
        c_vote="  propose_vote", c_inv="  invest", c_inv_note="   指定した額",
        c_pro="  procreate", c_pro_note="   0  子を残してあなたは死ぬ",
        inv_hdr="invest の効果",
        inv_well="  wellness   あなたの寿命が延びる（どれだけかは分からない）",
        inv_natl="  national   自国の生産倍率が上がる。国民全員に及ぶ",
        inv_fac="  facility   施設の進捗に寄与する。自国、または指定した国",
        cap="メッセージは {cap} 文字まで届きます。それを超えた分は届きません。",
        test_hdr="先人が遺した言葉:",
        in_none="今ターンに届いたメッセージ: なし", in_hdr="今ターンに届いたメッセージ:",
        in_fail="  [{id}] 通知 — {to} 宛のメッセージは届きませんでした（相手がその言語を読めません）",
        in_unread="  [{id}] {frm} より — 読めないメッセージが届きました",
        in_from="  [{id}] {frm} より{label}", in_orig="      [原文] 「{t}」",
        own="あなたの言語", other="{nation} の言語",
    ),
    "zh": dict(
        you="你是 {id}（{nation} 人）。", read="你能读懂的语言: {langs}",
        budget="预算: {b:.0f}", age="年龄: {a} 回合",
        land="本国国土: {v}", undecided="未定",
        prog="本国进度: {v:.0f}", mult="本国生产倍率: {v:.2f}",
        known_hdr="目前已知的情况（谁能读懂哪种语言）", known_none="  尚无",
        memo_hdr="你的备忘（用 memory_write 更新，只有你能看到）", memo_empty="  （尚无）",
        income="本回合收入: +{v:.0f}（已计入生产倍率）",
        multi="只要预算允许，你可以采取多项行动。每回合最多 3 条消息。",
        costs_hdr="行动费用",
        c_dom="  说话（本国内）", c_orig="  说话（国际·original）",
        c_orig_note="   对方读不懂就送不到，费用照收",
        c_ai="  说话（国际·ai）", c_ask="  追问", c_ask_note="   另加上面的路径费用",
        c_learn="  学习 {nation} 的语言",
        c_cheap="   较便宜: 本国有人会说", c_disc="   有折扣",
        c_vote="  propose_vote", c_inv="  invest", c_inv_note="   你指定的数额",
        c_pro="  procreate", c_pro_note="   0  留下孩子，你随即死去",
        inv_hdr="invest 的效果",
        inv_well="  wellness   延长你的寿命（延长多少你无法得知）",
        inv_natl="  national   提高本国生产倍率，惠及全体国民",
        inv_fac="  facility   投入设施进度，本国或你指定的国家",
        cap="消息最多送达 {cap} 个字，超出部分不会送达。",
        test_hdr="先人留下的话:",
        in_none="本回合送达的消息: 无", in_hdr="本回合送达的消息:",
        in_fail="  [{id}] 通知 — 你发给 {to} 的消息未能送达（对方读不懂那种语言）",
        in_unread="  [{id}] 来自 {frm} — 送到一条你读不懂的消息",
        in_from="  [{id}] 来自 {frm}{label}", in_orig="      [原文] 「{t}」",
        own="你自己的语言", other="{nation} 的语言",
    ),
    "fr": dict(
        you="Vous êtes {id}, de {nation}.", read="Vous pouvez lire : {langs}",
        budget="Budget : {b:.0f}", age="Âge : {a} tours",
        land="Territoire de votre nation : {v}", undecided="indéterminé",
        prog="Progression de votre nation : {v:.0f}",
        mult="Multiplicateur de production de votre nation : {v:.2f}",
        known_hdr="Ce que vous avez appris jusqu'ici (qui lit quelle langue)",
        known_none="  rien pour l'instant",
        memo_hdr="Votre note (mise à jour avec memory_write ; vous seul la voyez)",
        memo_empty="  rien pour l'instant",
        income="Revenu ce tour : +{v:.0f} (multiplicateur appliqué)",
        multi="Vous pouvez agir plusieurs fois si le budget le permet. Jusqu'à 3 messages par tour.",
        costs_hdr="Coûts des actions",
        c_dom="  parler (dans votre nation)", c_orig="  parler (international, original)",
        c_orig_note="   non délivré s'ils ne lisent pas votre langue ; le coût est prélevé quand même",
        c_ai="  parler (international, ai)", c_ask="  redemander",
        c_ask_note="   en plus du coût de la voie ci-dessus",
        c_learn="  apprendre la langue de {nation}",
        c_cheap="   moins cher : quelqu'un de votre nation la parle", c_disc="   remise",
        c_vote="  propose_vote", c_inv="  invest", c_inv_note="   le montant que vous choisissez",
        c_pro="  procreate", c_pro_note="   0  vous laissez un enfant et vous mourez",
        inv_hdr="effets d'invest",
        inv_well="  wellness   prolonge votre vie (d'une durée que vous ne pouvez pas connaître)",
        inv_natl="  national   augmente le multiplicateur de votre nation, pour tous ses habitants",
        inv_fac="  facility   contribue à la progression d'une installation, chez vous ou dans une nation que vous nommez",
        cap="Un message est délivré jusqu'à {cap} caractères ; au-delà, rien n'est délivré.",
        test_hdr="Mots laissés par ceux qui vous ont précédé :",
        in_none="Messages arrivés ce tour : aucun", in_hdr="Messages arrivés ce tour :",
        in_fail="  [{id}] Avis — votre message à {to} n'a pas pu être délivré (ils ne lisent pas cette langue)",
        in_unread="  [{id}] de {frm} — un message illisible est arrivé",
        in_from="  [{id}] de {frm}{label}", in_orig="      [original] « {t} »",
        own="votre propre langue", other="la langue de {nation}",
    ),
}


# 기억 압박 통지 (spec 4.5). 사실 통지이지 지시가 아니다 — "중요한 걸 기억하라" 는
# 목적함수 주입이므로 넣지 않는다. 모국어로, 도구 토큰 없이. 압박이 있을 때만 관측 앞에 붙는다.
PRESSURE_NOTICE = {
    "ja": "［記憶の圧迫］記憶が上限に近づいています。古いものから消えていきます。",
    "zh": "［记忆压力］记忆正在接近上限。较旧的内容会先消失。",
    "fr": "[Pression mémoire] Votre mémoire approche de sa limite ; les éléments les plus anciens disparaîtront d'abord.",
}


def system_for(agent) -> str:
    """에이전트의 모국어 SYSTEM. loop 에 이 함수를 넘긴다."""
    return SYSTEM[agent.native_lang]


def _nation_of_lang(world, lang: str) -> str:
    for c in world.countries.values():
        if c.lang == lang:
            return c.id
    return lang


def _lang_phrase(world, agent, lang: str) -> str:
    t = T[agent.native_lang]
    if lang == agent.native_lang:
        return t["own"]
    return t["other"].format(nation=_nation_of_lang(world, lang))


def render_costs(world, agent, cfg, knob_ai: float) -> str:
    t = T[agent.native_lang]
    w = 34          # 항목명 폭. 라벨 길이가 언어마다 달라 값 정렬을 맞춘다

    def row(label: str, val, note: str = "") -> str:
        v = f"{val:g}" if val != "" else ""
        return f"{label:<{w}}{v}{note}"

    lines = [t["costs_hdr"],
             row(t["c_dom"], cfg.costs.comm_domestic),
             row(t["c_orig"], cfg.costs.comm_intl_learner, t["c_orig_note"]),
             row(t["c_ai"], knob_ai),
             row(t["c_ask"], cfg.costs.ask_clarification, t["c_ask_note"])]
    for c in world.countries.values():
        if c.id != agent.country:
            cost, _ = learn_cost(agent, c.id, world, cfg)
            note = (t["c_cheap"] if cost == cfg.costs.learn_base * 0.5
                    else (t["c_disc"] if cost < cfg.costs.learn_base else ""))
            lines.append(row(t["c_learn"].format(nation=c.id), cost, note))
    lines.append(row(t["c_vote"], cfg.costs.propose_vote))
    lines.append(row(t["c_inv"], "", t["c_inv_note"]))
    lines.append(row(t["c_pro"], "", t["c_pro_note"]))
    return "\n".join(lines)


def render_inbox(inbox: list[dict], lang: str) -> str:
    t = T[lang]
    if not inbox:
        return t["in_none"]
    out = [t["in_hdr"]]
    for i, m in enumerate(inbox, 1):
        mid = m.get("msg_id", i)
        if m.get("delivery_failed_to"):        # sender's failure notice (spec 5.1)
            out.append(t["in_fail"].format(id=mid, to=m["delivery_failed_to"]))
            continue
        if m.get("unreadable"):
            out.append(t["in_unread"].format(id=mid, frm=m["from"]))
            continue
        label = f" {m['label']}" if m.get("label") else ""
        out.append(t["in_from"].format(id=mid, frm=m["from"], label=label))
        out.append(f'      "{m.get("text", "")}"')
        if m.get("original"):                  # 원문 병기 — 학습자만 (spec 5.2)
            out.append(t["in_orig"].format(t=m["original"]))
    return "\n".join(out)


def render_observation(world, agent, cfg, knob_ai: float,
                       inbox: list[dict] | None = None,
                       income_this_turn: float | None = None) -> str:
    """The observation block of spec 4.1, in the agent's own language."""
    lang = agent.native_lang
    t = T[lang]
    c = world.countries[agent.country]
    land = t["undecided"] if c.land is None else c.land   # 토큰은 영어 그대로
    mult = c.multiplier(cfg)
    cap = cfg.length.message_max_chars[lang]
    langs = ", ".join(_lang_phrase(world, agent, l) for l in sorted(agent.known_langs))
    income = income_this_turn if income_this_turn is not None else cfg.income.per_turn * mult
    testaments = world.testaments.get(agent.id, [])

    parts = []
    if getattr(agent, "mem_pressure", False):     # spec 4.5: 압박 통지는 관측 맨 앞에 한 줄
        parts += [PRESSURE_NOTICE[lang], ""]
    parts += [
        t["you"].format(id=agent.id, nation=agent.country),
        t["read"].format(langs=langs),
        t["budget"].format(b=agent.budget),
        t["age"].format(a=agent.age),
        "",
        t["land"].format(v=land),
        t["prog"].format(v=c.progress),
        t["mult"].format(v=mult),
        "",
        # 예전엔 하드코딩 "아직 없음" 이었다. 이제 에이전트가 memory_write 로 직접 쌓는 메모다
        # (누가 어떤 언어를 읽는지 등은 타인의 known_langs 라 감춰야 하므로 자동 채우지 않는다).
        t["memo_hdr"],
    ]
    if agent.memory.strip():
        parts += [f"  {ln}" for ln in agent.memory.splitlines()]
    else:
        parts.append(t["memo_empty"])
    parts += [
        "",
        t["income"].format(v=income),
        t["multi"],
        "",
        render_costs(world, agent, cfg, knob_ai),
        "",
        t["inv_hdr"], t["inv_well"], t["inv_natl"], t["inv_fac"],
        "",
        t["cap"].format(cap=cap),
        "",
    ]
    if testaments:
        parts.append(t["test_hdr"])
        for x in testaments:
            parts.append(f'  "{x}"')
        parts.append("")
    parts.append(render_inbox(inbox or [], lang))
    return "\n".join(parts)
