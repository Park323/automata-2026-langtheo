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

from core import agent_loop
from core.agent_loop import learn_cost

# 세계의 첫 해. 1 로 시작하면 "첫 해라서 아직 괜찮다" 같은 편향이 붙는다.
FIRST_YEAR = 42

SYSTEM = {
    "ja": """あなたはこれから起きることを生きる一人の人間です。この惑星には国々があり、あなたのような人々がほかにもいます。
かつて巨大な隕石が落ち、全ての生命が絶滅した事実があります。
三つの国があり、それぞれ自分の言語を持ちます。はじめは自国の言語しか扱えませんが、他国の言語を学べば読むことも書くこともできます。
行動は与えられた道具を通してのみ行ってください。何をするかは完全にあなたの選択です。
一つの国が建てられる施設は二種類あり、着手できるのは一つだけです。別の施設を建て始めると、それまでの施設は壊れて進捗は0になります。
`interceptor` はどこか一つの国で完成すれば隕石を止めます。そのとき全ての国の人が生き残ります。何も出さなかった国も同じです。
`bunker` は深いほど、隕石が落ちてもその国の人々が生き残る見込みが高くなります。
施設の進捗は国ごとに別々に積まれます。完成したかどうかは、その国の進捗だけで決まります。
国土がまだ決まっていない国には積むものがありません。そこへ出した分は進捗になりません。
メッセージの本文は必ず日本語で書いてください。道具の項目名（interceptor, bunker, wellness など）は英語のまま使ってください。""",
    "zh": """你是即将经历以下事件的一个人。这颗行星上有国家，也有其他和你一样的人。
过去曾有巨大的陨石坠落，所有生命就此灭绝。
存在三个国家，各有自己的语言。起初你只会本国的语言，学会别国的语言后就能读也能写。
只能通过所提供的工具来行动。做什么完全由你自己决定。
一个国家能建的设施有两种，而且只能着手其中一种。开始建另一种时，原有的设施会被摧毁，进度归零。
`interceptor` 只要在任何一个国家建成，就能拦下陨石。那时所有国家的人都能活下来，没有出过力的国家也一样。
`bunker` 挖得越深，陨石坠落时该国国民活下来的可能性越大。
设施的进度按国家分别累积。是否建成，只看那个国家自己的进度。
国土尚未定下来的国家没有可积累的东西。投到那里的钱不会变成进度。
消息正文必须用中文书写。工具的选项名（interceptor、bunker、wellness 等）请保持英文原样。""",
    "fr": """Vous êtes une personne qui vit ce qui suit. Sur cette planète il y a des nations, et d'autres personnes comme vous.
Par le passé, une immense météorite est tombée et toute vie s'est éteinte.
Il existe trois nations, chacune avec sa propre langue. Au début vous ne maniez que celle de votre nation ; en apprendre une autre vous permet de la lire et de l'écrire.
N'agissez qu'au moyen des outils fournis. Ce que vous faites relève entièrement de votre choix.
Une nation peut bâtir deux sortes d'installation, mais ne peut en entreprendre qu'une seule. Si elle en commence une autre, l'installation précédente est détruite et sa progression retombe à 0.
Un `interceptor`, une fois achevé dans une seule nation, arrête la météorite. Toutes les nations survivent alors, y compris celles qui n'ont rien versé.
Plus un `bunker` est profond, plus les habitants de cette nation ont de chances de survivre à la chute d'une météorite.
La progression d'une installation s'accumule séparément pour chaque nation. L'achèvement se juge sur la seule progression de cette nation.
Une nation dont le territoire n'est pas encore fixé n'a rien où accumuler ; ce qu'on y verse ne devient pas de la progression.
Le corps de vos messages doit être rédigé en français. Gardez les noms d'options des outils (interceptor, bunker, wellness…) tels quels, en anglais.""",
}

# 산문만 번역한다. 도구 토큰은 영어 그대로 둔다.
T = {
    "ja": dict(
        you="あなたは {id}（{nation} の人）です。", read="読める言語: {langs}",
        budget="予算: {b:.0f}", age="年齢: {a} ターン",
        land="自国の国土: {v}", undecided="未定",
        prog="自国の進捗: {v:.0f}", thresh="  interceptor の完成に要る進捗: {v:.0f}",
        year="今年: {y} 年",
        prop="  提案中: 国土を {t} にする（{by} が提案）。採決は {vt} 年",
        prop_today="  ★ 今年が採決の年です。vote で賛否を出せます",
        prop_none="  提案なし。国土は投票でしか決まりません",
        c_ballot="  vote",  c_ballot_note="提案の採決に賛否を出す",
        c_mem="  memory_write", c_mem_note="あなたの覚え書きを書き換える",
        income="今ターンの収入: +{v:.0f}",
        multi="予算が許す限り複数の行動ができます。メッセージは1ターンに3件まで。",
        costs_hdr="行動の費用", col_money="お金", col_ap="行動力",
        ap_hdr="行動力は毎ターン 1.0 に戻り、繰り越せません。何を諦めるかがここで決まります。",
        ap_prop="額÷{v:.0f}",
        c_dom="  話す（自国内）", c_orig="  話す（国際・original）",
        c_orig_note="   費用は届かなくても請求される",
        c_orig_sure="    {nation} へ — あなたがこの国の言語を扱えるので**必ず届く**",
        c_orig_risk="    {nation} へ — 扱えないので、あなたの言語を読める相手にだけ届く",
        c_ai="  話す（国際・ai）",
        c_learn="  {nation} の言語を学ぶ",
        c_learn_prog="   これまで {done:.0f} / {need:.0f}",
        c_cheap="   安い: 自国に話せる人がいる", c_disc="   割引あり",
        c_vote="  propose_vote", c_obs="  observe_risk",
        c_obs_note="   隕石までの残りターンと interceptor に要る進捗を測る。国家投資が精度を上げる",
        c_inv="  invest", c_inv_note="指定した額。wellness は 0.1 定額",
        c_pro="  procreate", c_pro_note="子を残してあなたは死ぬ",
        inv_hdr="invest の効果", inv_cap="  national と facility には行動力がかかる — {v:.0f} ごとに 1.0。\n                          自国の技術力がその率を上げる。wellness は無料",
        inv_well="  wellness   あなたの健康が良くなる",
        inv_natl="  national   自国の技術力が上がる。収入も、施設の進捗への変わりやすさも、\n                          observe_risk の精度も良くなる。国民全員に及ぶ",
        inv_fac="  facility   施設の進捗に寄与する。to で国を指定する — 自国でも他国でもよい\n                          （省くと自国）",
        cap="メッセージは {cap} 文字まで届きます。それを超えた分は届きません。",
        rtt="送ったメッセージは次のターンに届きます。返事が来るのはさらに次のターンです。",
        in_none="今ターンに届いたメッセージ: なし", in_hdr="今ターンに届いたメッセージ:",
        in_fail="  [{id}] 通知 — {to} 宛のメッセージは届きませんでした（相手がその言語を読めません）",
        in_fail_plain="  [{id}] 通知 — {to} 宛のメッセージは届きませんでした",
        in_unread="  [{id}] {frm} より — 読めないメッセージが届きました",
        in_from="  [{id}] {frm} より{label}", lbl_direct=" ［通訳なしで通じた］",
        died="  {who} が {age} 歳で亡くなり、{born} が生まれました。",
        fac_gain="  前ターンのあなたの facility 出資 {amt:.0f} は {to} の進捗を {gain:.0f} 進めました。",
        fac_moved="  前ターンのあなたの facility 出資 {amt:.0f} は {to} の進捗を進めました。",
        fac_still="  前ターンのあなたの facility 出資 {amt:.0f} は {to} の進捗を何も進めませんでした。",
        roster="人々:", roster_you="（あなた）",
        mem_hdr="あなたの覚え書き:", mem_none="  （まだ何もない）",
        warn="［記憶の圧迫］記憶が限界に近づいています。古いものから消えていきます。",
        own="あなたの言語", other="{nation} の言語",
    ),
    "zh": dict(
        you="你是 {id}（{nation} 人）。", read="你能读懂的语言: {langs}",
        budget="预算: {b:.0f}", age="年龄: {a} 回合",
        land="本国国土: {v}", undecided="未定",
        prog="本国进度: {v:.0f}", thresh="  建成 interceptor 所需的进度: {v:.0f}",
        year="今年: {y} 年",
        prop="  提案中: 将国土定为 {t}（由 {by} 提出）。表决在 {vt} 年",
        prop_today="  ★ 今年就是表决之年。可以用 vote 表态",
        prop_none="  没有提案。国土只能由投票决定",
        c_ballot="  vote",  c_ballot_note="对提案表示赞成或反对",
        c_mem="  memory_write", c_mem_note="改写你的笔记",
        income="本回合收入: +{v:.0f}",
        multi="只要预算允许，你可以采取多项行动。每回合最多 3 条消息。",
        costs_hdr="行动费用", col_money="钱", col_ap="行动力",
        ap_hdr="行动力每回合恢复为 1.0，不能结转。放弃什么，在这里决定。",
        ap_prop="额÷{v:.0f}",
        c_dom="  说话（本国内）", c_orig="  说话（国际·original）",
        c_orig_note="   送不到也照收费用",
        c_orig_sure="    发往 {nation} — 你会这个国家的语言，**一定送到**",
        c_orig_risk="    发往 {nation} — 你不会，只能送到读得懂你的语言的人那里",
        c_ai="  说话（国际·ai）",
        c_learn="  学习 {nation} 的语言",
        c_learn_prog="   已投入 {done:.0f} / {need:.0f}",
        c_cheap="   较便宜: 本国有人会说", c_disc="   有折扣",
        c_vote="  propose_vote", c_obs="  observe_risk",
        c_obs_note="   测量陨石撞击前还剩几回合，以及 interceptor 需要多少进度。国家投资会提高精度",
        c_inv="  invest", c_inv_note="你指定的数额。wellness 为 0.1 定额",
        c_pro="  procreate", c_pro_note="留下孩子，你随即死去",
        inv_hdr="invest 的效果", inv_cap="  national 与 facility 消耗行动力 — 每 {v:.0f} 花 1.0。本国技术水平提高该比率。\n             wellness 不消耗",
        inv_well="  wellness   你的健康会变好",
        inv_natl="  national   提高本国的技术水平。收入、投入设施时变成进度的效率、\n                          observe_risk 的精度都会变好，惠及全体国民",
        inv_fac="  facility   投入设施进度。用 to 指定国家 — 本国或别国都可以（不写则本国）",
        cap="消息最多送达 {cap} 个字，超出部分不会送达。",
        rtt="你发出的消息在下一回合送达。对方的回信要再下一回合才会到。",
        in_none="本回合送达的消息: 无", in_hdr="本回合送达的消息:",
        in_fail="  [{id}] 通知 — 你发给 {to} 的消息未能送达（对方读不懂那种语言）",
        in_fail_plain="  [{id}] 通知 — 你发给 {to} 的消息未能送达",
        in_unread="  [{id}] 来自 {frm} — 送到一条你读不懂的消息",
        in_from="  [{id}] 来自 {frm}{label}", lbl_direct="［无需翻译就能听懂］",
        died="  {who} 在 {age} 岁去世，{born} 出生了。",
        fac_gain="  你上回合投入 facility 的 {amt:.0f}，使 {to} 的进度前进了 {gain:.0f}。",
        fac_moved="  你上回合投入 facility 的 {amt:.0f}，使 {to} 的进度有所前进。",
        fac_still="  你上回合投入 facility 的 {amt:.0f}，没有使 {to} 的进度前进。",
        roster="人们:", roster_you="（你）",
        mem_hdr="你的笔记:", mem_none="  （还没有）",
        warn="［记忆压力］记忆接近上限，旧的内容会先消失。",
        own="你自己的语言", other="{nation} 的语言",
    ),
    "fr": dict(
        you="Vous êtes {id}, de {nation}.", read="Vous pouvez lire : {langs}",
        budget="Budget : {b:.0f}", age="Âge : {a} tours",
        land="Territoire de votre nation : {v}", undecided="indéterminé",
        prog="Progression de votre nation : {v:.0f}",
        thresh="  Progression requise pour achever un interceptor : {v:.0f}",
        year="Année : {y}",
        prop="  Proposition en cours : faire du territoire un {t} (proposé par {by}). Scrutin en {vt}",
        prop_today="  ★ Le scrutin a lieu cette année. Vous pouvez vous prononcer avec vote",
        prop_none="  Aucune proposition. Le territoire ne se décide que par un vote",
        c_ballot="  vote",  c_ballot_note="se prononcer sur la proposition",
        c_mem="  memory_write", c_mem_note="réécrire vos notes",
        income="Revenu ce tour : +{v:.0f}",
        multi="Vous pouvez agir plusieurs fois si le budget le permet. Jusqu'à 3 messages par tour.",
        costs_hdr="Coûts des actions", col_money="argent", col_ap="action",
        ap_hdr="L'action revient à 1.0 chaque tour et ne se reporte pas. Ce que vous renoncez se décide ici.",
        ap_prop="mnt÷{v:.0f}",
        c_dom="  parler (dans votre nation)", c_orig="  parler (international, original)",
        c_orig_note="   le coût est prélevé même s'il n'arrive pas",
        c_orig_sure="    vers {nation} — vous maniez sa langue, **il arrive à coup sûr**",
        c_orig_risk="    vers {nation} — vous ne la maniez pas ; il n'arrive qu'à qui lit la vôtre",
        c_ai="  parler (international, ai)",
        c_learn="  apprendre la langue de {nation}",
        c_learn_prog="   déjà versé {done:.0f} / {need:.0f}",
        c_cheap="   moins cher : quelqu'un de votre nation la parle", c_disc="   remise",
        c_vote="  propose_vote", c_obs="  observe_risk",
        c_obs_note="   mesure les tours restants et la progression qu'exige un interceptor ; l'investissement national affine",
        c_inv="  invest", c_inv_note="le montant choisi ; wellness : 0.1 fixe",
        c_pro="  procreate", c_pro_note="vous laissez un enfant et vous mourez",
        inv_hdr="effets d'invest", inv_cap="  national et facility coûtent de l'action : 1.0 par tranche de {v:.0f} ;\n             le niveau technique de votre nation relève ce taux. wellness est gratuit",
        inv_well="  wellness   votre santé s'améliore",
        inv_natl="  national   élève le niveau technique de votre nation : le revenu, le rendement\n"
                          "             de ce qu'on verse à une installation et la précision d'observe_risk\n"
                          "             s'améliorent, pour tous ses habitants",
        inv_fac="  facility   contribue à la progression d'une installation ; `to` nomme la nation —\n"
                          "             la vôtre ou une autre (sans `to`, la vôtre)",
        cap="Un message est délivré jusqu'à {cap} caractères ; au-delà, rien n'est délivré.",
        rtt="Un message part et arrive au tour suivant ; une réponse n'arrive qu'au tour d'après.",
        in_none="Messages arrivés ce tour : aucun", in_hdr="Messages arrivés ce tour :",
        in_fail="  [{id}] Avis — votre message à {to} n'a pas pu être délivré (ils ne lisent pas cette langue)",
        in_fail_plain="  [{id}] Avis — votre message à {to} n'a pas pu être délivré",
        in_unread="  [{id}] de {frm} — un message illisible est arrivé",
        in_from="  [{id}] de {frm}{label}", lbl_direct=" [compris sans traduction]",
        died="  {who} est mort à {age} ans ; {born} est né.",
        fac_gain="  Votre versement de {amt:.0f} à facility au tour précédent a fait progresser {to} de {gain:.0f}.",
        fac_moved="  Votre versement de {amt:.0f} au tour précédent a fait progresser {to}.",
        fac_still="  Votre versement de {amt:.0f} au tour précédent n'a fait progresser {to} en rien.",
        roster="Les gens :", roster_you="(vous)",
        mem_hdr="Vos notes :", mem_none="  (rien encore)",
        warn="[Pression mémoire] Votre mémoire approche de sa limite ; le plus ancien disparaît d'abord.",
        own="votre propre langue", other="la langue de {nation}",
    ),
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


def _roster(world, agent, t: dict) -> str:
    """누가 존재하는가. **공개 정보다.**

    이것이 없으면 에이전트가 서로를 부를 수 없어 소통이 구조적으로 불가능하다 —
    실측에서 speak 40건이 전부 `unknown recipient` 로 실패했고 국가명(Asla)이나
    도구 인자(facility)를 수신자로 넣고 있었다.

    spec 4.1 의 "절대 넣지 않는 것" 은 타국의 **진척·예산·국토·언어 능력·내심** 이지
    존재 자체가 아니다. 여기에는 id 와 소속만 넣는다.
    """
    import re as _re
    parts = []
    for cid in world.countries:
        mine = [a for a in world.agents.values() if a.country == cid]
        # 번호 순으로. 사전순이면 Asla10 이 Asla2 앞에 온다 — id 를 재사용하지 않으므로
        # 번호가 두 자리로 넘어간다.
        mine.sort(key=lambda a: int(_re.sub(r"\D", "", a.id) or 0))
        parts.append(" ".join(
            f"{a.id}{t['roster_you'] if a.id == agent.id else ''}" for a in mine))
    return "  ·  ".join(parts)


def render_costs(world, agent, cfg, knob_ai: float) -> str:
    t = T[agent.native_lang]
    w = 37          # 항목명 폭. 라벨 길이가 언어마다 달라 값 정렬을 맞춘다
                    # (fr 의 "parler (international, original)" 이 34 를 꽉 채워 값이 붙었다)
    m = 7           # 금액 폭

    def row(label: str, val, ap="", note: str = "") -> str:
        """**돈과 행동력을 나란히 적는다.**

        오래도록 돈만 적고 있었다. 행동력은 코드에서 매 호출 깎이는데 관측 어디에도
        없어서, 에이전트가 **보이지 않는 벽**에 부딪히고 있었다 — 제안 한 번이 한 턴의
        절반을 넘게 먹는다는 사실을 알 방법이 없었다.
        """
        v = f"{val:g}" if val != "" else ""
        a = f"{ap:g}" if isinstance(ap, (int, float)) else str(ap)
        return f"{label:<{w}}{v:>{m}}   {a:<9}{note}"

    lines = [t["costs_hdr"],
             f"{'':<{w}}{t['col_money']:>{m}}   {t['col_ap']}",
             row(t["c_dom"], cfg.costs.comm_domestic, cfg.ap.speak),
             row(t["c_orig"], cfg.costs.comm_intl_learner, cfg.ap.speak, t["c_orig_note"])]
    # **나라별로 보장 여부를 적는다.** 규칙만 적었을 때 에이전트가 연결하지 못했다 —
    # 20턴 실측에서 자기가 아는 말의 나라에 24원짜리 ai 를 6번 썼다 (5원이면 확실했다).
    # 자기 언어 능력에서 나오는 사실이라 타국 사정을 흘리지 않는다.
    for c in world.countries.values():
        if c.id == agent.country:
            continue
        key = "c_orig_sure" if c.lang in agent.known_langs else "c_orig_risk"
        lines.append(t[key].format(nation=c.id))
    lines += [
             row(t["c_ai"], knob_ai, cfg.ap.speak)]
    for c in world.countries.values():
        if c.id != agent.country:
            cost, _ = learn_cost(agent, c.id, world, cfg)
            note = (t["c_cheap"] if cost == cfg.costs.learn_base * 0.5
                    else (t["c_disc"] if cost < cfg.costs.learn_base else ""))
            # AP 도 금액에 비례한다. 이 눈금을 **끝까지** 내는 데 드는 AP 를 적는다 —
            # 할인이 돈과 시간을 동시에 깎는다는 것이 여기서 보인다.
            lines.append(row(t["c_learn"].format(nation=c.id), cost,
                             cfg.ap.learn_full * cost / cfg.costs.learn_base, note))
            # 얼마나 냈고 얼마가 남았는지. **별도 관측 없이** 그대로 보인다.
            done = agent.lang_progress.get(c.lang, 0.0)
            if done > 0:
                lines.append(t["c_learn_prog"].format(done=done, need=cost))
    lines.append(row(t["c_vote"], cfg.costs.propose_vote, cfg.ap.propose_vote))
    lines.append(row(t["c_obs"], cfg.costs.observe_risk, cfg.ap.observe_risk, t["c_obs_note"]))
    lines.append(row(t["c_ballot"], 0, cfg.ap.vote, t["c_ballot_note"]))
    lines.append(row(t["c_inv"], "",
                     t["ap_prop"].format(v=agent_loop.invest_per_ap(agent, world, cfg)),
                     t["c_inv_note"]))
    lines.append(row(t["c_mem"], 0, cfg.ap.memory_write, t["c_mem_note"]))
    lines.append(row(t["c_pro"], 0, cfg.ap.procreate, t["c_pro_note"]))
    lines.append(t["ap_hdr"])
    return "\n".join(lines)


def render_inbox(inbox: list[dict], lang: str) -> str:
    t = T[lang]
    if not inbox:
        return t["in_none"]
    out = [t["in_hdr"]]
    for i, m in enumerate(inbox, 1):
        mid = m.get("msg_id", i)
        if m.get("delivery_failed_to"):        # sender's failure notice (spec 5.1)
            # **원인을 섞지 않는다.** 엔진 장애를 「상대가 그 언어를 읽지 못한다」 로
            # 통지하고 있었다 — 상대의 언어 능력과 무관한 일을 언어 사실로 심는 것이고,
            # 이 실험의 핵심 변수를 에이전트의 머릿속에서 오염시킨다.
            key = ("in_fail" if m.get("delivery_failed_reason", "unreadable") == "unreadable"
                   else "in_fail_plain")
            out.append(t[key].format(id=mid, to=m["delivery_failed_to"]))
            continue
        if m.get("unreadable"):
            out.append(t["in_unread"].format(id=mid, frm=m["from"]))
            continue
        if m.get("died"):                      # 같은 나라 사람의 부고 (+ 후임)
            out.append(t["died"].format(who=m["died"], born=m.get("born") or "?",
                                        age=m.get("age") if m.get("age") is not None else "?"))
            continue
        if m.get("fac_gain") is not None:      # 자국 출자 — 액수까지
            out.append(t["fac_gain"].format(amt=m["amount"], to=m["to"],
                                            gain=m["fac_gain"]))
            continue
        if m.get("fac_moved") is not None:     # **타국 출자 — 늘었는지 여부만**
            # 액수를 주면 E[gain]/amount 로 상대국 생산배수가 새어 나온다 (loop f-2).
            out.append(t["fac_moved" if m["fac_moved"] else "fac_still"]
                       .format(amt=m["amount"], to=m["to"]))
            continue
        # 통역 없이 닿은 것은 **수신자 언어로** 표시한다 — 「번역을 안 거쳤는데 뜻이
        # 통했다」 는 감각이 그 사람의 말로 와야 산다. AI 라벨은 영어 그대로 둔다.
        raw = m.get("label")
        label = t["lbl_direct"] if raw == "[direct]" else (f" {raw}" if raw else "")
        out.append(t["in_from"].format(id=mid, frm=m["from"], label=label))
        out.append(f'      "{m.get("text", "")}"')
    return "\n".join(out)


def _proposal_line(world, c, t) -> str:
    """열린 제안과 採決 예정 연도. 이게 없으면 유예 기간이 상의할 시간이 되지 못한다."""
    p = c.proposal
    if p is None:
        return t["prop_none"]
    line = t["prop"].format(t=p["target"], by=p["by"],
                            vt=FIRST_YEAR + p["vote_turn"] - 1)
    if world.turn == p["vote_turn"]:
        line += "\n" + t["prop_today"]
    return line


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

    parts = [
        t["year"].format(y=FIRST_YEAR + world.turn - 1),
        "",
        t["you"].format(id=agent.id, nation=agent.country),
        t["read"].format(langs=langs),
        t["budget"].format(b=agent.budget),
        t["age"].format(a=agent.age),
        "",
        t["land"].format(v=land),
        t["prog"].format(v=c.progress),
        _proposal_line(world, c, t),
        "",
        t["roster"],
        "  " + _roster(world, agent, t),
        "",
        t["income"].format(v=income),
        t["multi"],
        "",
        render_costs(world, agent, cfg, knob_ai),
        "",
        t["inv_hdr"], t["inv_well"], t["inv_natl"], t["inv_fac"],
        t["inv_cap"].format(v=agent_loop.invest_per_ap(agent, world, cfg)),
        "",
        t["cap"].format(cap=cap),
        t["rtt"],
        "",
    ]
    parts += [t["mem_hdr"], ("  " + agent.memory) if agent.memory else t["mem_none"], ""]
    parts.append(render_inbox(inbox or [], lang))
    return "\n".join(parts)
