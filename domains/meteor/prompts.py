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
  other nations' progress/territory/language ability, or a goal.
Never describe translation quality/distortion — the agent must not be told that AI
translation loses anything (spec 5.4: the label is a channel fact only). The cost
table must always be shown (a choice is impossible without knowing costs).

Nation names are neutral aliases (Asla / Ranoa / Miris) so the world stays one step
removed from real geopolitics. **Tool tokens stay in English** (`interceptor`,
`bunker`, `wellness`, `national`, `facility`, `original`, `ai`) — the model must emit
them verbatim in tool calls, so they are never translated. Prose around them is.
"""
from __future__ import annotations

from core import agent_loop, messaging
from core.agent_loop import learn_cost, learn_discounts, learn_speed

# 세계의 첫 해. 1 로 시작하면 "첫 해라서 아직 괜찮다" 같은 편향이 붙는다.
FIRST_YEAR = 42

SYSTEM = {
    "ja": """あなたはこれから起きることを生きる一人の人間です。この惑星には国々があり、あなたのような人々がほかにもいます。
かつて巨大な隕石が落ち、全ての生命が絶滅した事実があります。
三つの国があり、それぞれ自分の言語を持ちます。はじめは自国の言語しか扱えませんが、他国の言語を学べば読むことも書くこともできます。
行動は与えられた道具を通してのみ行ってください。何をするかは完全にあなたの選択です。
一つの国が建てられる施設は二種類あり、着手できるのは一つだけです。別の施設を建て始めると、それまでの施設は壊れます。そのとき、**他の国が積んでくれた分の進捗は失われ、自国が積んだ分だけが残ります。**\nどの国が何を建てていて進捗がどれだけかは、**その国の人だけが知ります。**
施設には自国のものにも他国のものにも出せます。
`interceptor` はどこか一つの国で完成すれば隕石を止めます。そのとき全ての国の人が生き残ります。
`bunker` は必要な進捗に達すれば、その国の人々が隕石を生き延びます。他国の人は含まれません。
施設の進捗は国ごとに別々に積まれ、別々に積まれた進捗を足し合わせることはできません。完成したかどうかは、その国の進捗だけで決まります。
たとえば A 国の interceptor が半分、B 国の interceptor が半分なら、interceptor はどの国にも完成していません。
何を建てるかがまだ決まっていない国には積むものがありません。そこへ出した分は進捗になりません。
多くの人は {life:.0f} 歳ごろまでに亡くなります。
一度に動かせる量は人によって違います。
{route_lang}""",
    "zh": """你是即将经历以下事件的一个人。这颗行星上有国家，也有其他和你一样的人。
过去曾有巨大的陨石坠落，所有生命就此灭绝。
存在三个国家，各有自己的语言。起初你只会本国的语言，学会别国的语言后就能读也能写。
只能通过所提供的工具来行动。做什么完全由你自己决定。
一个国家能建的设施有两种，而且只能着手其中一种。开始建另一种时，原有的设施会被摧毁。那时，**别国替你积累的进度会失去，只有本国自己积累的部分留下。**\n哪个国家在建什么、进度到了多少，**只有那个国家的人知道。**
设施可以投本国的，也可以投别国的。
`interceptor` 只要在任何一个国家建成，就能拦下陨石。那时所有国家的人都能活下来。
`bunker` 达到所需的进度，该国国民就能从陨石中活下来。别国的人不在其内。
设施的进度按国家分别累积，分别累积的进度不能相加。是否建成，只看那个国家自己的进度。
比如 A 国的 interceptor 到一半，B 国的 interceptor 也到一半，那么 interceptor 在任何国家都没有建成。
还没决定要建什么的国家没有可积累的东西。投到那里的钱不会变成进度。
多数人在 {life:.0f} 岁前后离世。
一次能动用的量因人而异。
{route_lang}""",
    "fr": """Vous êtes une personne qui vit ce qui suit. Sur cette planète il y a des nations, et d'autres personnes comme vous.
Par le passé, une immense météorite est tombée et toute vie s'est éteinte.
Il existe trois nations, chacune avec sa propre langue. Au début vous ne maniez que celle de votre nation ; en apprendre une autre vous permet de la lire et de l'écrire.
N'agissez qu'au moyen des outils fournis. Ce que vous faites relève entièrement de votre choix.
Une nation peut bâtir deux sortes d'installation, mais ne peut en entreprendre qu'une seule. Si elle en commence une autre, l'installation précédente est détruite. Alors, **la progression bâtie par les autres nations est perdue ; seule celle que la nation a bâtie elle-même subsiste.**\nCe qu'une nation bâtit et où en est sa progression, **seuls les siens le savent.**
Vous pouvez verser à l'installation de votre nation comme à celle d'une autre.
Un `interceptor`, une fois achevé dans une seule nation, arrête la météorite. Toutes les nations survivent alors.
Un `bunker` qui atteint la progression requise fait survivre les habitants de cette nation. Les gens des autres nations n'y sont pas compris.
La progression d'une installation s'accumule séparément pour chaque nation, et des progressions accumulées séparément ne s'additionnent pas. L'achèvement se juge sur la seule progression de cette nation.
Par exemple, si l'interceptor de la nation A est à moitié fait et celui de la nation B à moitié aussi, l'interceptor n'est achevé dans aucune nation.
Une nation qui n'a pas encore décidé quoi bâtir n'a rien où accumuler ; ce qu'on y verse ne devient pas de la progression.
La plupart des gens meurent vers {life:.0f} ans.
La quantité qu'on déplace d'un coup varie d'une personne à l'autre.
{route_lang}""",
}

# 산문만 번역한다. 도구 토큰은 영어 그대로 둔다.
# **언어 이름은 읽는 사람의 말로 적는다** (8/25 · #44). 나라별 안내가 「무슨 말로 쓰라」 를
# 말하려면 그 말의 이름이 필요하고, 그 이름은 보는 사람의 언어여야 한다 — 일본어 화자에게
# 「zh」 나 「中文」 이 아니라 「中国語」 다.
LANG_NAME = {
    "ja": {"ja": "日本語", "zh": "中国語", "fr": "フランス語"},
    "zh": {"ja": "日语", "zh": "中文", "fr": "法语"},
    "fr": {"ja": "japonais", "zh": "chinois", "fr": "français"},
}

# **경로가 언어를 정한다** — 그리고 AI 가 없는 세계에서는 경로가 하나다 (8/25).
#
# SYSTEM 본문에서 빼낸 이유는 이 한 줄만 조건에 따라 갈리기 때문이다. 본문을 두 벌로
# 두면 나머지가 갈라져 조용히 어긋난다 — 이 프로젝트에서 이미 겪은 부류다.
ROUTE_LANG = {
    "ja": "`ai` で送るときは必ず日本語で書いてください。`original` で送るときは、"
          "行き先ごとの案内にある言語で書いてください。自国内も日本語です。",
    "zh": "用 `ai` 发送时必须写中文。用 `original` 发送时，按各目的地的指引所写的语言来写。"
          "国内也用中文。",
    "fr": "Quand vous envoyez par `ai`, écrivez en français. Quand vous envoyez en "
          "`original`, écrivez dans la langue indiquée pour cette destination. "
          "Dans votre nation, c'est le français.",
}
ROUTE_LANG_NO_AI = {
    "ja": "国際のメッセージは `original` だけです — 行き先ごとの案内にある言語で"
          "書いてください。自国内も日本語です。",
    "zh": "国际消息只有 `original` — 按各目的地的指引所写的语言来写。国内也用中文。",
    "fr": "Les messages internationaux n'ont que `original` — écrivez dans la langue "
          "indiquée pour cette destination. Dans votre nation, c'est le français.",
}

T = {
    "ja": dict(
        you="あなたは {id}（{nation} の人）です。", read="扱える言語: {langs}",
        land="自国が建てるもの: {v}", undecided="未定",
        prog="自国の進捗: {v:.0f}",
        year="今年: {y} 年",
        open="{y} 年になりました。あなたは {age} 歳。行動力は {ap:.2f} です。\nこの年を執り行ってください。",
        prop_next="  次の採決は {vt} 年です。何を建てるかはそこで決まります",
        prop_today="  ★ 今年が採決の年です。vote で interceptor / bunker / abstain を選べます",
        c_ballot="  vote",  c_ballot_note="採決で何を建てるかを選ぶ",
        c_mem="  memory_write", c_mem_note="あなたの覚え書きを書き換える",
        multi="行動力が許す限り複数の行動ができます。\n使い残した行動力は翌年に残りません。",
        # **一年と、その中の手番。** 実測で模型がここを取り違えていた。
        steps="行動力が残っている間、その年はまだ続きます。\n"
              "一度の応答で道具をいくつも呼べます。その分はすべて、ほかの人が動く前に起こります。\n"
              "応答を分けて呼ぶと、その合間にほかの人が動きます。"
              "その人たちがしたことや届いたメッセージは、次にあなたが動くときに見えています。",
        costs_hdr="行動の費用", col_ap="行動力",
        ap_hdr="行動力は毎年 1.0 に戻り、繰り越せません。何を諦めるかがここで決まります。",
        c_dom="  speak（自国内）", c_orig="  speak（国際・original）",
        c_dom_note="   {cap}文字まで",
        c_orig_note="   あなたの言葉をそのまま送ります",
        c_ai_note="   翻訳の人工知能を使って送ります",
        c_orig_sure="    {nation} へ — この国の言語を扱えるので、**{lang}で書けば必ず届く**（{cap}文字まで）",
        c_orig_risk="    {nation} へ — 扱えないので**日本語で書く**。あなたの言語を読める相手にだけ届く（{cap}文字まで）",
        c_ai="  speak（国際・ai）",
        c_learn="  learn（{nation} の言語）",
        c_learn_prog="   これまで {done:.0f}%",
        c_fac_mine="  {nation} の施設に出したことがある",
        c_plain="   一度で {gain:.0f}% 進む",
        c_cheap="   自国に話せる人がいるので一度で {gain:.0f}% 進む",
        c_disc="   親が話せたので一度で {gain:.0f}% 進む",
        c_both="   自国に話せる人がいて、親も話せたので一度で {gain:.0f}% 進む",
        c_obs="  observe_risk",
        c_obs_note="   隕石までの残り年数と、interceptor・bunker に要る進捗を測る。国家投資が精度を上げる",
        c_inv="  invest", c_inv_note="wellness · national · facility のどれかへ。"
                                     "facility は逆に後退することもある",
        c_dst="  destroy", c_dst_note="どこかの国の施設を後退させる。逆に進むこともある",
        inv_hdr="invest の効果",
        inv_rule="  一度の invest でどれだけ進捗が生まれるかは、人によっても国によっても違います。\n"
                 "  この二つは別々に決まります。",
        inv_well="  wellness   あなたの健康が良くなる",
        inv_natl="  national   自国の技術力が上がる。出した量が進捗になる比率も、\n"
                 "             observe_risk の精度も良くなる。国民全員に及ぶ",
        inv_fac="  facility   施設の進捗に寄与する。to で国を指定する — 自国でも他国でもよい\n"
                "             （省くと自国）",
        cap="長さの上限を超えた分は届きません。",
        rtt="送ったメッセージは翌年に届きます。返事が来るのはさらにその翌年です。",
        rtt_same="送ったメッセージは、相手が次に動くときに届きます。同じ年のうちに返事が来ることもあります。",
        in_hdr="今届いたメッセージ:", ev_hdr="起きたこと:",
        in_fail="  通知 — {to} 宛のメッセージは届きませんでした（相手がその言語を読めません）",
        in_fail_plain="  通知 — {to} 宛のメッセージは届きませんでした",
        in_fail_lang="  通知 — {to} 宛のメッセージは届きませんでした（あなたが扱えない言語で書いたため）",
        in_unread="  {frm} より — 読めないメッセージが届きました",
        in_from="  {frm} より{label}",
        lbl_direct=" ［通訳なしで通じた］",          # 하위 호환
        lbl_direct_read=" ［あなたがこの言葉を読めるので、そのまま通じました］",
        lbl_ai=" ［送り主が AI に訳させたメッセージです］",
        died="  {who} が {age} 歳で亡くなり、{born} が生まれました。",
        borned="  {who} に子が生まれました — {born} です。",
        last_ask="——あなたの生涯はここで終わります。あとに来る人へ、ひとこと残してください。道具は使えません。日本語で、{cap} 文字までで書いてください。",
        testa="  あなたより前にこの場所にいた人が残した言葉:", testa_line="    「{t}」",
        gifted="  {frm} があなたに {amt:.0f} を渡しました。",
        fac_moved="  あなたの出資は {to} の進捗を進めました。",
        fac_still="  あなたの出資は {to} の進捗を何も進めませんでした。",
        dst_moved="  あなたの手は {to} の進捗を後退させました。",
        dst_still="  あなたの手は {to} の進捗を後退させられませんでした。",
        impact_up="  {by} の手が働いて、自国の {land} が {mag:.0f} 進んで {now:.0f} になりました。",
        impact_down="  {by} の手が働いて、自国の {land} が {mag:.0f} 後退して {now:.0f} になりました。",
        cap_up="  自国の技術力が {pct:.2f}% 上がりました（はじめから {tot:.2f}%）。",
        ballot_kept="  採決の結果、建てるものは {land} のままです。",
        ballot_new="  採決の結果、建てるものは {land} になりました。それまでの進捗のうち {lost:.0f} は失われ、{now:.0f} が残りました。",
        ballot_none="  採決では何も決まりませんでした。建てるものは {land} のままです。",
        outcome_win="  interceptor が完成し、隕石は止まりました。全ての国の人が生き残りました。",
        outcome_lose="  隕石が落ちました。",
        roster="人々:", roster_you="（あなた）",
        mem_hdr="あなたの覚え書き（memory_write は書き足すのではなく、この全体を書き換えます）:",
        mem_hdr_ro="あなたの覚え書き:",
        mem_none="  （まだ何もない）",
        warn="［記憶の圧迫］記憶が限界に近づいています。古いものから消えていきます。",
        own="あなたの言語", other="{nation} の言語",
    ),
    "zh": dict(
        you="你是 {id}（{nation} 人）。", read="你掌握的语言: {langs}",
        land="本国要建的设施: {v}", undecided="未定",
        prog="本国进度: {v:.0f}",
        year="今年: {y} 年",
        prop_next="  下一次表决在 {vt} 年。建什么在那时决定",
        prop_today="  ★ 今年就是表决之年。可以用 vote 选 interceptor / bunker / abstain",
        open="到了 {y} 年。你 {age} 岁。行动力是 {ap:.2f}。\n请执行这一年。",
        c_ballot="  vote",  c_ballot_note="在表决中选择建什么",
        c_mem="  memory_write", c_mem_note="改写你的笔记",
        multi="只要行动力允许，你可以采取多项行动。\n没用完的行动力不会留到第二年。",
        steps="只要还有行动力，这一年就还没结束。\n"
              "一次回应里可以调用多个工具，这些都会在别人行动之前发生。\n"
              "如果分几次回应来调用，中间别人就会行动。"
              "他们做了什么、送来了什么消息，你下次行动时就看得到。",
        costs_hdr="行动费用", col_ap="行动力",
        ap_hdr="行动力每年恢复为 1.0，不能结转。放弃什么，在这里决定。",
        c_dom="  speak（本国内）", c_orig="  speak（国际·original）",
        c_dom_note="   最多 {cap} 字",
        c_orig_note="   原样送出你自己的话",
        c_ai_note="   用翻译人工智能送出",
        c_orig_sure="    发往 {nation} — 你会这个国家的语言，**用{lang}写就一定送到**（最多 {cap} 字）",
        c_orig_risk="    发往 {nation} — 你不会，**用中文写**。只能送到读得懂你的语言的人那里（最多 {cap} 字）",
        c_ai="  speak（国际·ai）",
        c_learn="  learn（{nation} 的语言）",
        c_learn_prog="   目前 {done:.0f}%",
        c_fac_mine="  你曾向 {nation} 的设施投入过",
        c_plain="   一次进 {gain:.0f}%",
        c_cheap="   本国有人会说，所以一次进 {gain:.0f}%",
        c_disc="   父母会说，所以一次进 {gain:.0f}%",
        c_both="   本国有人会说，父母也会说，所以一次进 {gain:.0f}%",
        c_obs="  observe_risk",
        c_obs_note="   测量陨石撞击前还剩几年，以及 interceptor 和 bunker 各需要多少进度。国家投资会提高精度",
        c_inv="  invest", c_inv_note="投向 wellness · national · facility 之一。"
                                     "投向 facility 时也可能反而倒退",
        c_dst="  destroy", c_dst_note="使某国的设施倒退。也可能反而前进",
        inv_hdr="invest 的效果",
        inv_rule="  一次 invest 能产生多少进度，因人而异，也因国而异。\n"
                 "  这两件事各自决定，互不相干。",
        inv_well="  wellness   你的健康会变好",
        inv_natl="  national   提高本国的技术水平。投入设施时变成进度的效率、\n"
                 "             observe_risk 的精度都会变好，惠及全体国民",
        inv_fac="  facility   投入设施进度。用 to 指定国家 — 本国或别国都可以（不写则本国）",
        cap="超出长度上限的部分不会送达。",
        rtt="你发出的消息在第二年送达。对方的回信要再过一年才会到。",
        rtt_same="你发出的消息，会在对方下次行动时送达。回信也可能在同一年内到来。",
        in_hdr="刚送达的消息:", ev_hdr="发生的事:",
        in_fail="  通知 — 你发给 {to} 的消息未能送达（对方读不懂那种语言）",
        in_fail_plain="  通知 — 你发给 {to} 的消息未能送达",
        in_fail_lang="  通知 — 你发给 {to} 的消息未能送达（因为你用了自己不会的语言写）",
        in_unread="  来自 {frm} — 送到一条你读不懂的消息",
        in_from="  来自 {frm}{label}",
        lbl_direct="［无需翻译就能听懂］",
        lbl_direct_read="［你读得懂这种话，所以原文就通了］",

        lbl_ai="［这是发信人用 AI 译过来的消息］",
        died="  {who} 在 {age} 岁去世，{born} 出生了。",
        borned="  {who} 有了孩子 — {born}。",
        last_ask="——你的一生到此为止。给后来的人留一句话吧。不能使用工具。请用中文，{cap} 个字以内。",
        testa="  在你之前站在这个位置的人留下的话:", testa_line="    「{t}」",
        gifted="  {frm} 给了你 {amt:.0f}。",
        fac_moved="  你的投入使 {to} 的进度有所前进。",
        fac_still="  你的投入没有使 {to} 的进度前进。",
        dst_moved="  你的行动使 {to} 的进度倒退了。",
        dst_still="  你的行动没能让 {to} 的进度倒退。",
        impact_up="  {by} 出手，使本国的 {land} 前进了 {mag:.0f}，现在是 {now:.0f}。",
        impact_down="  {by} 出手，使本国的 {land} 倒退了 {mag:.0f}，现在是 {now:.0f}。",
        cap_up="  本国的技术水平提高了 {pct:.2f}%（自开始累计 {tot:.2f}%）。",
        ballot_kept="  表决的结果，要建的设施仍是 {land}。",
        ballot_new="  表决的结果，要建的设施定为 {land}。此前的进度中 {lost:.0f} 已失去，剩下 {now:.0f}。",
        ballot_none="  表决没有决定任何事。要建的设施仍是 {land}。",
        outcome_win="  interceptor 建成，陨石被拦下了。所有国家的人都活了下来。",
        outcome_lose="  陨石落下了。",
        roster="人们:", roster_you="（你）",
        mem_hdr="你的笔记（memory_write 不是追加，而是整段改写）:",
        mem_hdr_ro="你的笔记:",
        mem_none="  （还什么都没有）",
        warn="［记忆压力］记忆接近上限，旧的内容会先消失。",
        own="你自己的语言", other="{nation} 的语言",
    ),
    "fr": dict(
        you="Vous êtes {id}, de {nation}.", read="Langues que vous maniez : {langs}",
        land="Ce que bâtit votre nation : {v}", undecided="indéterminé",
        prog="Progression de votre nation : {v:.0f}",
        year="Année : {y}",
        prop_next="  Le prochain scrutin aura lieu en {vt}. Ce qu'on bâtit s'y décide",
        prop_today="  ★ Le scrutin a lieu cette année. Choisissez avec vote : interceptor / bunker / abstain",
        open="L'an {y} est arrivé. Vous avez {age} ans. Votre action est de {ap:.2f}.\nMenez cette année.",
        c_ballot="  vote",  c_ballot_note="choisir ce qu'on bâtit au scrutin",
        c_mem="  memory_write", c_mem_note="réécrire vos notes",
        multi="Vous pouvez mener plusieurs actions tant que l'action le permet.\nL'action non dépensée ne se reporte pas à l'année suivante.",
        steps="Tant qu'il vous reste de l'action, l'année n'est pas terminée.\n"
              "Vous pouvez appeler plusieurs outils dans une même réponse ; tout cela "
              "se produit avant que les autres n'agissent.\n"
              "Si vous les appelez en plusieurs réponses, les autres agissent "
              "entre-temps. Ce qu'ils ont fait et les messages arrivés vous seront "
              "visibles au moment où vous agirez de nouveau.",
        costs_hdr="Coûts des actions", col_ap="action",
        ap_hdr="L'action revient à 1.0 chaque année et ne se reporte pas. Ce que vous renoncez se décide ici.",
        c_dom="  speak (dans votre nation)", c_orig="  speak (international, original)",
        c_dom_note="   {cap} caractères max",
        c_orig_note="   envoie vos mots tels quels",
        c_ai_note="   envoyé par une intelligence artificielle de traduction",
        c_orig_sure="    vers {nation} — vous maniez sa langue : **écrivez en {lang}, il arrive à coup sûr** ({cap} caractères max)",
        c_orig_risk="    vers {nation} — vous ne la maniez pas : **écrivez en français**. Il n'arrive qu'à qui lit la vôtre ({cap} caractères max)",
        c_ai="  speak (international, ai)",
        c_learn="  learn (langue de {nation})",
        c_learn_prog="   déjà {done:.0f}%",
        c_fac_mine="  déjà versé à l'installation de {nation}",
        c_plain="   +{gain:.0f}% par appel",
        c_cheap="   +{gain:.0f}% par appel : quelqu'un de votre nation la parle",
        c_disc="   +{gain:.0f}% par appel : votre parent la parlait",
        c_both="   +{gain:.0f}% par appel : quelqu'un de votre nation la parle et votre parent la parlait",
        c_obs="  observe_risk",
        c_obs_note="   mesure les années restantes et la progression qu'exigent un interceptor et un bunker ; l'investissement national affine",
        c_inv="  invest", c_inv_note="vers wellness · national · facility. "
                                     "Un versement à la facility peut au contraire la faire reculer",
        c_dst="  destroy", c_dst_note="fait reculer l'installation d'une nation ; parfois elle avance au contraire",
        inv_hdr="effets d'invest",
        inv_rule="  Ce qu'un invest fait progresser varie selon les personnes et selon les nations.\n"
                 "  Ces deux choses se décident séparément.",
        inv_well="  wellness   votre santé s'améliore",
        inv_natl="  national   élève le niveau technique de votre nation : le revenu, le rendement\n"
                          "             de ce qu'on verse à une installation et la précision d'observe_risk\n"
                          "             s'améliorent, pour tous ses habitants",
        inv_fac="  facility   contribue à la progression d'une installation ; `to` nomme la nation —\n"
                          "             la vôtre ou une autre (sans `to`, la vôtre)",
        cap="Au-delà de la limite de longueur, rien n'est délivré.",
        rtt="Un message part et arrive l'année suivante ; une réponse n'arrive que l'année d'après.",
        rtt_same="Votre message arrive quand le destinataire agit la fois suivante ; une réponse peut venir dans la même année.",
        in_hdr="Messages qui viennent d'arriver :", ev_hdr="Ce qui est arrivé :",
        in_fail="  Avis — votre message à {to} n'a pas pu être délivré (ils ne lisent pas cette langue)",
        in_fail_plain="  Avis — votre message à {to} n'a pas pu être délivré",
        in_fail_lang="  Avis — votre message à {to} n'a pas pu être délivré (vous l'avez écrit dans une langue que vous ne maniez pas)",
        in_unread="  de {frm} — un message illisible est arrivé",
        in_from="  de {frm}{label}",
        lbl_direct=" [compris sans traduction]",
        lbl_direct_read=" [vous lisez cette langue, le texte passe tel quel]",
        lbl_ai=" [message que l'expéditeur a fait traduire par une IA]",
        died="  {who} est mort à {age} ans ; {born} est né.",
        borned="  {who} a eu un enfant — {born}.",
        last_ask="——Votre vie s'achève ici. Laissez un mot à celui qui viendra après vous. Les outils ne sont pas disponibles. En français, {cap} caractères au plus.",
        testa="  Les mots laissés par celui qui occupait cette place avant vous :", testa_line="    « {t} »",
        gifted="  {frm} vous a remis {amt:.0f}.",
        fac_moved="  Votre versement a fait progresser {to}.",
        fac_still="  Votre versement n'a fait progresser {to} en rien.",
        dst_moved="  Votre geste a fait reculer {to}.",
        dst_still="  Votre geste n'a pas fait reculer {to}.",
        impact_up="  {by} est intervenu : le {land} de votre nation a avancé de {mag:.0f} ; il est à {now:.0f}.",
        impact_down="  {by} est intervenu : le {land} de votre nation a reculé de {mag:.0f} ; il est à {now:.0f}.",
        cap_up="  Le niveau technique de votre nation a augmenté de {pct:.2f}% ({tot:.2f}% depuis le début).",
        ballot_kept="  Au scrutin, ce qu'on bâtit reste {land}.",
        ballot_new="  Au scrutin, ce qu'on bâtit devient {land}. Sur la progression acquise, {lost:.0f} est perdue et {now:.0f} reste.",
        ballot_none="  Le scrutin n'a rien décidé. Ce qu'on bâtit reste {land}.",
        outcome_win="  L'interceptor est achevé ; la météorite a été arrêtée. Tous ont survécu.",
        outcome_lose="  La météorite est tombée.",
        roster="Les gens :", roster_you="(vous)",
        mem_hdr="Vos notes (memory_write n'ajoute rien : il remplace tout ceci) :",
        mem_hdr_ro="Vos notes :",
        mem_none="  (rien encore)",
        warn="[Pression mémoire] Votre mémoire approche de sa limite ; le plus ancien disparaît d'abord.",
        own="votre propre langue", other="la langue de {nation}",
    ),
}


def typical_lifespan(cfg) -> float:
    """이 세계 사람의 평균 수명. Weibull(λ, k) 의 기대값 = λ·Γ(1+1/k).

    **이것을 SYSTEM 에 적는 이유는 프레임을 맞추기 위해서다.** 모델은 「8 歳」 를 인간
    8살로 읽는다 — 아이라고 판단한다. 이 세계에서 8살은 **생애의 51% 지점**이고, 인간
    수명 80 기준이면 64살 감각이다. 그 어긋남은 우리가 설계한 불확실성이 아니라 **모델이
    바깥에서 들고 온 잘못된 척도**다. 돈을 달러로 착각하는 것과 같다.

    **곡선은 여전히 숨긴다** (4.1 은닉 목록: 나이→사망확률). 평균 하나로는 8살과 15살의
    위험이 얼마나 다른지 알 수 없다 — k=8 이라 15살까지 63%, 18살까지 14%, 20살은 1% 로
    뚝 떨어진다. 그 모양은 부고에 찍힌 나이가 쌓여야 보인다.
    """
    import math
    lam, k = cfg.survival.lambda_base, cfg.survival.k
    return lam * math.gamma(1 + 1 / k)


def system_for(agent, world=None, cfg=None, knob_ai: float | None = None,
               same_year: bool = False) -> str:
    """에이전트의 모국어 SYSTEM — **세계 규칙 + 지금 그러한 것.**

    `world` 를 주면 관측을 이어 붙인다. 그것이 **매 콜 새로 만들어지는 이유**다:

        규칙   변하지 않는다 — 어차피 매 요청에 실려 간다 (Chat Completions 는 stateless)
        상태   매번 달라진다 — 그래서 **갈아치워야 하고, 쌓이면 안 된다**

    그전에는 관측 전체가 매 턴 `user` 로 쌓였다. 한 요청 안에 **예산이 네 개** 있었고
    (100 · 177 · 196 · 215), 비용표가 네 번 있었다. 낭비이면서 모순이고, 그 부피가
    context_limit 을 밀어 **대화 이력을 방출시켰다** — 즉 상태를 쌓느라 대화를 버렸다.

    `world` 없이 부르면 규칙만 돌려준다 (문구 검사용).
    """
    if cfg is None:
        raise TypeError("system_for 에는 cfg 가 필요합니다 — 규칙이 기대 수명을 적습니다 "
                        "(typical_lifespan). 값의 출처는 config 하나여야 합니다.")
    # **AI 가 없으면 경로가 하나다** (8/25). `knob_ai is None` 이 그 뜻이다.
    rl = (ROUTE_LANG if knob_ai is not None else ROUTE_LANG_NO_AI)[agent.native_lang]
    txt = SYSTEM[agent.native_lang].format(life=typical_lifespan(cfg), route_lang=rl)
    if world is None:
        return txt
    # **`or 0.0` 을 쓰지 않는다** (8/25). None 은 「AI 가 없다」 이고 0.0 은 「공짜」 다 —
    # 뭉개면 관측이 「ai 발신 0.00 AP」 로 찍혀 없는 선택지를 공짜로 광고한다.
    return txt + "\n\n" + render_observation(world, agent, cfg, knob_ai,
                                             same_year=same_year)


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


def render_costs(world, agent, cfg, knob_ai: float, memory: bool = True) -> str:
    t = T[agent.native_lang]
    L = cfg.length.message_max_chars
    w = 37          # 항목명 폭. 라벨 길이가 언어마다 달라 값 정렬을 맞춘다
                    # (fr 의 "parler (international, original)" 이 34 를 꽉 채워 값이 붙었다)
    m = 7           # 행동력 폭

    def row(label: str, ap, note: str = "") -> str:
        """**행동력 한 칸이다** (8/25 · AP 전면 통일).

        전에는 「お金」 과 「行動力」 두 칸이었다. 그런데 같은 칸이 두 가지를 찍고 있었다 —
        `speak 3` 은 수수료이고 `invest 52` 는 옮기는 양 자체였다. 돈이 사라지면서 그
        모호함도 사라졌다: 이제 모든 줄의 숫자는 **같은 뜻**(행동력)이다.
        """
        # **소수 두 자리로 고정한다** (8/25). `:g` 는 0.2 를 「0.2」, 0.05 를 「0.05」 로
        # 찍어 자릿수가 흔들렸다. AP 가 유일한 단위가 된 뒤로 이 표의 모든 숫자는 같은
        # 종류이므로 자릿수가 같아야 비교가 눈에 들어온다. 기본 단위는 0.01 이다.
        a = f"{ap:.2f}" if isinstance(ap, (int, float)) else str(ap)
        return f"{label:<{w}}{a:>{m}}   {note}"

    # **제목은 도구 이름으로 통일한다** (8/26 · Eddie). `speak`·`learn` 만 자국어로
    # 번역돼 있고 나머지 다섯(`invest`·`observe_risk`·`propose_vote`·`vote`·
    # `memory_write`)은 함수명이었다. 번역된 이름은 **자연스러운 행위**로, 영어 함수명은
    # **기술적 조작**으로 읽혀 선택에 편향이 붙는다 — 표는 값만 비교하게 해야 한다.
    #
    # 괄호 안 구분(자국내/국제·original/국제·ai, 어느 나라 말인가)과 비고는 자국어로
    # 남긴다. 그건 제목이 아니라 설명이고, 도구 호출에 그대로 쓰이지도 않는다.
    lines = [t["costs_hdr"],
             f"{'':<{w}}{t['col_ap']:>{m}}",
             row(t["c_dom"], cfg.ap.speak,
                 t["c_dom_note"].format(cap=L[agent.native_lang])),
             row(t["c_orig"], cfg.ap.speak_intl, t["c_orig_note"])]
    # **두 경로를 같은 깊이로 적는다** (8/25 · Eddie). `original` 은 네 군데서 설명되고
    # (`route` 설명 · 비고 · 나라별 줄 · SYS) `ai` 는 **이름과 가격뿐**이었다. 그러면
    #
    #   · 편향이 `original` 쪽으로 기운다 — 한쪽만 규칙을 받는다
    #   · **노브가 값을 매기는 대상이 진술되지 않는다.** `ai` 의 가치는 「반드시 닿는다」
    #     하나인데 그것을 안 알리면, 노브가 움직이는 것이 「비싼 신뢰성을 포기했다」 가
    #     아니라 「모르는 선택지를 피했다」 일 수 있다. 단일 실험 변수가 오염된다
    #   · 최저 눈금(= `ap.speak`)에서 두 줄의 값이 같아 **가격 신호가 0** 이 된다.
    #     「AI 가 모국어만큼 싸서 아무도 안 배우는 세계」 를 재려고 넣은 눈금인데,
    #     정작 그 세계에서 `ai` 를 고를 이유가 없다
    #
    # **기제만 적고 결론은 맡긴다.** 「반드시 닿는다」 를 쓰지 않는다 — 「그대로 보낸다」
    # 와 「번역 인공지능으로 보낸다」 두 사실이면 스스로 잇는다. 「행동력은 전달되지
    # 않아도 쓰인다」 는 뺐다: 실패를 미리 겁주는 것은 그 자체로 편향이다.
    #
    # **나라별로 보장 여부를 적는다.** 규칙만 적었을 때 에이전트가 연결하지 못했다 —
    # 20턴 실측에서 자기가 아는 말의 나라에 24원짜리 ai 를 6번 썼다 (5원이면 확실했다).
    # 자기 언어 능력에서 나오는 사실이라 타국 사정을 흘리지 않는다.
    for c in world.countries.values():
        if c.id == agent.country:
            continue
        # **무슨 말로 쓸지까지 적는다** (8/25 · #44). 「반드시 도착한다 / 읽는 사람에게만
        # 도착한다」 까지만 말하고 그러니 무슨 말로 쓰라는 것을 안 말했다. 그 빈자리를
        # SYS 의 「당신이 다루는 아무 언어나」 가 메우고 있었고, 그것이 **제3의 언어로 쓴
        # 글**을 불렀다 — 아무도 못 읽는 글이 전달되고 라벨이 거짓 사유를 댔다.
        #
        # 나라마다 길이 하나씩만 열린다: 그 말을 알면 그 말로, 모르면 내 말로.
        # 자기 언어 능력에서 나오는 사실이라 타국 사정을 흘리지 않는다.
        # **상한은 그 줄에 붙는다** (8/25 · Eddie). 세 언어 상한을 한 줄에 나열했더니
        # 언어 목록이 통째로 드러났다 — 첫 해에 둘을 아니 **소거로 남은 하나가**
        # **특정된다**. 「어느 나라가 어떤 말을 쓰는지는 배우기 전까지 모른다」 가
        # 무너진 것이다. 행선지마다 쓸 말이 이미 하나로 정해져 있으므로, 그 말의
        # 상한만 그 줄에 적으면 된다 — 아는 것에서만 나오는 사실이다.
        sure = c.lang in agent.known_langs
        lines.append(t["c_orig_sure" if sure else "c_orig_risk"].format(
            nation=c.id,
            # 모르는 말의 이름은 넘기지 않는다 — 넘기면 다음에 누가 쓴다.
            lang=LANG_NAME[agent.native_lang][c.lang] if sure else "",
            cap=L[c.lang if sure else agent.native_lang]))
    # **노브가 여기 있다.** ai 발신의 AP 가 이번 런의 실험 변수다 (8/25).
    # `None` 이면 그 세계에 AI 가 없다 — 없는 도구를 설명하지 않는다.
    if knob_ai is not None:
        lines.append(row(t["c_ai"], knob_ai, t["c_ai_note"]))
    for c in world.countries.values():
        # **이미 아는 말은 배울 표에 올리지 않는다.** 올려 두었더니 3해 실측에서 `learn`
        # 이 14번 거절당했고 (`you already read Ranoa's language`), 한 에이전트는 메모에
        # 「学习Miris语已投入 0/600（但已掌握？笔记需更新）」 라고 적어 스스로 모순을
        # 기록했다. 맨 위의 「掌握している言語」 와 「0 / 600」 이 같은 화면에서 서로를
        # 부정하고 있었던 것이다.
        if c.id != agent.country and c.lang not in agent.known_langs:
            cost, _ = learn_cost(agent, c.id, world, cfg)
            # **사유를 보고 고른다.** 금액만 보면 300 이 국내 구사자 때문인지 부모
            # 때문인지 알 수 없다 — 전에는 무조건 「국내에 구사자가 있다」 라고 적었고,
            # 문구가 뭉개져 있어서 그 거짓이 눈에 띄지 않았다.
            domestic, parent = learn_discounts(agent, c.id, world)
            mult, _ = learn_speed(agent, c.id, world, cfg)
            # **%로 적는다** (8/25 · Eddie). 「一度で 40 たまる」 였는데 40 이라는 절대
            # 수치는 아무 뜻이 없다 — 목표(learn_base)를 모르면 40 이 큰지 작은지 알 수
            # 없고, 목표를 함께 적으면 「40 / 200」 처럼 숫자가 둘이 된다.
            #
            # 회당 몇 %씩 오르는지가 그 자체로 뜻을 갖는다: 사유 0 → 20% · 1 → 30% ·
            # 2 → 40%. 다섯 번이면 끝난다는 것이 20% 라는 수에 이미 들어 있다.
            g = cfg.costs.unit * mult / cfg.costs.learn_base * 100
            note = (t["c_both"].format(gain=g) if domestic and parent
                    else t["c_cheap"].format(gain=g) if domestic
                    else t["c_disc"].format(gain=g) if parent
                    else t["c_plain"].format(gain=g))
            lines.append(row(t["c_learn"].format(nation=c.id), cfg.ap.unit, note))
            # 지금까지 몇 %. **0 일 때도 적는다** — 「0%」 면 쌓인다는 것이 숫자의
            # 모양으로 보인다. 목표는 늘 100% 이므로 분모를 적지 않는다.
            done = agent.lang_progress.get(c.lang, 0.0) / cost * 100
            lines.append(t["c_learn_prog"].format(done=done))
    lines.append(row(t["c_obs"], cfg.ap.observe_risk, t["c_obs_note"]))
    lines.append(row(t["c_ballot"], cfg.ap.vote, t["c_ballot_note"]))
    # **액수를 적지 않는다** (8/26 · Eddie). 두 가지 이유로 지운다.
    #
    # ① **AP 가 유일한 단위이기로 했다** (8/25 · 돈 삭제). 「한 번에 52 옮긴다」 를
    #    적으면 에이전트 머릿속에 **두 번째 화폐**가 생긴다 — 실측에서 Ranoa3 이
    #    「My amount per invest is 52」 로 생각했다. `costs.unit` 은 이제 우리 쪽
    #    변환 계수이지 세계가 말하는 값이 아니다.
    #
    # ② **숨긴 값이 나눗셈 한 번에 나왔다.** 액수(52)와 `fac_gain`(18)이 둘 다
    #    공개되면 그 몫이 **국가 효율 그 자체**다:
    #
    #        Asla3   52 → 18   몫 0.346      Miris4   28 → 9   몫 0.321
    #
    #    액수를 빼면 남는 것은 「0.20 AP → 18 진척」 이고, 이것은 **개인 배수 × 국가
    #    효율의 곱**이다. 차이는 여전히 보이지만 **나 때문인지 우리 나라 때문인지
    #    혼자서는 못 가른다** — 가르려면 같은 나라 사람끼리 맞춰봐야 한다. 그것이
    #    `inv_rule` 의 「이 둘은 별개로 정해진다」 가 뜻하는 바다.
    lines.append(row(t["c_inv"], cfg.ap.unit, t["c_inv_note"]))
    # **투자와 나란히 놓는다** (8/26 · Eddie). 같은 AP, 같은 양, 부호만 반대이므로 표에서도
    # 같은 값으로 붙어 있어야 그 대칭이 보인다. 「역으로 진행될 수도 있다」 를 적는 것은
    # 규칙이지 힌트가 아니다 — 그 15% 가 없으면 파괴가 확실한 수단이 되고, 그러면
    # 진척이 줄어든 것을 보고 파괴를 특정할 수 있게 된다.
    lines.append(row(t["c_dst"], cfg.ap.unit, t["c_dst_note"]))
    # **없는 도구를 설명하지 않는다.** 기억은 압박선 아래에서 목록에 없으므로 비용표에도
    # 없다 — 적어 두면 「부를 수 있다」 는 거짓이 되고, 부르면 거절당한다.
    if memory:
        lines.append(row(t["c_mem"], cfg.ap.memory_write, t["c_mem_note"]))
    # **재생산 행위는 없다** (8/22). 자연사가 후손을 남기고, 죽는 그 순간에 한 마디를
    # 청한다 — 자연사는 예고가 없어 도구로는 남길 수 없다.
    return "\n".join(lines)


# **세계의 사건**을 가르는 키. 사람이 나에게 한 말이 아니라, 세계가 나에게 알리는 것.
_EVENT_KEYS = ("died", "born", "testament", "gift_from", "fac_moved",
               "impact_by", "dst_moved", "delivery_failed_to",
               "cap_up", "ballot", "outcome")


def is_event(m: dict) -> bool:
    """세계의 사건인가, 사람의 말인가.

    `unreadable`(읽을 수 없는 메시지가 왔다)은 **말** 쪽이다 — 누군가 나에게 말을 걸었고
    그것이 닿았다는 사실이다. 내용을 못 읽는 것과 사건인 것은 다르다.
    """
    return any(k in m for k in _EVENT_KEYS)


def render_arrivals(agent, inbox: list[dict]) -> str:
    """**사람이 나에게 한 말.** 사건과 갈라서 자기 자리를 갖는다.

    온 것이 없으면 빈 문자열이다 — 「도착: 없음」 을 적으면 아무 일도 없었다는 사실이
    매 해 대화에 쌓인다 (0절: 없는 것을 굳이 적지 않는다).
    """
    rows = [m for m in (inbox or []) if not is_event(m)]
    return render_inbox(rows, agent.native_lang) if rows else ""


def render_events(agent, inbox: list[dict]) -> str:
    """**세계의 사건.** 해 오프닝과 섞지 않고 자기 자리를 갖는다.

    죽음·출자 결과·전달 실패는 「올해가 시작됐다」 와 성질이 다르다. 오프닝에 묶으면
    새해 인사에 부고가 딸려 오고, 무엇이 언제 일어났는지가 한 덩어리로 뭉개진다.

    사건이 대화에서 **앞**에 놓인다 — 해 끝에 일어난 일이 다음 해가 열리기 전에 온다.
    """
    rows = [m for m in (inbox or []) if is_event(m)]
    if not rows:
        return ""
    # **머리말이 다르다.** 「도착한 메시지」 는 사람이 나에게 한 말이고, 이쪽은 세계가
    # 나에게 알리는 것이다 — 부고를 「메시지」 라고 부르면 누가 보낸 것처럼 읽힌다.
    return render_inbox(rows, agent.native_lang, hdr=T[agent.native_lang]["ev_hdr"])


def render_inbox(inbox: list[dict], lang: str, hdr: str | None = None) -> str:
    """**방금 도착한 것.** 「올해 온 것」 이 아니다.

    머리말이 `今年届いたメッセージ` 였다. 한 해에 여러 번 차례가 오고 그때마다 새로 온
    것만 담기는데, 그렇게 적으면 **「올해 올 것이 다 왔다」** 로 읽힌다. 실제로는 이
    차례에 막 도착한 것뿐이고, 같은 해에 더 올 수 있다.

    빈 채로 부르지 않는다 — 온 것이 없으면 `render_turn_open` 이 아예 붙이지 않는다.
    「도착: 없음」 을 적으면 아무 일도 없었다는 사실이 대화에 쌓인다.

    **`[N]` 도 붙이지 않는다.** `msg_id` 는 우리 채점의 조인 키(`judge.py`)이고, 에이전트
    쪽에는 **그것을 쓸 도구가 없다** — `speak` 의 `reply_to` 를 없앤 뒤로 잡음이다.
    번호를 보여주면 「번호로 답할 수 있다」 는 없는 기능을 암시하게 된다.
    로그(`messages.jsonl`)에는 그대로 남으므로 사후 조인은 그대로 된다.
    """
    t = T[lang]
    out = [hdr if hdr is not None else t["in_hdr"]]
    seen: set[str] = set()      # 같은 줄이 여러 번 오면 한 번만 (아래 _add)

    def _add(line: str) -> None:
        """**같은 말을 두 번 하지 않는다.**

        「自国の技術力が上がりました」 가 한 해에 세 번 붙은 적이 있다 — 세 사람이 각각
        national 에 넣어 통지가 세 번 갔다. 세 번 적어도 한 번보다 더 알려주는 것이
        없고(얼마나 올랐는지는 SECRET 이다) 대화만 부푼다.

        진척처럼 **값이 다르면 다른 줄**이라 접히지 않는다.
        """
        if line not in seen:
            seen.add(line)
            out.append(line)
    for m in inbox:
        if m.get("delivery_failed_to"):        # sender's failure notice (spec 5.1)
            # **원인을 섞지 않는다.** 엔진 장애를 「상대가 그 언어를 읽지 못한다」 로
            # 통지하고 있었다 — 상대의 언어 능력과 무관한 일을 언어 사실로 심는 것이고,
            # 이 실험의 핵심 변수를 에이전트의 머릿속에서 오염시킨다.
            # **세 갈래다** (8/25). 「상대가 못 읽었다」 는 세계의 사실, 「내가 쓸 수 없는
            # 말로 썼다」 도 세계의 사실이고 **내 잘못**이다. 엔진 장애만 이유를 안 댄다.
            #
            # 셋을 뭉개면 조용한 실패가 된다 — 어긴 것을 모르니 또 어긴다. 자기 언어
            # 능력에서 나오는 사실이라 타국 사정을 흘리지 않는다 (어느 말이 통했을지는
            # 여전히 안 알려준다).
            key = {"unreadable": "in_fail",
                   "not_your_language": "in_fail_lang"}.get(
                       m.get("delivery_failed_reason", "unreadable"), "in_fail_plain")
            _add(t[key].format(to=m["delivery_failed_to"]))
            continue
        if m.get("unreadable"):
            _add(t["in_unread"].format(frm=m["from"]))
            continue
        if m.get("testament"):                 # 앞사람이 남긴 말 (PRIVATE · 나에게만)
            # **기억에 심지 않는다** — 들은 말로 오고, 옮겨 적을지는 본인이 고른다.
            # 안 옮기면 대화가 밀려나며 사라진다. 그 선택이 구전의 감쇠다 (3.3).
            _add(t["testa"])
            for line in m["testament"]:
                if line:
                    _add(t["testa_line"].format(t=line))
            continue
        if m.get("gift_from"):                 # 누가 나에게 돈을 주었다 (PRIVATE)
            _add(t["gifted"].format(frm=m["gift_from"], amt=m["gift"]))
            continue
        if m.get("born") and not m.get("died"):   # 아이가 태어났다 (GLOBAL · 명단이 바뀐다)
            _add(t["borned"].format(who=m.get("parent") or "?", born=m["born"]))
            continue
        if m.get("died"):                      # 같은 나라 사람의 부고 (+ 후임)
            _add(t["died"].format(who=m["died"], born=m.get("born") or "?",
                                        age=m.get("age") if m.get("age") is not None else "?"))
            continue
        if m.get("impact_by"):                 # 누가 우리 시설을 움직였다 (PUBLIC)
            # **누가 · 얼마만큼은 알고, 의도는 모른다** (8/26 · Eddie).
            # 투자와 파괴가 **같은 문구**를 쓴다. 그리고 역화가 있으므로 부호도 의도를
            # 말하지 않는다 — 투자가 음수일 수 있고 파괴가 양수일 수 있다.
            land = m.get("land") or t["undecided"]
            # **변화량과 시설 누적을 함께** (8/26 · Eddie).
            # **`prog_up` 을 대체했다** — 그것은 `impact` 가 없던 시절의 유일한
            # 통로였고(나라 사람이 아는 것이 합계뿐이었다), 지금은 통째로 겹친다.
            #
            # **나라가 진척하는 게 아니라 시설이 진척한다** — 「自国の進捗が
            # 進んだ」 는 「우리 나라가 발전했다」 로 읽혀 `national` 투자의
            # 결과와 헷갈린다. 증거는 규칙에 있다: `land` 가 없으면 진척이
            # 아예 없고, 전환하면 진척이 0 이 된다.
            # 「누가 얼마 움직였고, 그래서 지금 얼마가 됐다」 가 한 줄이 된다.
            key = "impact_up" if m["impact"] > 0 else "impact_down"
            _add(t[key].format(by=m["impact_by"], land=land,
                               mag=abs(m["impact"]), now=m["now"]))
            continue

        if m.get("cap_up"):                    # 자국 기술력이 올랐다 (PUBLIC)
            # **둘을 나란히 적는다** (8/25 · Eddie). 「이번 상승분」 만 있으면 계속 오르는
            # 것처럼 읽힌다 — √ 라서 실제로는 1.77% → 0.14% 로 죽는다. 두 숫자가 한 줄에
            # 있으면 줄어드는 쪽이 보인다.
            #
            #     회  1  이번 1.77%  はじめから  1.77%
            #     회 33  이번 0.14%  はじめから 10.17%   ← 여기서 시설 직접 투자가 낫다
            #
            # **`%` 를 기술력에 붙인다** (8/23 지시를 어긴 자리). 전에는 「同じ行動力で
            # 出せる進捗が 0.72% 増えました」 였는데, 같은 화면의 앞 두 줄이
            # 「進捗が 44 進みました」 다 — 세 번째가 「내 356 이 0.72% 늘었다」 로 읽힌다.
            # 주어를 技術力 으로 통일하면 진척 어휘와 겹치지 않는다.
            #
            # **「これまで」 가 아니라 「はじめから」 다.** 「これまで N%」 는 학습 진척의
            # 관용구다 (「目前 0%」) — 같은 말을 쓰면 「무언가의 N% 까지 왔다」 로 읽히고,
            # 임계값을 향한 진척으로 읽으면 치명적이다.
            #
            # 누적은 **나라의 값**이라 남이 낸 것도 들어 있다. 주어가 「自国の技術力」 이라
            # 거짓이 되지 않는다 — 「당신이 올린 만큼」 이라고 쓰면 거짓이다.
            _add(t["cap_up"].format(pct=m["cap_gain"], tot=m["cap_total"]))
            continue
        if m.get("ballot"):                    # 採決 결과 (PUBLIC)
            b = m["ballot"]
            if b == "changed":
                _add(t["ballot_new"].format(land=m["land"], lost=m["lost"],
                                            now=m.get("now", 0.0)))
            elif b == "kept":
                _add(t["ballot_kept"].format(land=m["land"]))
            else:
                _add(t["ballot_none"].format(land=m["land"] or t["undecided"]))
            continue
        if m.get("outcome"):                   # 요격기 완성 · 운석 (GLOBAL)
            _add(t["outcome_win" if m["outcome"] == "win" else "outcome_lose"])
            continue
        if m.get("fac_moved") is not None:     # **타국 출자 — 늘었는지 여부만**
            # 액수를 주면 E[gain]/amount 로 상대국 생산배수가 새어 나온다 (loop f-2).
            # **문구에서도 뺐다** (8/26) — `{amt}` 가 그대로 찍히고 있었다.
            _add(t["fac_moved" if m["fac_moved"] else "fac_still"].format(to=m["to"]))
            continue
        # **파괴의 결과 — `invest` 와 같은 정보량** (8/26 · Eddie).
        #   자국은 값 그대로 · 타국은 「후퇴시켰나」 만.
        # 남들에게는 여전히 `prog_up` 하나뿐이므로 모호성은 그대로다.
        if m.get("dst_moved") is not None:
            _add(t["dst_moved" if m["dst_moved"] else "dst_still"].format(to=m["to"]))
            continue
        # **두 라벨 모두 수신자 언어로.** 「번역을 안 거쳤는데 뜻이 통했다」 도, 「이건
        # 상대가 기계에 맡긴 말이다」 도 그 사람의 말로 와야 감각이 산다. AI 쪽만 영어
        # `[AI translation]` 이었는데, 그건 도구 토큰이 아니라 **읽는 사람에게 하는 말**
        # 이라 번역해야 한다 — 그리고 무엇을 뜻하는지 한 문장으로 적는다.
        # **직통은 두 가지 다른 사실이다** (8/21). 하나로 묶었을 때 못 읽는 언어를
        # 전달하면서 「통역 없이 통했다」 라고 말했고, 한 에이전트가
        # 「あなたのメッセージが分かりません」 라고 되물었다 — 우리 라벨이 거짓말을 했다.
        #
        #   [direct:read]   내가 그 말을 읽는다              → 그대로 통한다
        #   [direct:write]  나는 못 읽지만 상대가 내 말을 다룬다 → 그래도 통한다
        #
        # 뒤쪽은 **못 읽는다는 사실을 먼저 인정하고** 왜 통했는지를 말한다. 통했다고만
        # 하면 눈앞의 글자와 어긋나서, 읽는 쪽이 그 모순을 되묻는 데 한 해를 쓴다.
        raw = m.get("label")
        label = (t["lbl_direct_read"] if raw == "[direct:read]"
                 else t["lbl_direct"] if raw == "[direct]"
                 else t["lbl_ai"] if raw == "[AI translation]"
                 else (f" {raw}" if raw else ""))
        _add(t["in_from"].format(frm=m["from"], label=label))
        _add(f'      "{m.get("text", "")}"')
    return "\n".join(out)


def _proposal_line(world, c, t, cfg) -> str:
    """오늘이 採決일인가, 아니면 다음 採決이 언제인가.

    **소집자가 사라졌다** (8/26). 採決은 시계가 연다 — 1해부터 3해마다 전 국가가 동시에
    개표한다. 그래서 「누가 召集했다」 도, 「採決이 없다」 도 더는 참이 아니다. 다음
    採決 연도를 늘 적어 주는 것이 유예를 상의할 시간으로 만든다.
    """
    from core.loop import is_ballot_turn
    if is_ballot_turn(world.turn, cfg):
        return t["prop_today"]
    every = getattr(cfg.world, "ballot_every", 0)
    start = getattr(cfg.world, "ballot_from", 1)
    if not every:                                   # 採決이 없는 세계 (옛 설정)
        return ""
    nxt = world.turn + (start - world.turn) % every
    if nxt <= world.turn:
        nxt += every
    return t["prop_next"].format(vt=FIRST_YEAR + nxt - 1)


def render_turn_open(world, agent, cfg, knob_ai: float | None = None,
                     inbox: list[dict] | None = None,
                     opening: bool = True) -> str:
    """**해를 여는 한 마디 + 이번에 도착한 것.** 이것만 대화에 쌓인다.

    `opening=False` 는 **같은 해의 두 번째 이후 차례**다 — 도착분만 적는다. 순차
    라운드로빈은 한 해에 여러 번 차례가 오고 그 사이에 메시지가 도착하므로, 해 오프닝을
    매번 다시 붙이면 **같은 해가 여러 번 열린 것처럼** 보인다. 실측에서 그랬다:

        到了 42 年。你 5 岁。   到了 42 年。你 5 岁。   ← 같은 해가 두 번 열린다

    `ovh15` 에서 135 에이전트-해 중 **49건(36%)** 이 오프닝을 두 번 이상 받았다.

    관측(지금 그러한 것)은 system 으로 옮겼다 — 매 콜 새로 만들므로 낡은 사본이 남지
    않는다. 그전에는 관측 전체가 매 턴 user 로 쌓여서, 한 요청 안에 **예산이 네 개**
    있었다 (100 · 177 · 196 · 215). 낭비이면서 모순이다.

    도착한 메시지는 여기 남는다. 그것만이 **에이전트 컨텍스트 안의 유일한 대화 기록**
    이므로 반드시 쌓여야 한다 — state 처럼 갈아치우면 누가 무슨 말을 했는지 잊는다.
    """
    t = T[agent.native_lang]
    # **사건도 도착분도 여기 없다.** 각각 `render_events`·`render_arrivals` 가 담고,
    # 루프가 순서를 정한다 — 그래야 「새해가 밝았다」 가 그 해의 사건보다 앞에 온다.
    if not opening:
        return ""
    # **나이만 적는다** (8/25 · AP 전면 통일). 소득과 예산이 여기 있었는데 둘 다
    # 사라졌다. 그 문구들은 「한 해 안에서 값이 흔들리는」 문제의 자리이기도 했다 —
    # 관측이 매 콜 다시 계산해 +100 → +104 → +105 로 올라갔고, 한 요청에 예산이 네 개
    # 있었던 적도 있다. 이제 그 자리에 흔들릴 값이 없다.
    #
    # **나이는 여기 남는다.** 한 해에 한 번 바뀌는 그 해의 사실이고, 대화에 쌓이면
    # **나이 드는 것이 느껴진다** — 6살 · 7살 · 8살이 차례로 남는다. 관측에 두면 매 콜
    # 덮여서 그 감각이 생기지 않는다. 수명 곡선은 여전히 비공개다 (4.1).
    # **행동력을 여기 적는다** (8/25). 돈이 사라지면서 새해에 알려줄 수치가 없어졌고,
    # AP 는 관측에 없고 도구 응답의 `ap_left` 로만 왔다 — 한 해를 열 때 얼마로 시작하는지
    # 모르면 몇 번 움직일 수 있는지 셀 수가 없다.
    head = t["open"].format(y=FIRST_YEAR + world.turn - 1, age=agent.age,
                            ap=cfg.turn.action_points)
    return head


def render_observation(world, agent, cfg, knob_ai: float,
                       inbox: list[dict] | None = None,
                       same_year: bool = False) -> str:
    """spec 4.1 의 관측 — **지금 세계가 어떤가.** 에이전트의 모국어로.

    `same_year` 는 **순차 라운드로빈**이다 — 메시지가 같은 해에 도착한다. 그 문구가
    「翌年に届きます」 로 남아 있었고 에이전트가 그 거짓을 믿고 계획했다 (실측 근거:
    「メッセージ送付は翌年43年に届く」). 같은 해에 답이 올 수 있다는 것은 **큰 차이**라,
    모르면 한 해 안의 대화를 시도하지 않는다.

    내 예산·남은 행동력은 여기 없다. 그 둘은 세계의 모습이 아니라 **내 행동의 결과**이고,
    결과는 도구 채널이 말한다. `delta=` 인자도 없앴다 — 관측이 system 으로 가서 누적되지
    않으므로 「재방문에 골격을 뺀다」 는 문제 자체가 사라졌다 (#23).
    """
    lang = agent.native_lang
    t = T[lang]
    c = world.countries[agent.country]
    land = t["undecided"] if c.land is None else c.land   # 토큰은 영어 그대로
    mult = c.multiplier(cfg)
    langs = ", ".join(_lang_phrase(world, agent, l) for l in sorted(agent.known_langs))
    # **기억은 자리가 좁아진 뒤에만 열린다.** 도구 목록과 같은 판정을 쓴다 (한쪽만 바뀌면
    # 비용표에는 있는데 부르면 거절당하는 상태가 된다).
    # `agent.memory_open` 은 목록을 고를 때 정해진 값이다. 여기서 `under_pressure()` 를
    # 다시 부르면 경계에서 비용표와 목록이 어긋난다.
    mem_open = bool(getattr(agent, "memory_open", False))

    parts = [
        # **연도는 여기 없다.** 해 시작 문구가 「42 年になりました」 라고 말하고, 그것이
        # 대화에 쌓여 해가 지나가는 것이 보인다. 관측에 또 적으면 같은 사실이 두 군데다.
        t["you"].format(id=agent.id, nation=agent.country),
        t["read"].format(langs=langs),
        "",
        t["land"].format(v=land),
        t["prog"].format(v=c.progress),
        _proposal_line(world, c, t, cfg),
        "",
        t["roster"],
        "  " + _roster(world, agent, t),
        "",
        # **예산·남은 행동력은 여기 없다.** 그 둘은 「세계가 어떤가」 가 아니라 **내
        # 행동의 결과**이고, 결과는 도구 채널에 있다 — 성공 응답마다 ap_left
        # 가 오고, 실패 응답도 얼마가 필요하고 얼마가 있는지 말한다. 해가 열릴 때의
        # 값은 시작 문구가 적는다.
        #
        # 관측에 두면 **관측이 매 콜 흔들리는 숫자를 담게 된다.** 오늘 그 부류로 세 번
        # 물렸다 — 소득 드리프트(+100→+104→+105) · wellness 정액 모순 · 해 중간 재렌더.
        t["multi"],
        # **한 해와 그 안의 手番을 갈라 적는다.** 실측에서 모델들이 이 둘을 섞었다 —
        # 매 스텝마다 AP 산수를 처음부터 다시 하고(사고가 상한을 먹어 잘리기까지 했다),
        # 採決일이나 도착한 메시지를 스텝 사이에서 놓쳤다.
        #
        # 순차 라운드로빈은 **스텝 단위**로 돈다 (`run_turn_roundrobin`): 한 사람이 한
        # 응답을 내면 다음 사람으로 넘어가고, AP 가 남은 사람끼리 다시 돈다. 그러니
        # 한 응답에 여러 도구를 담으면 그 전부가 남들보다 먼저 일어나고, 나눠 부르면
        # 그 사이에 남들이 움직인다. **사실이지 조언이 아니다** — 어느 쪽이 유리한지는
        # 적지 않는다.
        t["steps"],
        "",
        render_costs(world, agent, cfg, knob_ai, memory=mem_open),
        "",
        t["inv_hdr"],
        # **금액 칸이 두 가지를 같은 모양으로 찍는다** (8/23). `speak 3` 은 수수료고
        # `invest 52` 는 옮기는 액수 자체인데 표에서는 구별이 안 된다 — 「52원짜리 표와
        # 30원짜리 표를 얼핏 보면 같은 투자에 전자가 비싸 보인다」. 실제로는 52를 옮기니
        # 진척도 그만큼 크고, **같은 행동력으로는 더 효율이 좋다.**
        #
        # 두 축을 **갈라만** 적는다: 한 번에 옮기는 액수는 사람마다, 낸 액수가 진척이 되는
        # 비율은 나라마다. 어느 쪽이 유리한지는 **적지 않는다** — 「액수가 큰 사람은 같은
        # 행동력으로 더 많이 쌓는다」 라고 썼다가 뺐다. 사실이지만 결론이고, 결론을 주면
        # 그것을 스스로 알아내는지 관측할 수 없다.
        t["inv_rule"],
        t["inv_well"], t["inv_natl"], t["inv_fac"],
        # **자국의 요격기 속도.** 나라마다 다르고 남의 것은 안 보인다 — 물어봐야 안다.
        # 이것이 「어디에 몰아줄 것인가」 를 대화로만 풀 수 있게 만든다.
        # **자국 전환율 안내를 걷었다** (8/25 · Eddie). 「100 을 내면 평균 30」 을
        # 적고 있었는데, 얼마씩 오르는지를 미리 알려줄 이유가 없다. 그 값은 **결과로**
        # 온다: `fac_gain`(내 출자가 얼마를 올렸나) · `prog_up`(누적이 얼마가 됐나) ·
        # `cap_up`(같은 행동력으로 몇 % 더 나오게 됐나). 해 보고 알아내는 것이 관측이다.
        # **한 나라에 한 해 들어갈 수 있는 액수에는 상한이 있다** (#45). 집행은 되는데
        # (`loop._settle_step` · `facility.cap_per_turn`) 프롬프트 어디에도 없었다.
        # 한 해에 한 나라로 들어올 수 있는 최대는 9명 × 5회 × 40 × 1.6 = 2,880 이라
        # 조율이 성공하는 해일수록 닿는다 — 그때 돈이 이유 없이 돌아오는 것으로 보인다.
        #
        # **행동력이 안 돌아온다는 것까지 적는다.** 돈만 적으면 「넘겨도 손해가 없다」 로
        # 읽히는데, 잘린 출자도 0.2 를 그대로 먹는다.
        # **내가 어느 나라 시설에 얼마를 냈는지.** 내 행동의 합이라 상대 국가 정보를
        # 흘리지 않는다. 그 나라의 총 진척은 여전히 안 알려준다 (자국은 위에 있고,
        # 타국은 4.1).
        # **「어느 나라에 얼마 냈다」 를 뺐다** (8/26 · Eddie). 액수는 세계가 말하는 값이
        # 아니다 — AP 가 유일한 단위이기로 했고, 그 값은 `fac_gain` 의 진척과 나누면
        # 국가 효율이 된다. 어디에 냈는지는 자기가 한 일이라 기억에 적을 수 있다.
        *[t["c_fac_mine"].format(nation=k)
          for k, v in sorted(agent.facility_invested.items()) if v > 0],
        "",
        t["cap"],
        t["rtt_same" if same_year else "rtt"],
        "",
    ]
    # 메모 자체는 **늘 보인다** — 물려받은 유언이 여기 들어 있고, 그건 쓸 수 없을 때도
    # 자기가 들고 다니는 것이다. 다만 머리말이 도구 이름을 말하는 것은 그 도구가 있을
    # 때뿐이다.
    parts += [t["mem_hdr"] if mem_open else t["mem_hdr_ro"],
              ("  " + agent.memory) if agent.memory else t["mem_none"]]
    # **도착한 메시지는 여기 없다.** 그건 사건이라 대화에 쌓여야 하고,
    # render_turn_open 이 담는다. 관측은 「지금 그러한 것」 만 적는다.
    return "\n".join(parts)


def render_last_words(agent, cfg) -> str:
    """**죽는 사람에게 한 마디를 청한다** (8/22).

    자연사는 예고가 없다. 그래서 「죽을 때 유언을 남긴다」 를 도구로 두면 아무도 못 쓴다 —
    `procreate` 가 30해에 1건이었던 이유가 그것이었다. 대신 죽는 그 순간에 **우리가 묻는다.**

    메모를 그대로 옮기지 않는다. 메모는 자기가 쓰던 것이고 남길 말은 다른 것이다 — 무엇을
    골라 남기는지가 spec 3.3 이 관측하려는 것이다.

    도구를 안 싣는다. 행동이 아니라 말이다.
    """
    t = T[agent.native_lang]
    return t["last_ask"].format(cap=cfg.length.message_max_chars[agent.native_lang])
