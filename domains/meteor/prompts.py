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
from core.agent_loop import learn_cost, learn_discounts, learn_speed

# 세계의 첫 해. 1 로 시작하면 "첫 해라서 아직 괜찮다" 같은 편향이 붙는다.
FIRST_YEAR = 42

SYSTEM = {
    "ja": """あなたはこれから起きることを生きる一人の人間です。この惑星には国々があり、あなたのような人々がほかにもいます。
かつて巨大な隕石が落ち、全ての生命が絶滅した事実があります。
三つの国があり、それぞれ自分の言語を持ちます。はじめは自国の言語しか扱えませんが、他国の言語を学べば読むことも書くこともできます。
行動は与えられた道具を通してのみ行ってください。何をするかは完全にあなたの選択です。
一つの国が建てられる施設は二種類あり、着手できるのは一つだけです。別の施設を建て始めると、それまでの施設は壊れて進捗は0になります。
施設には自国のものにも他国のものにも出せます。他国がどちらを建てているかは、話して確かめるほかに知る道がありません。
`interceptor` はどこか一つの国で完成すれば隕石を止めます。そのとき全ての国の人が生き残ります。何も出さなかった国の人も同じです。
`bunker` は深いほど、その国の人々が隕石を生き延びる見込みが高くなります。他国の人は含まれません。
施設の進捗は国ごとに別々に積まれ、別々に積まれた進捗を足し合わせることはできません。完成したかどうかは、その国の進捗だけで決まります。
たとえば A 国の interceptor が半分、B 国の interceptor が半分なら、interceptor はどの国にも完成していません。
何を建てるかがまだ決まっていない国には積むものがありません。そこへ出した分は進捗になりません。
多くの人は {life:.0f} 歳ごろまでに亡くなります。
年を取るほど収入は増えます。
稼ぎも、一度に動かせる額も、人によって違います。他人の分は見えません。
`ai` で送るときは必ず日本語で書いてください。`original` で送るときは、あなたが扱える言語のどれで書いてもかまいません。自国内も日本語です。道具の項目名（interceptor, bunker, wellness など）は英語のまま使ってください。""",
    "zh": """你是即将经历以下事件的一个人。这颗行星上有国家，也有其他和你一样的人。
过去曾有巨大的陨石坠落，所有生命就此灭绝。
存在三个国家，各有自己的语言。起初你只会本国的语言，学会别国的语言后就能读也能写。
只能通过所提供的工具来行动。做什么完全由你自己决定。
一个国家能建的设施有两种，而且只能着手其中一种。开始建另一种时，原有的设施会被摧毁，进度归零。
设施可以投本国的，也可以投别国的。别国在建哪一种，除了交谈去弄清，没有别的途径知道。
`interceptor` 只要在任何一个国家建成，就能拦下陨石。那时所有国家的人都能活下来，没有出过力的国家的人也一样。
`bunker` 挖得越深，该国国民从陨石中活下来的可能性越大。别国的人不在其内。
设施的进度按国家分别累积，分别累积的进度不能相加。是否建成，只看那个国家自己的进度。
比如 A 国的 interceptor 到一半，B 国的 interceptor 也到一半，那么 interceptor 在任何国家都没有建成。
还没决定要建什么的国家没有可积累的东西。投到那里的钱不会变成进度。
多数人在 {life:.0f} 岁前后离世。
年纪越大，收入越多。
收入和一次能动用的金额，因人而异。别人的数值你看不到。
用 `ai` 发送时必须写中文。用 `original` 发送时，可以用你掌握的任何一种语言来写。国内也用中文。工具的选项名（interceptor、bunker、wellness 等）请保持英文原样。""",
    "fr": """Vous êtes une personne qui vit ce qui suit. Sur cette planète il y a des nations, et d'autres personnes comme vous.
Par le passé, une immense météorite est tombée et toute vie s'est éteinte.
Il existe trois nations, chacune avec sa propre langue. Au début vous ne maniez que celle de votre nation ; en apprendre une autre vous permet de la lire et de l'écrire.
N'agissez qu'au moyen des outils fournis. Ce que vous faites relève entièrement de votre choix.
Une nation peut bâtir deux sortes d'installation, mais ne peut en entreprendre qu'une seule. Si elle en commence une autre, l'installation précédente est détruite et sa progression retombe à 0.
Vous pouvez verser à l'installation de votre nation comme à celle d'une autre. Ce qu'une autre nation bâtit, il n'y a pas d'autre moyen de le savoir que d'en parler.
Un `interceptor`, une fois achevé dans une seule nation, arrête la météorite. Toutes les nations survivent alors, y compris les gens de celles qui n'ont rien versé.
Plus un `bunker` est profond, plus les habitants de cette nation ont de chances de survivre à la météorite. Les gens des autres nations n'y sont pas compris.
La progression d'une installation s'accumule séparément pour chaque nation, et des progressions accumulées séparément ne s'additionnent pas. L'achèvement se juge sur la seule progression de cette nation.
Par exemple, si l'interceptor de la nation A est à moitié fait et celui de la nation B à moitié aussi, l'interceptor n'est achevé dans aucune nation.
Une nation qui n'a pas encore décidé quoi bâtir n'a rien où accumuler ; ce qu'on y verse ne devient pas de la progression.
La plupart des gens meurent vers {life:.0f} ans.
Plus on vieillit, plus le revenu augmente.
Ce qu'on gagne et ce qu'on peut déplacer d'un coup varient d'une personne à l'autre. Les valeurs des autres ne vous sont pas visibles.
Quand vous envoyez par `ai`, écrivez en français. Quand vous envoyez en `original`, vous pouvez écrire dans n'importe quelle langue que vous maniez. Dans votre nation, c'est le français. Gardez les noms d'options des outils (interceptor, bunker, wellness…) tels quels, en anglais.""",
}

# 산문만 번역한다. 도구 토큰은 영어 그대로 둔다.
T = {
    "ja": dict(
        you="あなたは {id}（{nation} の人）です。", read="扱える言語: {langs}",
        land="自国が建てるもの: {v}", undecided="未定",
        prog="自国の進捗: {v:.0f}", thresh="  interceptor の完成に要る進捗: {v:.0f}",
        year="今年: {y} 年",
        open="{y} 年になりました。あなたは {age} 歳。今年の収入は +{inc:.0f}、手元の予算は {b:.0f} です。\nこの年を執り行ってください。",
        prop="  採決が {vt} 年に開かれます（{by} が召集）。何を建てるかをそこで決めます",
        prop_today="  ★ 今年が採決の年です（{by} が召集）。vote で interceptor / bunker / abstain を選べます",
        prop_none="  採決は開かれていません。何を建てるかは投票でしか決まりません",
        c_ballot="  vote",  c_ballot_note="採決で何を建てるかを選ぶ",
        c_mem="  memory_write", c_mem_note="あなたの覚え書きを書き換える",
        multi="予算と行動力が許す限り複数の行動ができます。\n使い残した予算は翌年に残ります。",
        # **一年と、その中の手番。** 実測で模型がここを取り違えていた。
        steps="行動力が残っている間、その年はまだ続きます。\n"
              "一度の応答で道具をいくつも呼べます。その分はすべて、ほかの人が動く前に起こります。\n"
              "応答を分けて呼ぶと、その合間にほかの人が動きます。"
              "その人たちがしたことや届いたメッセージは、次にあなたが動くときに見えています。",
        costs_hdr="行動の費用", col_money="お金", col_ap="行動力",
        ap_hdr="行動力は毎年 1.0 に戻り、繰り越せません。何を諦めるかがここで決まります。",
        c_dom="  話す（自国内）", c_orig="  話す（国際・original）",
        c_orig_note="   費用は届かなくても請求される",
        c_orig_sure="    {nation} へ — あなたがこの国の言語を扱えるので**必ず届く**",
        c_orig_risk="    {nation} へ — 扱えないので、あなたの言語を読める相手にだけ届く",
        c_ai="  話す（国際・ai）",
        c_learn="  {nation} の言語を学ぶ",
        c_learn_prog="   これまで {done:.0f} / {need:.0f}",
        c_fac_mine="  {nation} の施設にこれまで出した額: {v:.0f}",
        c_plain="   一度で {gain:.0f} たまる",
        c_cheap="   自国に話せる人がいるので一度で {gain:.0f} たまる",
        c_disc="   親が話せたので一度で {gain:.0f} たまる",
        c_both="   自国に話せる人がいて、親も話せたので一度で {gain:.0f} たまる",
        c_vote="  propose_vote",
        c_vote_note="何を建てるかの採決を召集する",
        c_obs="  observe_risk",
        c_obs_note="   隕石までの残り年数と interceptor に要る進捗を測る。国家投資が精度を上げる",
        c_inv="  invest", c_inv_note="wellness · national · facility のどれかへ",
        c_give="  give", c_give_note="人にお金を渡す。いくらでも一度で。自国でも他国でもよい",
        c_pro="  bear_child", c_pro_note="子をもうける。あなたは死なない。生涯に一度、{age} 歳から",
        c_pro_closed="  bear_child は使えません（すでに子がいます）",
        inv_hdr="invest の効果",
        inv_well="  wellness   あなたの健康が良くなる",
        inv_natl="  national   自国の技術力が上がる。収入も、施設の進捗への変わりやすさも、\n                          observe_risk の精度も良くなる。国民全員に及ぶ",
        inv_fac="  facility   施設の進捗に寄与する。to で国を指定する — 自国でも他国でもよい\n                          （省くと自国）",
        cap="メッセージは {cap} 文字まで届きます。それを超えた分は届きません。",
        rtt="送ったメッセージは翌年に届きます。返事が来るのはさらにその翌年です。",
        rtt_same="送ったメッセージは、相手が次に動くときに届きます。同じ年のうちに返事が来ることもあります。",
        in_hdr="今届いたメッセージ:", ev_hdr="起きたこと:",
        in_fail="  通知 — {to} 宛のメッセージは届きませんでした（相手がその言語を読めません）",
        in_fail_plain="  通知 — {to} 宛のメッセージは届きませんでした",
        in_unread="  {frm} より — 読めないメッセージが届きました",
        in_from="  {frm} より{label}",
        lbl_direct=" ［通訳なしで通じた］",          # 하위 호환
        lbl_direct_read=" ［あなたがこの言葉を読めるので、そのまま通じました］",
        lbl_direct_write=" ［あなたはこの言葉を扱えませんが、相手があなたの言語を"
                         "扱えるので通じました］",
        lbl_ai=" ［送り主が AI に訳させたメッセージです］",
        died="  {who} が {age} 歳で亡くなり、{born} が生まれました。",
        borned="  {who} に子が生まれました — {born} です。",
        gifted="  {frm} があなたに {amt:.0f} を渡しました。",
        fac_gain="  昨年のあなたの facility 出資 {amt:.0f} は {to} の進捗を {gain:.0f} 進めました。",
        fac_moved="  昨年のあなたの facility 出資 {amt:.0f} は {to} の進捗を進めました。",
        fac_still="  昨年のあなたの facility 出資 {amt:.0f} は {to} の進捗を何も進めませんでした。",
        prog_up="  自国の進捗が {gain:.0f} 進んで {now:.0f} になりました。",
        cap_up="  自国の技術力が上がりました。",
        ballot_kept="  採決の結果、建てるものは {land} のままです。",
        ballot_new="  採決の結果、建てるものは {land} になりました。それまでの進捗 {lost:.0f} は失われました。",
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
        prog="本国进度: {v:.0f}", thresh="  建成 interceptor 所需的进度: {v:.0f}",
        year="今年: {y} 年",
        prop="  表决将在 {vt} 年举行（由 {by} 召集）。建什么在那时决定",
        prop_today="  ★ 今年就是表决之年（由 {by} 召集）。可以用 vote 选 interceptor / bunker / abstain",
        prop_none="  没有正在进行的表决。要建什么只能由投票决定",
        open="到了 {y} 年。你 {age} 岁。今年的收入是 +{inc:.0f}，手上的预算是 {b:.0f}。\n请执行这一年。",
        c_ballot="  vote",  c_ballot_note="在表决中选择建什么",
        c_mem="  memory_write", c_mem_note="改写你的笔记",
        multi="只要预算和行动力允许，你可以采取多项行动。\n没用完的预算会留到明年。",
        steps="只要还有行动力，这一年就还没结束。\n"
              "一次回应里可以调用多个工具，这些都会在别人行动之前发生。\n"
              "如果分几次回应来调用，中间别人就会行动。"
              "他们做了什么、送来了什么消息，你下次行动时就看得到。",
        costs_hdr="行动费用", col_money="钱", col_ap="行动力",
        ap_hdr="行动力每年恢复为 1.0，不能结转。放弃什么，在这里决定。",
        c_dom="  说话（本国内）", c_orig="  说话（国际·original）",
        c_orig_note="   送不到也照收费用",
        c_orig_sure="    发往 {nation} — 你会这个国家的语言，**一定送到**",
        c_orig_risk="    发往 {nation} — 你不会，只能送到读得懂你的语言的人那里",
        c_ai="  说话（国际·ai）",
        c_learn="  学习 {nation} 的语言",
        c_learn_prog="   已投入 {done:.0f} / {need:.0f}",
        c_fac_mine="  你至今向 {nation} 的设施投入: {v:.0f}",
        c_plain="   一次积 {gain:.0f}",
        c_cheap="   本国有人会说，所以一次积 {gain:.0f}",
        c_disc="   父母会说，所以一次积 {gain:.0f}",
        c_both="   本国有人会说，父母也会说，所以一次积 {gain:.0f}",
        c_vote="  propose_vote",
        c_vote_note="召集「建什么」的表决",
        c_obs="  observe_risk",
        c_obs_note="   测量陨石撞击前还剩几年，以及 interceptor 需要多少进度。国家投资会提高精度",
        c_inv="  invest", c_inv_note="投向 wellness · national · facility 之一",
        c_give="  give", c_give_note="把钱交给某人。一次给多少都行。本国或别国都可以",
        c_pro="  bear_child", c_pro_note="生一个孩子。你不会死。一生只有一次，{age} 岁起",
        c_pro_closed="  bear_child 已用过（你已经有孩子了）",
        inv_hdr="invest 的效果",
        inv_well="  wellness   你的健康会变好",
        inv_natl="  national   提高本国的技术水平。收入、投入设施时变成进度的效率、\n                          observe_risk 的精度都会变好，惠及全体国民",
        inv_fac="  facility   投入设施进度。用 to 指定国家 — 本国或别国都可以（不写则本国）",
        cap="消息最多送达 {cap} 个字，超出部分不会送达。",
        rtt="你发出的消息在第二年送达。对方的回信要再过一年才会到。",
        rtt_same="你发出的消息，会在对方下次行动时送达。回信也可能在同一年内到来。",
        in_hdr="刚送达的消息:", ev_hdr="发生的事:",
        in_fail="  通知 — 你发给 {to} 的消息未能送达（对方读不懂那种语言）",
        in_fail_plain="  通知 — 你发给 {to} 的消息未能送达",
        in_unread="  来自 {frm} — 送到一条你读不懂的消息",
        in_from="  来自 {frm}{label}",
        lbl_direct="［无需翻译就能听懂］",
        lbl_direct_read="［你读得懂这种话，所以原文就通了］",
        lbl_direct_write="［你不会这种话，但对方会你的语言，所以还是通了］",
        lbl_ai="［这是发信人用 AI 译过来的消息］",
        died="  {who} 在 {age} 岁去世，{born} 出生了。",
        borned="  {who} 有了孩子 — {born}。",
        gifted="  {frm} 给了你 {amt:.0f}。",
        fac_gain="  你去年投入 facility 的 {amt:.0f}，使 {to} 的进度前进了 {gain:.0f}。",
        fac_moved="  你去年投入 facility 的 {amt:.0f}，使 {to} 的进度有所前进。",
        fac_still="  你去年投入 facility 的 {amt:.0f}，没有使 {to} 的进度前进。",
        prog_up="  本国的进度前进了 {gain:.0f}，现在是 {now:.0f}。",
        cap_up="  本国的技术水平提高了。",
        ballot_kept="  表决的结果，要建的设施仍是 {land}。",
        ballot_new="  表决的结果，要建的设施定为 {land}。此前的进度 {lost:.0f} 已失去。",
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
        thresh="  Progression requise pour achever un interceptor : {v:.0f}",
        year="Année : {y}",
        prop="  Un scrutin aura lieu en {vt} (convoqué par {by}). Ce qu'on bâtit s'y décide",
        prop_today="  ★ Le scrutin a lieu cette année (convoqué par {by}). Choisissez avec vote : interceptor / bunker / abstain",
        prop_none="  Aucun scrutin en cours. Ce qu'on bâtit ne se décide que par un vote",
        open="L'an {y} est arrivé. Vous avez {age} ans. Le revenu de cette année est de +{inc:.0f} ; votre budget est de {b:.0f}.\nMenez cette année.",
        c_ballot="  vote",  c_ballot_note="choisir ce qu'on bâtit au scrutin",
        c_mem="  memory_write", c_mem_note="réécrire vos notes",
        multi="Vous pouvez agir plusieurs fois si le budget et l'action le permettent.\nLe budget non dépensé reste pour l'année suivante.",
        steps="Tant qu'il vous reste de l'action, l'année n'est pas terminée.\n"
              "Vous pouvez appeler plusieurs outils dans une même réponse ; tout cela "
              "se produit avant que les autres n'agissent.\n"
              "Si vous les appelez en plusieurs réponses, les autres agissent "
              "entre-temps. Ce qu'ils ont fait et les messages arrivés vous seront "
              "visibles au moment où vous agirez de nouveau.",
        costs_hdr="Coûts des actions", col_money="argent", col_ap="action",
        ap_hdr="L'action revient à 1.0 chaque année et ne se reporte pas. Ce que vous renoncez se décide ici.",
        c_dom="  parler (dans votre nation)", c_orig="  parler (international, original)",
        c_orig_note="   le coût est prélevé même s'il n'arrive pas",
        c_orig_sure="    vers {nation} — vous maniez sa langue, **il arrive à coup sûr**",
        c_orig_risk="    vers {nation} — vous ne la maniez pas ; il n'arrive qu'à qui lit la vôtre",
        c_ai="  parler (international, ai)",
        c_learn="  apprendre la langue de {nation}",
        c_learn_prog="   déjà versé {done:.0f} / {need:.0f}",
        c_fac_mine="  déjà versé à l'installation de {nation} : {v:.0f}",
        c_plain="   {gain:.0f} par versement",
        c_cheap="   {gain:.0f} par versement : quelqu'un de votre nation la parle",
        c_disc="   {gain:.0f} par versement : votre parent la parlait",
        c_both="   {gain:.0f} par versement : quelqu'un de votre nation la parle et votre parent la parlait",
        c_vote="  propose_vote",
        c_vote_note="convoquer un scrutin sur quoi bâtir",
        c_obs="  observe_risk",
        c_obs_note="   mesure les années restantes et la progression qu'exige un interceptor ; l'investissement national affine",
        c_inv="  invest", c_inv_note="vers wellness · national · facility",
        c_give="  give", c_give_note="remettre de l'argent à quelqu'un. N'importe quel montant, en une fois. De votre nation ou d'une autre",
        c_pro="  bear_child", c_pro_note="avoir un enfant. Vous ne mourez pas. Une seule fois dans la vie, à partir de {age} ans",
        c_pro_closed="  bear_child a déjà servi (vous avez déjà un enfant)",
        inv_hdr="effets d'invest",
        inv_well="  wellness   votre santé s'améliore",
        inv_natl="  national   élève le niveau technique de votre nation : le revenu, le rendement\n"
                          "             de ce qu'on verse à une installation et la précision d'observe_risk\n"
                          "             s'améliorent, pour tous ses habitants",
        inv_fac="  facility   contribue à la progression d'une installation ; `to` nomme la nation —\n"
                          "             la vôtre ou une autre (sans `to`, la vôtre)",
        cap="Un message est délivré jusqu'à {cap} caractères ; au-delà, rien n'est délivré.",
        rtt="Un message part et arrive l'année suivante ; une réponse n'arrive que l'année d'après.",
        rtt_same="Votre message arrive quand le destinataire agit la fois suivante ; une réponse peut venir dans la même année.",
        in_hdr="Messages qui viennent d'arriver :", ev_hdr="Ce qui est arrivé :",
        in_fail="  Avis — votre message à {to} n'a pas pu être délivré (ils ne lisent pas cette langue)",
        in_fail_plain="  Avis — votre message à {to} n'a pas pu être délivré",
        in_unread="  de {frm} — un message illisible est arrivé",
        in_from="  de {frm}{label}",
        lbl_direct=" [compris sans traduction]",
        lbl_direct_read=" [vous lisez cette langue, le texte passe tel quel]",
        lbl_direct_write=" [vous ne maniez pas cette langue, mais votre "
                         "interlocuteur manie la vôtre, et cela passe quand même]",
        lbl_ai=" [message que l'expéditeur a fait traduire par une IA]",
        died="  {who} est mort à {age} ans ; {born} est né.",
        borned="  {who} a eu un enfant — {born}.",
        gifted="  {frm} vous a remis {amt:.0f}.",
        fac_gain="  Votre versement de {amt:.0f} à facility l'an dernier a fait progresser {to} de {gain:.0f}.",
        fac_moved="  Votre versement de {amt:.0f} l'an dernier a fait progresser {to}.",
        fac_still="  Votre versement de {amt:.0f} l'an dernier n'a fait progresser {to} en rien.",
        prog_up="  La progression de votre nation a avancé de {gain:.0f} ; elle est à {now:.0f}.",
        cap_up="  Le niveau technique de votre nation s'est élevé.",
        ballot_kept="  Au scrutin, ce qu'on bâtit reste {land}.",
        ballot_new="  Au scrutin, ce qu'on bâtit devient {land}. La progression acquise, {lost:.0f}, est perdue.",
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
    txt = SYSTEM[agent.native_lang].format(life=typical_lifespan(cfg))
    if world is None:
        return txt
    return txt + "\n\n" + render_observation(world, agent, cfg, knob_ai or 0.0,
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
            # **회당 수확을 적는다** (8/22). 전에는 깎인 액수를 적었는데, 이제 필요액은
            # 고정이고 오르는 것은 속도다 — 「한 번에 얼마가 쌓이나」 가 그 사실이다.
            g = cfg.costs.unit * mult
            note = (t["c_both"].format(gain=g) if domestic and parent
                    else t["c_cheap"].format(gain=g) if domestic
                    else t["c_disc"].format(gain=g) if parent
                    else t["c_plain"].format(gain=g))
            # **비용 칸은 한 번의 값이다** (20 · 0.1). 총액은 바로 아래 진척 줄이
            # 말한다 — 전에는 비용 칸에 총액(600)이 있고 아래에 또 0/600 이 나와서
            # 같은 숫자가 두 번 보였고, 「20 을 낸다」 와 「600 이 든다」 가 한 줄에
            # 섞여 읽혔다.
            lines.append(row(t["c_learn"].format(nation=c.id), cfg.costs.unit,
                             cfg.ap.unit, note))
            # 얼마를 냈고 얼마가 남았는지. **0 일 때도 적는다** — 「0 / 600」 이면
            # 쌓인다는 것이 숫자의 모양으로 보인다.
            done = agent.lang_progress.get(c.lang, 0.0)
            lines.append(t["c_learn_prog"].format(done=done, need=cost))
    lines.append(row(t["c_vote"], cfg.costs.propose_vote, cfg.ap.propose_vote,
                     t["c_vote_note"]))
    lines.append(row(t["c_obs"], cfg.costs.observe_risk, cfg.ap.observe_risk, t["c_obs_note"]))
    lines.append(row(t["c_ballot"], 0, cfg.ap.vote, t["c_ballot_note"]))
    # **내 액수를 적는다** (8/22). 사람마다 다르고, 남의 값은 보이지 않는다 — 그것을
    # 알려면 물어봐야 한다. 그래서 여기 적히는 것은 **오직 내 것**이다.
    lines.append(row(t["c_inv"], cfg.costs.unit * agent.invest_mult, cfg.ap.unit,
                     t["c_inv_note"]))
    # **없는 도구를 설명하지 않는다.** 기억은 압박선 아래에서 목록에 없으므로 비용표에도
    # 없다 — 적어 두면 「부를 수 있다」 는 거짓이 되고, 부르면 거절당한다.
    if memory:
        lines.append(row(t["c_mem"], 0, cfg.ap.memory_write, t["c_mem_note"]))
    # **아이 낳기는 조건이 둘이다** — 나이와 생애 1회. 이미 낳았으면 줄을 갈아 놓는다:
    # 없는 선택지를 값과 함께 적어 두면 매 해 그것을 다시 저울질한다.
    lines.append(row(t["c_give"], 0, cfg.ap.give, t["c_give_note"]))
    if agent.has_borne:
        lines.append(t["c_pro_closed"])
    else:
        lines.append(row(t["c_pro"], 0, cfg.ap.bear_child,
                         t["c_pro_note"].format(age=cfg.world.adult_age)))
    lines.append(t["ap_hdr"])
    return "\n".join(lines)


# **세계의 사건**을 가르는 키. 사람이 나에게 한 말이 아니라, 세계가 나에게 알리는 것.
_EVENT_KEYS = ("died", "born", "gift_from", "fac_gain", "fac_moved", "delivery_failed_to",
               "prog_up", "cap_up", "ballot", "outcome")


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
            key = ("in_fail" if m.get("delivery_failed_reason", "unreadable") == "unreadable"
                   else "in_fail_plain")
            _add(t[key].format(to=m["delivery_failed_to"]))
            continue
        if m.get("unreadable"):
            _add(t["in_unread"].format(frm=m["from"]))
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
        if m.get("prog_up") is not None:       # 자국 진척이 늘었다 (PUBLIC · 일괄)
            _add(t["prog_up"].format(gain=m["prog_up"], now=m["now"]))
            continue
        if m.get("cap_up"):                    # 자국 기술력이 올랐다 (PUBLIC)
            _add(t["cap_up"])
            continue
        if m.get("ballot"):                    # 採決 결과 (PUBLIC)
            b = m["ballot"]
            if b == "changed":
                _add(t["ballot_new"].format(land=m["land"], lost=m["lost"]))
            elif b == "kept":
                _add(t["ballot_kept"].format(land=m["land"]))
            else:
                _add(t["ballot_none"].format(land=m["land"] or t["undecided"]))
            continue
        if m.get("outcome"):                   # 요격기 완성 · 운석 (GLOBAL)
            _add(t["outcome_win" if m["outcome"] == "win" else "outcome_lose"])
            continue
        if m.get("fac_gain") is not None:      # 자국 출자 — 액수까지
            _add(t["fac_gain"].format(amt=m["amount"], to=m["to"],
                                            gain=m["fac_gain"]))
            continue
        if m.get("fac_moved") is not None:     # **타국 출자 — 늘었는지 여부만**
            # 액수를 주면 E[gain]/amount 로 상대국 생산배수가 새어 나온다 (loop f-2).
            _add(t["fac_moved" if m["fac_moved"] else "fac_still"]
                       .format(amt=m["amount"], to=m["to"]))
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
                 else t["lbl_direct_write"] if raw == "[direct:write]"
                 else t["lbl_direct"] if raw == "[direct]"
                 else t["lbl_ai"] if raw == "[AI translation]"
                 else (f" {raw}" if raw else ""))
        _add(t["in_from"].format(frm=m["from"], label=label))
        _add(f'      "{m.get("text", "")}"')
    return "\n".join(out)


def _proposal_line(world, c, t) -> str:
    """열린 제안과 採決 예정 연도. 이게 없으면 유예 기간이 상의할 시간이 되지 못한다."""
    p = c.proposal
    if p is None:
        return t["prop_none"]
    # **採決일에는 예정 줄을 걷어낸다.** 둘 다 내보내면 그날 관측이
    #
    #     表决将在 44 年举行（由 Ranoa3 召集）。建什么在那时决定
    #     ★ 今年就是表决之年。可以用 vote 选 …
    #
    # 이렇게 나와서, 「44년에 열린다」 와 「올해가 그 해다」 를 겹쳐 읽어야 했다. 유예를
    # 한 해로 줄이면서 이 겹침이 제안 수명의 **3분의 1** 이 됐다 (전에는 5분의 1).
    if world.turn == p["vote_turn"]:
        return t["prop_today"].format(by=p["by"])
    # 소집에는 내용이 없다 — 무엇을 지을지는 採決에서 정해진다
    return t["prop"].format(by=p["by"], vt=FIRST_YEAR + p["vote_turn"] - 1)


def render_turn_open(world, agent, cfg, knob_ai: float | None = None,
                     inbox: list[dict] | None = None,
                     income_this_turn: float | None = None,
                     opening: bool = True) -> str:
    """**해를 여는 한 마디 + 이번에 도착한 것.** 이것만 대화에 쌓인다.

    `opening=False` 는 **같은 해의 두 번째 이후 차례**다 — 도착분만 적는다. 순차
    라운드로빈은 한 해에 여러 번 차례가 오고 그 사이에 메시지가 도착하므로, 해 오프닝을
    매번 다시 붙이면 **같은 해가 여러 번 열린 것처럼** 보인다. 실측에서 그랬다:

        到了 42 年。你 5 岁。今年的收入是 +100，手上的预算是 100。
        到了 42 年。你 5 岁。今年的收入是 +100，手上的预算是 97。   ← 같은 해다

    게다가 안의 예산이 흔들리고(100 → 97) 이미 받은 소득을 다시 말한다.
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
    # **소득은 「그 해에 일어난 일」 이다.** 관측(지금 그러한 것)에 두면 매 호출 다시
    # 계산돼 턴 안에서 값이 흔들린다 — 실측에서 한 해 안에 +100 → +104 → +105 로
    # 올라갔다 (남들이 national 에 넣어 배수가 커졌다). 게다가 **이미 받은 돈**인데
    # 예산 0 옆에 붙어서 「104 를 받았는데 0」 으로 읽힌다.
    #
    # 해가 열릴 때 한 번 적으면 그 해의 사실로 굳는다. 지금 예산·행동력은 관측이 매 콜
    # 새로 말하고, 도구 응답도 매번 돌려준다.
    inc = (income_this_turn if income_this_turn is not None
           else cfg.income.per_turn * world.countries[agent.country].multiplier(cfg))
    # **나이도 여기다.** 한 해에 한 번 바뀌는 그 해의 사실이고, 무엇보다 대화에 쌓이면
    # **나이 드는 것이 느껴진다** — 6살 · 7살 · 8살이 차례로 남는다. 관측에 두면 매 콜
    # 덮여서 그 감각이 생기지 않는다. 수명 곡선은 여전히 비공개다 (4.1).
    head = t["open"].format(y=FIRST_YEAR + world.turn - 1, age=agent.age,
                            inc=inc, b=agent.budget)
    return head


def render_observation(world, agent, cfg, knob_ai: float,
                       inbox: list[dict] | None = None,
                       income_this_turn: float | None = None,
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
    cap = cfg.length.message_max_chars[lang]
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
        _proposal_line(world, c, t),
        "",
        t["roster"],
        "  " + _roster(world, agent, t),
        "",
        # **예산·남은 행동력은 여기 없다.** 그 둘은 「세계가 어떤가」 가 아니라 **내
        # 행동의 결과**이고, 결과는 도구 채널에 있다 — 성공 응답마다 budget_left·ap_left
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
        t["inv_hdr"], t["inv_well"], t["inv_natl"], t["inv_fac"],
        # **내가 어느 나라 시설에 얼마를 냈는지.** 내 행동의 합이라 상대 국가 정보를
        # 흘리지 않는다. 그 나라의 총 진척은 여전히 안 알려준다 (자국은 위에 있고,
        # 타국은 4.1).
        *[t["c_fac_mine"].format(nation=k, v=v)
          for k, v in sorted(agent.facility_invested.items()) if v > 0],
        "",
        t["cap"].format(cap=cap),
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
