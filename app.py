"""
app.py — 대전환기 국가 재난관리 역량 강화 방안 AHP 조사 (Streamlit + Turso)

엑셀 템플릿 로직을 그대로 반영:
  · 9-8-…-1(동등)-…-8-9 좌우 대칭 척도 쌍대비교
  · geomean 가중치, CR 10% 기준, RCI n=1~15
  · 3계층(목표 → 5개 분야 → 세부과제) + 분야별 장기/중기/단기 시급성 문항

배포:
  관리자   https://<app>.streamlit.app/            (ADMIN_PASSWORD)
  응답자   https://<app>.streamlit.app/?respond=1   (누구나, 로그인 불필요)

secrets.toml
  ADMIN_PASSWORD     = "..."
  TURSO_DATABASE_URL = "libsql://xxxx.turso.io"
  TURSO_AUTH_TOKEN   = "..."
"""

from __future__ import annotations

import io
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import ahp_core as C
import ahp_db as DB

# ----------------------------------------------------------------------
# 조사 구조 (엑셀 템플릿에서 추출한 기본값 — 관리자가 편집 가능)
# ----------------------------------------------------------------------

DEFAULT_MODEL = {
    "title": "대전환기 국가 재난관리 역량 강화 방안 정책 우선순위 AHP 조사",
    "goal": "대전환기 국가 재난관리 역량 강화 방안",
    "timeframe_question": True,
    "categories": [
        {"id": "C1", "name": "법·제도 정비", "items": [
            "'기후재난' 법적 정의 신설 및 법률간 연계", "기후취약계층 정의·지원기준 통합",
            "폭염한파쉼터 등 행정지침 의존 지원제도의 법제화", "기후적응-재난관리-취약계층지원 정책 연계",
            "국가안전관리계획 기후적응 주류화평가 도입"]},
        {"id": "C2", "name": "거버넌스·조정체계 강화", "items": [
            "국가/지자체 기후위기 조정권한 강화 및 이행점검 실질화", "'적응적 복구' 기반 기후적응-재난 통합운영체계 구축",
            "자연재난관리실 기능의 기후재난주류화", "지자체 기후·재난·복지부서 간 협력 강화",
            "지자체 간 협력 및 민관 협의체 활성화"]},
        {"id": "C3", "name": "중앙-지방 재정·조직 지원", "items": [
            "지자체 기후재난 전담조직 및 인력 확충 기준 마련", "지자체 특별교부세·전용보조금 신설",
            "기후취약성 가중 인프라 투자 및 예산 배정", "기후적응·재난대응·취약계층지원 중앙사무 지자체 연계집행 확립"]},
        {"id": "C4", "name": "기후취약계층 통합지원체계", "items": [
            "취약계층 사전 발굴·등록 체계 구축", "에너지바우처 등 지원제도 재설계",
            "기후위기 인지도 제고 및 푸시형 정보 제공 체계 구축", "부처 간 기후·취약계층 정보 연계 강화",
            "취약성 중첩평가 기반 주의·집중관리 사례관리 체계 구축"]},
        {"id": "C5", "name": "데이터 기반 구축 및 역량·인식 제고", "items": [
            "기후격차 데이터 플랫폼·대시보드 구축", "AI·디지털트윈 기반 예측형 재난대응 고도화",
            "전담인력 확충 및 전문교육 운영", "취약계층 맞춤형 이해도 제고 콘텐츠 보급"]},
    ],
}

METHOD = "geometric"        # 템플릿과 동일: 기하평균
CR_LIMIT = 0.10             # 템플릿 기준: 10%
BRAND = "#009FF4"

VERBAL = {1: "동등", 2: "약간 우세", 3: "약간 중요", 4: "다소 중요", 5: "중요",
          6: "꽤 중요", 7: "매우 중요", 8: "훨씬 중요", 9: "절대적으로 중요"}

st.set_page_config(page_title="재난관리 AHP 조사", page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html,body,[class*="css"]{{font-family:'Pretendard',-apple-system,sans-serif}}
.block-container{{padding-top:2rem;max-width:1120px}}
h1,h2,h3{{letter-spacing:-.02em}}
.ahp-ok{{background:#E7F6EE;color:#0F9D58;padding:3px 11px;border-radius:999px;font-size:.8rem;font-weight:700}}
.ahp-bad{{background:#FDECEA;color:#D93025;padding:3px 11px;border-radius:999px;font-size:.8rem;font-weight:700}}
.pair-a{{text-align:right;font-weight:700;color:#0F2B46}}
.pair-b{{text-align:left;font-weight:700;color:#0F2B46}}
.hint{{color:#64748b;font-size:.85rem}}
.suggest-box{{border-left:3px solid #B26A00;background:#FFFBF3;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:.9rem}}
</style>
""", unsafe_allow_html=True)

DB.init_db()


# ----------------------------------------------------------------------
# 척도 변환 (9-8-…-1-…-8-9 슬라이더 ↔ 행렬값)
# ----------------------------------------------------------------------

def slider_to_ratio(v: int) -> float:
    """슬라이더 값 v(-8..8) → 쌍대비교값 a_ij (i=왼쪽 항목, j=오른쪽 항목).
    왼쪽으로 끌수록(v<0) 왼쪽 항목이, 오른쪽으로 끌수록(v>0) 오른쪽 항목이 더 중요.
    v<0 → 왼쪽(i) (|v|+1)배,  v=0 → 동등,  v>0 → 오른쪽(j) (v+1)배."""
    v = int(v)
    if v == 0:
        return 1.0
    return float(abs(v) + 1) if v < 0 else 1.0 / (v + 1)


def ratio_to_slider(a: float) -> int:
    if abs(a - 1) < 1e-9:
        return 0
    # a>1 → 왼쪽(i) 우세 → 음수,  a<1 → 오른쪽(j) 우세 → 양수
    return -(int(round(a)) - 1) if a > 1 else (int(round(1 / a)) - 1)


# 슬라이더 눈금: 왼쪽 -9…-2 (왼쪽 항목 9~2배), 가운데 1(동등), 오른쪽 2…9 (오른쪽 항목 2~9배)
SLIDER_OPTS = list(range(-8, 9))   # 내부값 v, 좌→우 = -8..8


def _disp_label(v: int) -> str:
    v = int(v)
    if v < 0:
        return str(v - 1)   # -1→-2 … -8→-9
    if v > 0:
        return str(v + 1)   #  1→2  …  8→9
    return "1"              # 동등


# ----------------------------------------------------------------------
# 구조 → 비교 묶음
# ----------------------------------------------------------------------

def model_to_tree(m: dict) -> C.Node:
    root = C.Node(id="GOAL", name=m["goal"])
    for c in m["categories"]:
        cat = C.Node(id=c["id"], name=c["name"])
        for k, it in enumerate(c["items"]):
            cat.children.append(C.Node(id=f"{c['id']}I{k+1}", name=it if isinstance(it, str) else it["name"]))
        root.children.append(cat)
    return root


def comparison_nodes(m: dict) -> List[dict]:
    nodes = []
    ncat = len(m["categories"])
    if ncat >= 2:
        nodes.append({"id": "GOAL", "kind": "cat",
                      "title": f"{m['goal']} — {ncat}개 분야의 상대적 중요도",
                      "context": "정책 목표 달성 관점에서",
                      "items": [c["name"] for c in m["categories"]]})
    for c in m["categories"]:
        items = [it if isinstance(it, str) else it["name"] for it in c["items"]]
        if len(items) >= 2:
            nodes.append({"id": c["id"], "kind": "item", "cat": c["id"],
                          "title": f"「{c['name']}」 하위 과제의 우선순위",
                          "context": f"「{c['name']}」 분야 내에서", "items": items})
    return nodes


def flow_steps(m: dict) -> List[tuple]:
    """응답 진행 순서. '분야 간 중요도 → (분야별: 하위 과제 중요도 → 그 분야 시급성)'.

    반환: [(kind, ref), ...]
      kind='pair' → ref = 비교 묶음 노드(dict)
      kind='tf'   → ref = 분야 인덱스(int)
    """
    by_id = {n["id"]: n for n in comparison_nodes(m)}
    tfq = m.get("timeframe_question")
    flow: List[tuple] = []
    if "GOAL" in by_id:                       # 1) 분야 간 중요도 → 분야 시급성
        flow.append(("pair", by_id["GOAL"]))
        if tfq:
            flow.append(("tfg", None))
    for ci, c in enumerate(m["categories"]):  # 2) 분야마다 과제 중요도 → 그 분야 과제 시급성
        if c["id"] in by_id:
            flow.append(("pair", by_id[c["id"]]))
        if tfq:
            flow.append(("tf", ci))
    return flow


def total_pairs(m: dict) -> int:
    return sum(len(n["items"]) * (len(n["items"]) - 1) // 2 for n in comparison_nodes(m))


def item_names(m: dict) -> Dict[str, str]:
    out = {}
    for c in m["categories"]:
        for k, it in enumerate(c["items"]):
            out[f"{c['id']}I{k+1}"] = it if isinstance(it, str) else it["name"]
    return out


# ----------------------------------------------------------------------
# 응답 → 로컬/전역 가중치
# ----------------------------------------------------------------------

def response_matrices(m: dict, comps: List[dict]):
    """comps: [{rid,node_id,i,j,value}] → {rid:{node_id:matrix}}, sizes."""
    nodes = {n["id"]: len(n["items"]) for n in comparison_nodes(m)}
    by = {}
    for c in comps:
        by.setdefault(c["rid"], {}).setdefault(c["node_id"], {})[(c["i"], c["j"])] = c["value"]
    mats, locs = {}, {}
    for rid, nd in by.items():
        mats[rid], locs[rid] = {}, {}
        for nid, vals in nd.items():
            n = nodes.get(nid)
            if not n:
                continue
            A = C.build_matrix(n, vals)
            mats[rid][nid] = A
            locs[rid][nid] = C.priority_geometric(A)
    return mats, locs


def global_weights(m: dict, group_local: Dict[str, np.ndarray]):
    """반환: catW(list), leaf[{id,name,cat,local,global}]."""
    catW = group_local.get("GOAL")
    if catW is None:
        catW = np.ones(len(m["categories"])) / len(m["categories"])
    leaves = []
    for ci, c in enumerate(m["categories"]):
        iw = group_local.get(c["id"])
        items = [it if isinstance(it, str) else it["name"] for it in c["items"]]
        if iw is None:
            iw = np.ones(len(items)) / len(items)
        for k, nm in enumerate(items):
            leaves.append({"id": f"{c['id']}I{k+1}", "name": nm, "cat": c["name"],
                           "local": float(iw[k]), "global": float(catW[ci] * iw[k])})
    return catW, leaves


# ======================================================================
# 응답자 화면
# ======================================================================

def hbar_weight_chart(names, values, per_bar=34, label_px=460):
    """가로 막대그래프(Altair). 항목명이 잘리지 않도록 라벨 폭을 넉넉히 확보하고,
    입력된 항목 순서를 위→아래로 그대로 유지한다."""
    names = list(names)
    df = pd.DataFrame({"항목": names, "가중치": [float(v) for v in values]})
    n = len(df)
    chart = (
        alt.Chart(df)
        .mark_bar(color=BRAND, cornerRadiusEnd=3)
        .encode(
            x=alt.X("가중치:Q", axis=alt.Axis(title=None, format=".2f", grid=True)),
            y=alt.Y("항목:N", sort=names,      # 원래 제시 순서대로 위에서부터
                    axis=alt.Axis(title=None, labelLimit=label_px, labelFontSize=12)),
            tooltip=[alt.Tooltip("항목:N", title="항목"),
                     alt.Tooltip("가중치:Q", title="가중치", format=".4f")],
        )
        .properties(height=max(120, per_bar * n + 20), width="container")
        .configure_view(strokeWidth=0)
    )
    return chart


def sess_key(sid, node_id, i, j):
    return f"a::{sid}::{node_id}::{i}::{j}"


# Streamlit은 현재 화면에 없는 위젯의 상태를 지우므로, 응답값을 위젯과 별개의
# 영속 store에 콜백으로 즉시 복사해 둔다. 제출·집계는 이 store만 읽는다.
def ans_store(sid) -> dict:
    return st.session_state.setdefault(f"ANS::{sid}", {})   # {node_id: {(i,j): ratio}}


def tf_store(sid) -> dict:
    return st.session_state.setdefault(f"TF::{sid}", {})     # {item_id: '장기|중기|단기'}


def _save_pair(sid, node_id, i, j, wk):
    ans_store(sid).setdefault(node_id, {})[(i, j)] = slider_to_ratio(st.session_state[wk])


def _save_tf(sid, item_id, wk):
    v = st.session_state.get(wk)
    if v:
        tf_store(sid)[item_id] = v


def node_matrix_from_session(sid, node):
    n = len(node["items"])
    store = ans_store(sid).get(node["id"], {})
    vals = {(i, j): store.get((i, j), 1.0) for (i, j) in C.pair_indices(n)}
    return C.build_matrix(n, vals)


def render_respondent():
    survey = current_survey()
    if not survey:
        st.error("현재 진행 중인 조사가 없습니다. 관리자에게 문의해 주세요.")
        return
    if survey["status"] != "open":
        st.warning("이 조사는 현재 응답을 받지 않습니다.")
        return

    sid = survey["sid"]
    m = survey["structure"]
    nodes = comparison_nodes(m)
    flow = flow_steps(m)
    total = len(flow) + 1                      # +1 = 안내(step 0)

    st.title(m["title"])

    if st.session_state.get(f"done::{sid}"):
        st.success("응답이 정상적으로 저장되었습니다. 참여해 주셔서 감사합니다. 🙏")
        st.balloons()
        return

    step = st.session_state.setdefault(f"step::{sid}", 0)
    st.progress(step / total, text=f"진행 {step} / {total}")

    if step == 0:
        n_tf = len(m["categories"]) if m.get("timeframe_question") else 0
        order_note = ("- 진행 순서: 먼저 **분야 간 중요도**를 평가한 뒤, **분야마다 세부 과제 중요도 → 그 분야의 시급성** 순으로 이어집니다.\n"
                      if n_tf else "")
        with st.container(border=True):
            st.subheader("응답 안내")
            st.markdown(f"""
- 두 항목을 한 쌍씩 비교해 **어느 쪽이 얼마나 더 중요한지** 선택합니다. 가운데(동등)에서 좌·우로 갈수록 더 중요하며 최대 9배까지 표시됩니다.
- 총 **{total_pairs(m)}문항** / **{len(nodes)}개 비교 묶음**{' + 분야별 시급성 문항' if n_tf else ''} 입니다.
{order_note}- 각 묶음이 끝나면 응답의 **일관성(CR)** 을 자동 점검하고, 필요 시 재검토 지점을 안내합니다.
            """)
            st.text_input("성함 또는 식별번호", key=f"name::{sid}")
            st.text_input("소속 (선택)", key=f"org::{sid}")
        if st.button("응답 시작", type="primary"):
            st.session_state[f"step::{sid}"] = 1
            st.rerun()
        return

    if step <= len(flow):
        kind, ref = flow[step - 1]
        is_last = (step == len(flow))
        if kind == "pair":
            render_pair_step(sid, m, ref, is_last)
        elif kind == "tfg":
            render_timeframe_goal(sid, m, is_last)
        else:
            render_timeframe(sid, m, ref, is_last)
        return

    render_submit(sid, survey, nodes)


def render_pair_step(sid, m, node, is_last=False):
    n = len(node["items"])
    tag = "분야 중요도" if node["id"] == "GOAL" else "과제 중요도"
    st.caption(tag)
    st.subheader(node["title"])
    st.caption(node["context"] + " 아래에서 각 쌍을 비교하면 상단 요약이 실시간으로 갱신됩니다.")

    # ---- 상단 요약: 현재 응답 기준 CR + 가중치 막대 (응답 내내 고정 노출) ----
    A = node_matrix_from_session(sid, node)
    cons = C.consistency(A, METHOD)
    w = C.priority_geometric(A)

    with st.container(border=True):
        if n >= 3:
            badge = f'<span class="ahp-ok">CR {cons.cr:.3f} ≤ {CR_LIMIT:.2f} · 일관성 양호</span>' if cons.cr <= CR_LIMIT \
                else f'<span class="ahp-bad">CR {cons.cr:.3f} &gt; {CR_LIMIT:.2f} · 일관성 미달</span>'
            st.markdown(badge + f' <span class="hint">λmax {cons.lambda_max:.3f} · CI {cons.ci:.3f} · RI {cons.ri:.2f}</span>',
                        unsafe_allow_html=True)
        else:
            st.caption("항목이 2개이므로 일관성 검토는 해당되지 않습니다.")

        st.caption("현재까지의 항목 가중치")
        st.altair_chart(hbar_weight_chart(node["items"], [float(x) for x in w]))

    # ---- 쌍대비교 슬라이더 ----
    store = ans_store(sid).get(node["id"], {})
    with st.container(border=True):
        for (i, j) in C.pair_indices(n):
            k = sess_key(sid, node["id"], i, j)
            if k not in st.session_state:      # 화면 재진입 시 store에서 복원
                st.session_state[k] = ratio_to_slider(store.get((i, j), 1.0))
            c1, c2, c3 = st.columns([3, 6, 3])
            c1.markdown(f'<div class="pair-a">{node["items"][i]}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="pair-b">{node["items"][j]}</div>', unsafe_allow_html=True)
            with c2:
                st.select_slider("x", options=SLIDER_OPTS, key=k, format_func=_disp_label,
                                 label_visibility="collapsed",
                                 on_change=_save_pair, args=(sid, node["id"], i, j, k))
                v = st.session_state[k]
                mag = abs(v) + 1
                if v == 0:
                    msg = "두 항목이 **동등하게** 중요"
                elif v < 0:   # 왼쪽으로 → 왼쪽 항목(i) 우세
                    msg = f"**{node['items'][i]}**가 **{mag}배** 더 중요 ({VERBAL[mag]})"
                else:         # 오른쪽으로 → 오른쪽 항목(j) 우세
                    msg = f"**{node['items'][j]}**가 **{mag}배** 더 중요 ({VERBAL[mag]})"
                st.markdown(f'<div class="hint" style="text-align:center">{msg}</div>', unsafe_allow_html=True)
            st.divider()

    # ---- 일관성 개선 제안: 슬라이더 아래에 배치(응답 중 혼동 방지) ----
    if n >= 3 and cons.cr > CR_LIMIT:
        with st.container(border=True):
            st.markdown("**일관성 개선 제안** — 아래 판단이 응답 전체와 가장 어긋납니다. 실제 생각과 다르지 않다면 그대로 두셔도 됩니다.")
            for idx, s in enumerate(C.suggest_revisions(A, top_k=3, method=METHOD)):
                col1, col2 = st.columns([5, 2])
                col1.markdown(
                    f'<div class="suggest-box"><b>{node["items"][s.i]}</b> ↔ <b>{node["items"][s.j]}</b> · '
                    f'현재 {_fmt(s.current)}, {s.deviation:.1f}배 어긋남<br>'
                    f'권장 <b>{_fmt(s.suggested)}</b> → CR {s.cr_before:.3f} → {s.cr_after:.3f}</div>',
                    unsafe_allow_html=True)
                col2.button("권장값 적용", key=f"fix::{sid}::{node['id']}::{idx}",
                            on_click=_apply_fix, args=(sid, node["id"], s.i, s.j, s.suggested))

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("이전", disabled=(st.session_state[f'step::{sid}'] <= 1), width='stretch'):
        st.session_state[f"step::{sid}"] -= 1
        st.rerun()
    if b2.button("검토" if is_last else "다음", type="primary", width='stretch'):
        st.session_state[f"step::{sid}"] += 1
        st.rerun()


def _fmt(a):
    if abs(a - 1) < 1e-9:
        return "동등(1:1)"
    return f"A가 {a:.3g}배" if a > 1 else f"B가 {1/a:.3g}배"


def _apply_fix(sid, node_id, i, j, suggested):
    ans_store(sid).setdefault(node_id, {})[(i, j)] = suggested
    st.session_state[sess_key(sid, node_id, i, j)] = ratio_to_slider(suggested)


def render_timeframe_goal(sid, m, is_last=False):
    st.caption("시급성")
    st.subheader(f"{len(m['categories'])}개 분야의 시급성")
    st.caption("각 분야가 장기·중기·단기 중 어디에 해당하는지 선택해 주세요. (장기: 5년 이상 / 중기: 2~5년 / 단기: 2년 미만)")
    tfs = tf_store(sid)
    labels = ["장기", "중기", "단기"]
    with st.container(border=True):
        for ci, c in enumerate(m["categories"]):
            cid = c["id"]
            wk = f"tf::{sid}::GOAL::{ci}"
            if wk not in st.session_state:      # 재진입 시 store에서 복원
                prev = tfs.get(cid)
                st.session_state[wk] = prev if prev in labels else None
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**{c['name']}**")
            with c2:
                st.radio("tf", labels, key=wk, horizontal=True, label_visibility="collapsed",
                         index=None, on_change=_save_tf, args=(sid, cid, wk))
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("이전", width='stretch'):
        st.session_state[f"step::{sid}"] -= 1
        st.rerun()
    if b2.button("검토" if is_last else "다음", type="primary", width='stretch'):
        st.session_state[f"step::{sid}"] += 1
        st.rerun()


def render_timeframe(sid, m, tf_idx, is_last=False):
    cat = m["categories"][tf_idx]
    items = [it if isinstance(it, str) else it["name"] for it in cat["items"]]
    st.caption("시급성")
    st.subheader(f"「{cat['name']}」 세부 과제의 시급성")
    st.caption("각 과제가 장기·중기·단기 중 어디에 해당하는지 선택해 주세요. (장기: 5년 이상 / 중기: 2~5년 / 단기: 2년 미만)")
    tfs = tf_store(sid)
    labels = ["장기", "중기", "단기"]
    with st.container(border=True):
        for k, nm in enumerate(items):
            iid = f"{cat['id']}I{k+1}"
            wk = f"tf::{sid}::{cat['id']}::{k}"
            if wk not in st.session_state:      # 재진입 시 store에서 복원
                prev = tfs.get(iid)
                st.session_state[wk] = prev if prev in labels else None
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**{nm}**")
            with c2:
                st.radio("tf", labels, key=wk, horizontal=True, label_visibility="collapsed",
                         index=None, on_change=_save_tf, args=(sid, iid, wk))
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("이전", width='stretch'):
        st.session_state[f"step::{sid}"] -= 1
        st.rerun()
    if b2.button("검토" if is_last else "다음", type="primary", width='stretch'):
        st.session_state[f"step::{sid}"] += 1
        st.rerun()


def render_submit(sid, survey, nodes):
    m = survey["structure"]
    # 미리보기
    localW, worst = {}, 0.0
    for nd in nodes:
        A = node_matrix_from_session(sid, nd)
        localW[nd["id"]] = C.priority_geometric(A)
        worst = max(worst, C.consistency(A, METHOD).cr)
    _, leaves = global_weights(m, localW)
    leaves.sort(key=lambda x: -x["global"])

    st.subheader("응답 검토")
    st.caption(f"최대 CR {worst:.3f} {'(기준 충족)' if worst <= CR_LIMIT else '(일부 묶음 기준 초과 — 이전 단계에서 재검토 가능)'}")
    with st.container(border=True):
        st.markdown("**나의 세부과제 우선순위 (전역 가중치)**")
        df = pd.DataFrame([{"순위": k+1, "세부 과제": x["name"], "분야": x["cat"],
                            "전역 가중치": round(x["global"], 4)} for k, x in enumerate(leaves)])
        st.dataframe(df, hide_index=True, width='stretch')

    c1, c2 = st.columns([1, 1])
    if c1.button("이전", width='stretch'):
        st.session_state[f"step::{sid}"] -= 1
        st.rerun()
    if c2.button("제출", type="primary", width='stretch'):
        _do_submit(sid, survey, nodes)
        st.session_state[f"done::{sid}"] = True
        st.rerun()


def _do_submit(sid, survey, nodes):
    m = survey["structure"]
    store = ans_store(sid)
    comps, crs = [], []
    for nd in nodes:
        n = len(nd["items"])
        nd_vals = store.get(nd["id"], {})
        for (i, j) in C.pair_indices(n):
            comps.append((nd["id"], i, j, float(nd_vals.get((i, j), 1.0))))
        crs.append((nd["id"], n, C.consistency(node_matrix_from_session(sid, nd), METHOD).cr))
    tf = dict(tf_store(sid)) if m.get("timeframe_question") else {}
    meta = {"org": st.session_state.get(f"org::{sid}", ""), "tf": tf}
    DB.save_response(sid, st.session_state.get(f"name::{sid}", ""), meta, comps, crs)


# ======================================================================
# 관리자
# ======================================================================

def current_survey():
    surveys = DB.list_surveys()
    for s in surveys:
        if s["status"] == "open":
            return DB.get_survey(s["sid"])
    if surveys:
        return DB.get_survey(surveys[0]["sid"])
    return None


def ensure_default_survey():
    if not DB.list_surveys():
        DB.create_survey(DEFAULT_MODEL["title"], "", DEFAULT_MODEL,
                         {"method": METHOD, "cr_limit": CR_LIMIT})


def render_admin():
    st.sidebar.title("⚖️ 재난관리 AHP")
    st.sidebar.caption(f"저장소: {DB.backend_name()}")

    pw_needed = DB._secret("ADMIN_PASSWORD")
    if pw_needed and not st.session_state.get("admin_ok"):
        with st.sidebar:
            pw = st.text_input("관리자 비밀번호", type="password")
            if st.button("로그인"):
                if pw == pw_needed:
                    st.session_state["admin_ok"] = True
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        st.title("재난관리 AHP 조사")
        st.info("좌측에서 관리자 비밀번호를 입력해 주세요. 응답자는 `?respond=1` 링크로 접속합니다.")
        return

    ensure_default_survey()
    menu = st.sidebar.radio("메뉴", ["조사 관리", "결과 집계", "응답 미리보기"])
    if menu == "조사 관리":
        render_manage()
    elif menu == "결과 집계":
        render_results()
    else:
        render_admin_preview()


def render_manage():
    st.title("조사 관리")
    surveys = DB.list_surveys()
    df = pd.DataFrame(surveys).rename(columns={"sid": "조사ID", "title": "제목", "status": "상태",
                                               "created_at": "생성일시", "n": "응답 수"})
    st.dataframe(df, hide_index=True, width='stretch')

    sid = st.selectbox("조사 선택", [s["sid"] for s in surveys],
                       format_func=lambda x: next(s["title"] for s in surveys if s["sid"] == x))
    sv = DB.get_survey(sid)

    base = DB._secret("APP_BASE_URL") or ""
    url = f"{base.rstrip('/')}/?respond=1" if base else "?respond=1"
    st.markdown("**응답자 배포 링크**")
    st.code(url, language=None)
    if not base:
        st.caption("secrets.toml 에 APP_BASE_URL(예: https://your-app.streamlit.app)을 넣으면 전체 주소로 표시됩니다.")

    c1, c2, c3 = st.columns(3)
    if c1.button("응답 마감" if sv["status"] == "open" else "응답 재개", width='stretch'):
        DB.update_survey(sid, status="closed" if sv["status"] == "open" else "open")
        st.rerun()
    comp = pd.DataFrame(DB.fetch_comparisons(sid))
    c2.download_button("원자료 CSV", comp.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ahp_raw_{sid}.csv", width='stretch',
                       disabled=comp.empty)
    with c3.popover("구조 편집", width='stretch'):
        st.caption("들여쓰기(공백 2칸)로 하위 과제를 표현합니다. 저장 시 새 응답부터 적용됩니다.")
        txt = st.text_area("구조", value=_model_to_text(sv["structure"]), height=320)
        tfq = st.checkbox("분야별 시급성(장기/중기/단기) 문항 포함", value=sv["structure"].get("timeframe_question", True))
        if st.button("구조 저장", type="primary"):
            newm = _text_to_model(txt, tfq)
            if len(newm["categories"]) >= 2:
                DB.update_survey(sid, title=newm["title"], structure=newm)
                st.success("구조가 저장되었습니다.")
                st.rerun()
            else:
                st.error("분야를 2개 이상 입력하세요.")

    st.markdown("##### 응답자 목록")
    resp = DB.fetch_responses(sid)
    if resp:
        crs = pd.DataFrame(DB.fetch_crs(sid))
        worst = crs.groupby("rid")["cr"].max().to_dict() if not crs.empty else {}
        rd = pd.DataFrame([{"응답ID": r["rid"], "응답자": r["respondent"], "소속": r["meta"].get("org", ""),
                            "제출시각": r["submitted_at"], "최대 CR": round(worst.get(r["rid"], 0), 4)} for r in resp])
        st.dataframe(rd, hide_index=True, width='stretch')
        rid = st.selectbox("삭제할 응답", [""] + [r["rid"] for r in resp])
        if rid and st.button("응답 삭제"):
            DB.delete_response(rid)
            st.rerun()
    else:
        st.caption("아직 응답이 없습니다.")


def _model_to_text(m):
    t = f"제목: {m['title']}\n목표: {m['goal']}\n"
    for c in m["categories"]:
        t += f"- {c['name']}\n"
        for it in c["items"]:
            t += f"  - {it if isinstance(it, str) else it['name']}\n"
    return t


def _text_to_model(text, tfq):
    title, goal, cats, cur = "AHP 조사", "", [], None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        low = raw.strip().replace(" ", "")
        if low.startswith("제목:"):
            title = raw.split(":", 1)[1].strip(); continue
        if low.startswith("목표:"):
            goal = raw.split(":", 1)[1].strip(); continue
        indent = len(raw) - len(raw.lstrip())
        label = raw.strip().lstrip("-*•").strip()
        if not label:
            continue
        if indent < 2:
            cur = {"id": f"C{len(cats)+1}", "name": label, "items": []}
            cats.append(cur)
        elif cur:
            cur["items"].append(label)
    return {"title": title, "goal": goal or title, "categories": cats, "timeframe_question": tfq}


# ---- 결과 집계 ----

def build_wide_df(m, comps, resp):
    """응답자 1인 = 1행. 열 = 쌍대비교(GOAL_0_1 …) + 시급성(TF_분야 / TF_과제) + 응답자정보."""
    nodes = comparison_nodes(m)
    # 열 순서 정의
    pair_cols = []
    for nd in nodes:
        for (i, j) in C.pair_indices(len(nd["items"])):
            pair_cols.append(f"{nd['id']}_{i}_{j}")
    tf_cat_cols = [f"TF_{c['id']}" for c in m["categories"]] if m.get("timeframe_question") else []
    tf_item_cols = ([f"TF_{c['id']}I{k+1}" for c in m["categories"] for k in range(len(c["items"]))]
                    if m.get("timeframe_question") else [])

    # rid → 값
    pv = {}
    for c in comps:
        pv.setdefault(c["rid"], {})[f"{c['node_id']}_{c['i']}_{c['j']}"] = c["value"]
    meta_by = {r["rid"]: r for r in resp}

    rows = []
    for r in resp:
        rid = r["rid"]
        row = {"rid": rid, "응답자": r.get("respondent", ""),
               "소속": (r.get("meta") or {}).get("org", ""), "제출시각": r.get("submitted_at", "")}
        vals = pv.get(rid, {})
        for col in pair_cols:
            row[col] = vals.get(col, "")
        tf = (r.get("meta") or {}).get("tf") or {}
        for c in m["categories"]:
            if m.get("timeframe_question"):
                row[f"TF_{c['id']}"] = tf.get(c["id"], "")
        for c in m["categories"]:
            for k in range(len(c["items"])):
                if m.get("timeframe_question"):
                    row[f"TF_{c['id']}I{k+1}"] = tf.get(f"{c['id']}I{k+1}", "")
        rows.append(row)

    cols = ["rid", "응답자", "소속", "제출시각"] + pair_cols + tf_cat_cols + tf_item_cols
    return pd.DataFrame(rows, columns=cols)


def render_results():
    st.title("결과 집계")
    surveys = DB.list_surveys()
    sid = st.selectbox("조사 선택", [s["sid"] for s in surveys],
                       format_func=lambda x: next(f"{s['title']} (응답 {s['n']}명)" for s in surveys if s["sid"] == x))
    survey = DB.get_survey(sid)
    m = survey["structure"]
    comps = DB.fetch_comparisons(sid)
    resp = DB.fetch_responses(sid)
    if not comps:
        st.info("아직 집계할 응답이 없습니다.")
        return

    nodes = comparison_nodes(m)
    node_by = {n["id"]: n for n in nodes}
    mats, locs = response_matrices(m, comps)

    c1, c2 = st.columns(2)
    agg = c1.selectbox("집단 집계 방식", ["AIJ", "AIP"],
                       format_func=lambda x: "AIJ · 판단행렬 기하평균 (합의형)" if x == "AIJ" else "AIP · 개인 가중치 기하평균")
    excl = c2.checkbox("CR 10% 초과 응답 제외", value=False)

    # CR 표 & 필터
    cr_rows = []
    for rid, nd in mats.items():
        for nid, A in nd.items():
            cc = C.consistency(A, METHOD)
            cr_rows.append({"rid": rid, "node_id": nid, "묶음": node_by[nid]["title"],
                            "n": cc.n, "CR": cc.cr, "ok": cc.cr <= CR_LIMIT})
    cr_df = pd.DataFrame(cr_rows)
    bad = set(cr_df.loc[~cr_df["ok"], "rid"]) if not cr_df.empty else set()
    use = [r for r in locs if not (excl and r in bad)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 응답", f"{len(resp)}명")
    k2.metric("집계 대상", f"{len(use)}명")
    k3.metric("CR 충족률", f"{100*(1-len(bad)/max(len(locs),1)):.0f}%")
    k4.metric("평균 CR", f"{cr_df['CR'].mean():.3f}" if not cr_df.empty else "-")

    if not use:
        st.error("필터 조건을 만족하는 응답이 없습니다.")
        return

    # 집단 로컬 가중치
    group_local, group_cr = {}, {}
    for nid in node_by:
        vs = [locs[r][nid] for r in use if nid in locs[r]]
        if not vs:
            continue
        if agg == "AIJ":
            Ms = [mats[r][nid] for r in use if nid in mats[r]]
            Ag = C.aggregate_matrices(Ms)
            group_local[nid] = C.priority_geometric(Ag)
            group_cr[nid] = C.consistency(Ag, METHOD).cr
        else:
            group_local[nid] = C.aggregate_priorities(vs)
            group_cr[nid] = float(np.mean([C.consistency(mats[r][nid], METHOD).cr for r in use if nid in mats[r]]))

    catW, leaves = global_weights(m, group_local)
    leaves.sort(key=lambda x: -x["global"])

    tabs = st.tabs(["분야·과제 우선순위", "일관성 진단", "응답자별"] +
                   (["시급성 분포"] if m.get("timeframe_question") else []) + ["내려받기"])

    with tabs[0]:
        st.markdown("##### 분야(중범주) 가중치")
        cat_df = pd.DataFrame([{"분야": c["name"], "가중치": float(catW[i])}
                               for i, c in enumerate(m["categories"])]).sort_values("가중치", ascending=False)
        cat_df.insert(0, "순위", range(1, len(cat_df) + 1))
        st.dataframe(cat_df.style.format({"가중치": "{:.4f}"}), hide_index=True, width='stretch')
        st.altair_chart(hbar_weight_chart([c["name"] for c in m["categories"]],
                                          [float(catW[i]) for i in range(len(m["categories"]))]))

        st.markdown("##### 세부 과제 종합 우선순위 (전역 가중치, 합=1)")
        leaf_df = pd.DataFrame([{"순위": k+1, "세부 과제": x["name"], "분야": x["cat"],
                                 "지역 가중치": x["local"], "전역 가중치": x["global"]}
                                for k, x in enumerate(leaves)])
        st.dataframe(leaf_df.style.format({"지역 가중치": "{:.4f}", "전역 가중치": "{:.4f}"}),
                     hide_index=True, width='stretch')

    with tabs[1]:
        summ = (cr_df.groupby(["node_id", "묶음", "n"])
                .agg(평균CR=("CR", "mean"), 최대CR=("CR", "max"), 충족률=("ok", "mean"), 응답수=("CR", "size"))
                .reset_index())
        summ["충족률"] = (summ["충족률"] * 100).round(0)
        summ["집단행렬CR"] = summ["node_id"].map(group_cr)
        st.dataframe(summ[["묶음", "n", "응답수", "평균CR", "최대CR", "충족률", "집단행렬CR"]]
                     .style.format({"평균CR": "{:.3f}", "최대CR": "{:.3f}", "충족률": "{:.0f}%", "집단행렬CR": "{:.3f}"}),
                     hide_index=True, width='stretch')
        bad_df = cr_df[~cr_df["ok"]]
        if bad_df.empty:
            st.success("모든 응답이 CR 10% 기준을 충족합니다.")
        else:
            name_by = {r["rid"]: (r["respondent"] or r["rid"]) for r in resp}
            bad_df = bad_df.assign(응답자=bad_df["rid"].map(name_by))
            st.markdown("##### 기준 초과 응답")
            st.dataframe(bad_df[["응답자", "묶음", "n", "CR"]].sort_values("CR", ascending=False)
                         .style.format({"CR": "{:.3f}"}), hide_index=True, width='stretch')

    with tabs[2]:
        name_by = {r["rid"]: (r["respondent"] or r["rid"]) for r in resp}
        target = st.selectbox("비교 묶음", list(node_by.keys()), format_func=lambda x: node_by[x]["title"])
        items = node_by[target]["items"]
        rows = []
        for rid in use:
            v = locs[rid].get(target)
            if v is None:
                continue
            row = {"응답자": name_by.get(rid, rid), "CR": C.consistency(mats[rid][target], METHOD).cr}
            row.update({it: float(v[k]) for k, it in enumerate(items)})
            rows.append(row)
        idf = pd.DataFrame(rows)
        st.dataframe(idf.style.format({c: "{:.4f}" for c in ["CR"] + list(items)}),
                     hide_index=True, width='stretch')
        if len(idf) > 1:
            st.caption("항목별 응답자 간 표준편차 — 클수록 의견이 갈립니다.")
            st.dataframe(idf[list(items)].std().round(4).to_frame("표준편차").T, width='stretch')

    tf_tab_idx = 3
    if m.get("timeframe_question"):
        with tabs[3]:
            st.markdown("##### 분야 시급성")
            crows = []
            for c in m["categories"]:
                cnt = {"장기": 0, "중기": 0, "단기": 0}
                for r in resp:
                    v = (r["meta"].get("tf") or {}).get(c["id"])
                    if v in cnt:
                        cnt[v] += 1
                tot = sum(cnt.values())
                crows.append({"분야": c["name"], "장기": cnt["장기"], "중기": cnt["중기"],
                              "단기": cnt["단기"], "응답": tot})
            st.dataframe(pd.DataFrame(crows), hide_index=True, width='stretch')

            st.markdown("##### 세부 과제 시급성")
            names = item_names(m)
            rows = []
            for c in m["categories"]:
                for k, it in enumerate(c["items"]):
                    iid = f"{c['id']}I{k+1}"
                    cnt = {"장기": 0, "중기": 0, "단기": 0}
                    for r in resp:
                        v = (r["meta"].get("tf") or {}).get(iid)
                        if v in cnt:
                            cnt[v] += 1
                    tot = sum(cnt.values())
                    rows.append({"세부 과제": names[iid], "분야": c["name"],
                                 "장기": cnt["장기"], "중기": cnt["중기"], "단기": cnt["단기"], "응답": tot})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        tf_tab_idx = 4

    with tabs[tf_tab_idx]:
        wide_df = build_wide_df(m, comps, resp)

        st.markdown("##### 응답값 (wide) — 응답자 1인 = 1행")
        st.caption("쌍대비교(GOAL_0_1 …)와 시급성(TF_C1=분야, TF_C1I1=과제)을 한 행에 모은 형식입니다.")
        st.dataframe(wide_df, hide_index=True, width='stretch')
        st.download_button("응답값 (wide) CSV 내려받기",
                           wide_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"ahp_wide_{sid}.csv", mime="text/csv",
                           disabled=wide_df.empty)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            wide_df.to_excel(xw, sheet_name="응답값_wide", index=False)
            leaf_df.to_excel(xw, sheet_name="전역가중치", index=False)
            cat_df.to_excel(xw, sheet_name="분야가중치", index=False)
            cr_df.to_excel(xw, sheet_name="일관성", index=False)
            pd.DataFrame(comps).to_excel(xw, sheet_name="원자료_long", index=False)
        st.download_button("결과 Excel 내려받기 (wide 포함)", buf.getvalue(),
                           file_name=f"ahp_result_{sid}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")


def render_admin_preview():
    st.title("응답 화면 미리보기")
    st.caption("관리자가 실제 응답 흐름을 그대로 확인합니다. 제출하면 DB에 실제로 저장되니 테스트 후 삭제하세요.")
    render_respondent()


# ======================================================================
# 라우팅
# ======================================================================

def main():
    ensure_default_survey()   # 콜드 스타트에도 응답 링크가 바로 동작하도록 보장
    if st.query_params.get("respond"):
        render_respondent()
    else:
        render_admin()


if __name__ == "__main__":
    main()