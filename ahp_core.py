"""
ahp_core.py — AHP 계산 엔진
- 쌍대비교 행렬 → 우선순위 벡터 (고유벡터법 / 기하평균법)
- 일관성 지수(CI) · 일관성 비율(CR)
- 비일관 판단 탐지 및 개선안 제안 (Saaty 방식)
- 집단 집계 (AIJ / AIP)
- 계층 종합 (global weight)

의존성: numpy 만 사용.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------

# Saaty 무작위 지수(Random Index)
RANDOM_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}

# Saaty 권장 CR 허용 한계 (n=3, n=4 는 더 엄격)
CR_LIMIT_BY_N: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.05, 4: 0.08}
CR_LIMIT_DEFAULT = 0.10

# 9점 척도 (역수 포함)
SAATY_SCALE: List[float] = [1 / v for v in range(9, 1, -1)] + [float(v) for v in range(1, 10)]


def cr_limit(n: int) -> float:
    """비교 항목 수 n 에 대한 CR 허용 한계."""
    return CR_LIMIT_BY_N.get(n, CR_LIMIT_DEFAULT)


# ----------------------------------------------------------------------
# 척도 변환 (응답 UI ↔ 행렬값)
# ----------------------------------------------------------------------

def slider_to_ratio(v: int) -> float:
    """
    응답 슬라이더 정수값 → 쌍대비교 행렬값 a_ij.
    v > 0 : 왼쪽(i)이 (v+1)배 중요,  v = 0 : 동등,  v < 0 : 오른쪽(j)이 (|v|+1)배 중요
    v 의 범위는 -8 ~ +8 (즉 1~9배).
    """
    v = int(v)
    if v == 0:
        return 1.0
    if v > 0:
        return float(v + 1)
    return 1.0 / float(-v + 1)


def ratio_to_slider(a: float) -> int:
    """행렬값 → 슬라이더 정수값 (역변환, 근사)."""
    if a >= 1:
        return int(round(a)) - 1
    return -(int(round(1.0 / a)) - 1)


def nearest_saaty(target: float) -> float:
    """임의의 양수를 로그 거리 기준으로 가장 가까운 Saaty 척도값에 맞춤."""
    target = max(target, 1e-9)
    return min(SAATY_SCALE, key=lambda s: abs(math.log(s) - math.log(target)))


def format_ratio(a: float) -> str:
    """행렬값을 '왼쪽이 3배' 같은 사람이 읽는 문자열로."""
    if abs(a - 1.0) < 1e-9:
        return "동등 (1:1)"
    if a > 1:
        return f"왼쪽이 {a:.3g}배 중요"
    return f"오른쪽이 {1/a:.3g}배 중요"


# ----------------------------------------------------------------------
# 행렬 구성
# ----------------------------------------------------------------------

def pair_indices(n: int) -> List[Tuple[int, int]]:
    """상삼각 쌍 목록. 문항 수 = n(n-1)/2."""
    return list(itertools.combinations(range(n), 2))


def build_matrix(n: int, values: Dict[Tuple[int, int], float]) -> np.ndarray:
    """
    상삼각 응답값 dict {(i, j): a_ij} → 완전한 역수행렬(reciprocal matrix).
    누락된 쌍은 1.0 으로 채움.
    """
    A = np.ones((n, n), dtype=float)
    for (i, j) in pair_indices(n):
        a = float(values.get((i, j), 1.0))
        a = max(a, 1e-9)
        A[i, j] = a
        A[j, i] = 1.0 / a
    return A


# ----------------------------------------------------------------------
# 우선순위 벡터
# ----------------------------------------------------------------------

def priority_eigen(A: np.ndarray, tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """주고유벡터법 (거듭제곱 반복). 복소수 없이 안정적으로 수렴."""
    n = A.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        w_new = A @ w
        s = w_new.sum()
        if s <= 0:
            break
        w_new = w_new / s
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w / w.sum()


def priority_geometric(A: np.ndarray) -> np.ndarray:
    """기하평균법(행 기하평균 정규화). 이상치에 덜 민감."""
    g = np.exp(np.mean(np.log(np.clip(A, 1e-12, None)), axis=1))
    return g / g.sum()


def priority_vector(A: np.ndarray, method: str = "eigen") -> np.ndarray:
    return priority_geometric(A) if method == "geometric" else priority_eigen(A)


# ----------------------------------------------------------------------
# 일관성
# ----------------------------------------------------------------------

@dataclass
class Consistency:
    n: int
    lambda_max: float
    ci: float
    ri: float
    cr: float
    limit: float

    @property
    def ok(self) -> bool:
        return self.cr <= self.limit + 1e-12

    def as_dict(self) -> dict:
        return {
            "n": self.n, "lambda_max": self.lambda_max, "ci": self.ci,
            "ri": self.ri, "cr": self.cr, "limit": self.limit, "ok": self.ok,
        }


def consistency(A: np.ndarray, method: str = "eigen") -> Consistency:
    n = A.shape[0]
    if n < 3:
        # 2개 이하는 항상 완전 일관 (CR 정의 불가)
        return Consistency(n=n, lambda_max=float(n), ci=0.0, ri=0.0, cr=0.0, limit=cr_limit(n))
    w = priority_vector(A, method)
    Aw = A @ w
    lam = float(np.mean(Aw / np.clip(w, 1e-12, None)))
    ci = (lam - n) / (n - 1)
    ri = RANDOM_INDEX.get(n, 1.60)
    cr = ci / ri if ri > 0 else 0.0
    return Consistency(n=n, lambda_max=lam, ci=ci, ri=ri, cr=max(cr, 0.0), limit=cr_limit(n))


# ----------------------------------------------------------------------
# 개선 제안
# ----------------------------------------------------------------------

@dataclass
class Suggestion:
    i: int
    j: int
    current: float          # 현재 응답값 a_ij
    ideal: float            # 완전 일관 시 값 w_i / w_j
    suggested: float        # 9점 척도로 반올림한 권장값
    deviation: float        # 왜곡 배수 (1에 가까울수록 일관)
    cr_after: float         # 이 한 건만 고쳤을 때 예상 CR
    cr_before: float

    @property
    def gain(self) -> float:
        return self.cr_before - self.cr_after


def _cr_if_replaced(A: np.ndarray, i: int, j: int, new_val: float, method: str) -> float:
    B = A.copy()
    B[i, j] = new_val
    B[j, i] = 1.0 / new_val
    return consistency(B, method).cr


def suggest_revisions(A: np.ndarray, top_k: int = 3, method: str = "eigen") -> List[Suggestion]:
    """
    가장 비일관적인 판단을 찾아 개선안을 제시.
    기준: 왜곡비 d_ij = a_ij * w_j / w_i 가 1에서 멀수록 비일관.
    권장값은 완전일관값 w_i/w_j 를 9점 척도로 반올림한 값.
    """
    n = A.shape[0]
    if n < 3:
        return []
    w = priority_vector(A, method)
    cr0 = consistency(A, method).cr

    cands = []
    for (i, j) in pair_indices(n):
        ideal = w[i] / w[j]
        d = A[i, j] / ideal
        dev = max(d, 1.0 / d)
        sug = nearest_saaty(ideal)
        if abs(math.log(sug) - math.log(A[i, j])) < 1e-9:
            continue  # 이미 권장값과 동일
        cands.append((dev, i, j, ideal, sug))

    cands.sort(reverse=True, key=lambda t: t[0])

    out: List[Suggestion] = []
    for dev, i, j, ideal, sug in cands[: max(top_k * 3, 6)]:
        cr_after = _cr_if_replaced(A, i, j, sug, method)
        out.append(Suggestion(i=i, j=j, current=float(A[i, j]), ideal=float(ideal),
                              suggested=float(sug), deviation=float(dev),
                              cr_after=float(cr_after), cr_before=float(cr0)))
    # 실제 CR 감소폭이 큰 순으로 재정렬
    out.sort(key=lambda s: s.cr_after)
    return out[:top_k]


def auto_repair(A: np.ndarray, method: str = "eigen", max_steps: int = 10
                ) -> Tuple[np.ndarray, List[Suggestion]]:
    """
    CR 이 기준 이하가 될 때까지 가장 비일관적인 판단을 순차적으로 교체.
    ※ 응답자의 원래 의도를 변형하므로 '자동 적용'보다 '제안 목록' 용도로 쓸 것.
    """
    B = A.copy()
    steps: List[Suggestion] = []
    limit = cr_limit(B.shape[0])
    for _ in range(max_steps):
        c = consistency(B, method)
        if c.cr <= limit:
            break
        sugs = suggest_revisions(B, top_k=1, method=method)
        if not sugs:
            break
        s = sugs[0]
        B[s.i, s.j] = s.suggested
        B[s.j, s.i] = 1.0 / s.suggested
        steps.append(s)
    return B, steps


# ----------------------------------------------------------------------
# 집단 집계
# ----------------------------------------------------------------------

def aggregate_matrices(mats: Sequence[np.ndarray]) -> np.ndarray:
    """AIJ — 개별 판단행렬의 원소별 기하평균 (역수성 보존)."""
    stack = np.stack([np.clip(m, 1e-12, None) for m in mats])
    return np.exp(np.mean(np.log(stack), axis=0))


def aggregate_priorities(vecs: Sequence[np.ndarray]) -> np.ndarray:
    """AIP — 개별 가중치 벡터의 기하평균 후 정규화."""
    stack = np.stack([np.clip(v, 1e-12, None) for v in vecs])
    g = np.exp(np.mean(np.log(stack), axis=0))
    return g / g.sum()


# ----------------------------------------------------------------------
# 계층 구조 & 종합
# ----------------------------------------------------------------------

@dataclass
class Node:
    id: str
    name: str
    children: List["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def leaves(node: Node) -> List[Node]:
    if node.is_leaf:
        return [node]
    out: List[Node] = []
    for c in node.children:
        out.extend(leaves(c))
    return out


def comparison_nodes(root: Node) -> List[Node]:
    """쌍대비교가 필요한 노드(자식 2개 이상)를 위에서 아래로 나열."""
    out: List[Node] = []
    stack = [root]
    while stack:
        nd = stack.pop(0)
        if len(nd.children) >= 2:
            out.append(nd)
        stack = nd.children + stack
    return out


def global_weights(root: Node, local: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    local: {부모노드id: 자식들의 지역 가중치 벡터}
    반환: {노드id: 전역 가중치}  (루트=1.0)
    """
    gw: Dict[str, float] = {root.id: 1.0}

    def walk(nd: Node):
        if not nd.children:
            return
        vec = local.get(nd.id)
        if vec is None:
            vec = np.ones(len(nd.children)) / len(nd.children)
        for k, ch in enumerate(nd.children):
            gw[ch.id] = gw[nd.id] * float(vec[k])
            walk(ch)

    walk(root)
    return gw


def synthesize_alternatives(root: Node,
                            criteria_local: Dict[str, np.ndarray],
                            alt_local: Dict[str, np.ndarray],
                            alternatives: List[str]) -> np.ndarray:
    """
    기준 계층의 전역가중치 × 각 말단기준 하에서의 대안 지역가중치 → 대안 종합점수.
    alt_local: {말단기준id: 대안 가중치 벡터}
    """
    gw = global_weights(root, criteria_local)
    score = np.zeros(len(alternatives))
    for lf in leaves(root):
        vec = alt_local.get(lf.id)
        if vec is None:
            continue
        score += gw.get(lf.id, 0.0) * np.asarray(vec, dtype=float)
    s = score.sum()
    return score / s if s > 0 else score


# ----------------------------------------------------------------------
# 계층 구조 텍스트 파서
# ----------------------------------------------------------------------

def parse_structure(text: str) -> Tuple[Node, List[str]]:
    """
    들여쓰기 기반 구조 텍스트를 파싱.

    예)
        목표: 신규 채널 우선순위 결정
        기준:
        - 비용
          - 초기 투자
          - 운영비
        - 도달률
        - 브랜드 적합성
        대안:
        - A 채널
        - B 채널

    반환: (루트 Node, 대안 이름 리스트)
    """
    goal = "목표"
    crit_lines: List[Tuple[int, str]] = []
    alts: List[str] = []
    section = None

    for raw in text.splitlines():
        if not raw.strip():
            continue
        stripped = raw.strip()
        low = stripped.replace(" ", "")

        if low.startswith("목표:") or low.lower().startswith("goal:"):
            goal = stripped.split(":", 1)[1].strip() or goal
            section = None
            continue
        if low in ("기준:", "기준", "criteria:", "criteria"):
            section = "criteria"
            continue
        if low in ("대안:", "대안", "alternatives:", "alternatives"):
            section = "alternatives"
            continue

        indent = len(raw) - len(raw.lstrip(" \t"))
        label = stripped.lstrip("-*•").strip()
        if not label:
            continue

        if section == "alternatives":
            alts.append(label)
        else:
            section = "criteria"
            crit_lines.append((indent, label))

    root = Node(id="GOAL", name=goal)
    stack: List[Tuple[int, Node]] = [(-1, root)]
    counter = 0
    for indent, label in crit_lines:
        counter += 1
        nd = Node(id=f"C{counter}", name=label)
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        stack[-1][1].children.append(nd)
        stack.append((indent, nd))

    return root, alts


def node_map(root: Node) -> Dict[str, Node]:
    out: Dict[str, Node] = {}

    def walk(nd: Node):
        out[nd.id] = nd
        for c in nd.children:
            walk(c)

    walk(root)
    return out


def to_dict(nd: Node) -> dict:
    return {"id": nd.id, "name": nd.name, "children": [to_dict(c) for c in nd.children]}


def from_dict(d: dict) -> Node:
    return Node(id=d["id"], name=d["name"], children=[from_dict(c) for c in d.get("children", [])])
