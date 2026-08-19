import os
os.environ["AHP_DB_PATH"] = "/tmp/ahp_app_test.db"
for f in ["/tmp/ahp_app_test.db","/tmp/ahp_app_test.db-wal","/tmp/ahp_app_test.db-shm"]:
    if os.path.exists(f): os.remove(f)

import numpy as np
import ahp_core as C, ahp_db as DB
import app as APP

DB.init_db()
m = APP.DEFAULT_MODEL
nodes = APP.comparison_nodes(m)
print("비교 묶음:", [(n["id"], len(n["items"])) for n in nodes])
print("총 문항:", APP.total_pairs(m))
assert APP.total_pairs(m) == 52, "문항수 불일치"

# 척도 변환 왕복
for v in [-8,-3,0,2,8]:
    assert APP.ratio_to_slider(APP.slider_to_ratio(v)) == v, v
print("척도 변환 OK")

# 조사 생성 + 가상 응답 15명
sid = DB.create_survey(m["title"], "", m, {"method":"geometric","cr_limit":0.10})
rng = np.random.default_rng(11)
truth = {n["id"]: rng.dirichlet(np.ones(len(n["items"]))*3) for n in nodes}
for p in range(15):
    comps, crs = [], []
    for nd in nodes:
        n=len(nd["items"]); w=truth[nd["id"]]*np.exp(rng.normal(0,0.22,n)); vals={}
        for (i,j) in C.pair_indices(n):
            r=w[i]/w[j]*np.exp(rng.normal(0,0.3)); a=C.nearest_saaty(r); vals[(i,j)]=a
            comps.append((nd["id"],i,j,a))
        crs.append((nd["id"],n,C.consistency(C.build_matrix(n,vals),"geometric").cr))
    # 시급성
    tf={}
    for c in m["categories"]:
        for k in range(len(c["items"])):
            tf[f"{c['id']}I{k+1}"]=rng.choice(["장기","중기","단기"])
    DB.save_response(sid, f"응답자{p+1:02d}", {"org":"","tf":tf}, comps, crs)

comps = DB.fetch_comparisons(sid)
resp = DB.fetch_responses(sid)
print("\n저장:", len(comps),"비교값 /", len(resp),"명")

mats, locs = APP.response_matrices(m, comps)
crs=[C.consistency(A,"geometric").cr for mm in mats.values() for A in mm.values()]
print("CR 평균 %.3f 최대 %.3f 충족 %.0f%%" % (np.mean(crs),np.max(crs),100*np.mean([c<=0.10 for c in crs])))

# AIJ 집단
node_by={n["id"]:n for n in nodes}
group_local={}
for nid in node_by:
    Ms=[mats[r][nid] for r in mats if nid in mats[r]]
    group_local[nid]=C.priority_geometric(C.aggregate_matrices(Ms))
catW, leaves = APP.global_weights(m, group_local)
print("\n분야 가중치 합=%.6f" % catW.sum())
leaves.sort(key=lambda x:-x["global"])
print("전역가중치 합=%.6f" % sum(x["global"] for x in leaves))
print("상위 3:", [(round(x["global"],4), x["name"][:14]) for x in leaves[:3]])

# 시급성 집계 확인
names=APP.item_names(m)
tf_total=0
for r in resp:
    tf_total += len(r["meta"].get("tf") or {})
print("\n시급성 응답 총:", tf_total, "(기대 23*15=%d)"%(23*15))
assert tf_total==23*15

print("\n앱 로직 E2E OK")
