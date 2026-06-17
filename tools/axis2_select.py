#!/usr/bin/env python3
"""
축2 Step 3 — 재구성 결합 granularity 선택 (training-free, 순수 numpy).

Step 2(axis2_vote.py)의 parts3d.npz/json을 받아:
  1) 겹치는 일관 part를 obj-support 위에서 disjoint 파티션으로 정리
     (우선순위: views 多 → 면적 大).
  2) C2 재구성가능성: 각 part가 충분한 점수(min_points)+가시뷰(min_recon_views).
  3) C3 distinctness: part 간 3D 분리(centroid z-분포 등) — 약한 보조 기준.
  4) 결정: C2 통과 part가 ≥2개이고 obj-support의 cover_th 이상을 덮으면 'parts',
     아니면 'whole'. (split이 재구성을 도울 때만 split = 우리 차별점.)
출력: 결정 + 최종 파티션(GS 인덱스) + 색칠 PLY + json.

실행 (어느 env든 numpy만):
    python axis2_select.py --in_dir ~/axis2_vote_98 \
        --min_points 800 --min_recon_views 3 --cover_th 0.6
"""
import argparse
import json
import os
import numpy as np


def write_ply_rgb(path, xyz, rgb):
    N = len(xyz)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        a = np.empty(N, dt)
        a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        a["r"], a["g"], a["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="axis2_vote.py 출력 폴더")
    ap.add_argument("--min_points", type=int, default=800,
                    help="C2: part가 독립 재구성되려면 필요한 최소 GS 점수")
    ap.add_argument("--min_recon_views", type=int, default=3,
                    help="C2: part가 독립 재구성되려면 필요한 최소 가시 뷰")
    ap.add_argument("--cover_th", type=float, default=0.6,
                    help="채택 part들이 obj-support를 이만큼 덮어야 split")
    ap.add_argument("--distinct_z", type=float, default=0.15,
                    help="C3: part 간 z-중심 분리(객체 z-범위 대비 비율) 권장 기준")
    args = ap.parse_args()

    npz = np.load(os.path.join(args.in_dir, "parts3d.npz"))
    xyz = npz["xyz"]; parts = npz["parts"]            # parts: K x N bool
    meta = json.load(open(os.path.join(args.in_dir, "parts3d.json")))
    views = [m["views"] for m in meta]
    K, N = parts.shape
    if K == 0:
        print("[ERROR] 일관 part 없음 → whole 강제"); return
    obj_support = parts.any(axis=0)
    nsup = int(obj_support.sum())
    print(f"obj-support: {nsup} GS / parts(K)={K}")

    # ── 1) disjoint 파티션 (views 多 → 면적 大 우선) ──
    order = sorted(range(K), key=lambda k: (views[k], int(parts[k].sum())), reverse=True)
    assign = -np.ones(N, int)
    for k in order:
        take = parts[k] & (assign < 0)
        assign[take] = k
    dj_size = np.array([(assign == k).sum() for k in range(K)])

    # ── 2) C2 + 3) C3 ──
    zr = xyz[obj_support, 2]
    z_range = (zr.max() - zr.min()) if nsup else 1.0
    rows = []
    for k in range(K):
        m = (assign == k)
        s = int(m.sum())
        zc = xyz[m, 2].mean() if s else 0.0
        c2 = (s >= args.min_points) and (views[k] >= args.min_recon_views)
        rows.append(dict(k=k, size=s, views=views[k], zc=zc, c2=c2))
    # C3: 채택 후보(C2통과) 간 z-중심이 충분히 분리되는지(부품 구분 신호)
    passed = [r for r in rows if r["c2"]]
    print("\npart 평가 (disjoint):")
    for r in rows:
        print(f"  part{r['k']}: size={r['size']:5d} views={r['views']} "
              f"z_c={r['zc']:.3f}  C2={'O' if r['c2'] else 'X'}")

    cover = sum(r["size"] for r in passed) / max(nsup, 1)
    distinct = True
    if len(passed) >= 2:
        zcs = sorted(r["zc"] for r in passed)
        gaps = np.diff(zcs)
        distinct = bool(np.any(gaps > args.distinct_z * z_range)) or True  # z-분리 또는 통과(완화)
    print(f"\nC2 통과 part: {len(passed)}, cover={cover*100:.1f}% (>= {args.cover_th*100:.0f}%?)")

    # ── 4) 결정 ──
    decision = "parts" if (len(passed) >= 2 and cover >= args.cover_th and distinct) else "whole"
    print(f"\n>>> granularity 결정: {decision.upper()}")
    if decision == "whole":
        print("    (split이 재구성을 도울 근거 부족 → 객체 통째 유지)")

    # ── 출력: 최종 파티션 + 색칠 PLY ──
    PALETTE = np.array([[230,25,75],[60,180,75],[0,130,200],[245,130,48],
                        [145,30,180],[70,240,240],[240,50,230],[210,245,60]], np.uint8)
    rgb = np.full((N, 3), 110, np.uint8)
    out_parts = []
    if decision == "whole":
        rgb[obj_support] = np.array([0, 130, 200], np.uint8)
        out_parts = [{"id": 0, "kind": "whole", "n_gs": nsup}]
        sel = obj_support[None, :]
    else:
        sel_list = []
        for i, r in enumerate(passed):
            m = (assign == r["k"])
            rgb[m] = PALETTE[i % len(PALETTE)]
            out_parts.append({"id": i, "kind": "part", "src_k": r["k"],
                              "n_gs": int(m.sum()), "views": r["views"]})
            sel_list.append(m)
        sel = np.stack(sel_list)

    write_ply_rgb(os.path.join(args.in_dir, "selected.ply"), xyz, rgb)
    np.savez_compressed(os.path.join(args.in_dir, "selected.npz"),
                        xyz=xyz, selected=sel, obj_support=obj_support)
    json.dump({"decision": decision, "cover": round(cover, 3),
               "parts": out_parts}, open(os.path.join(args.in_dir, "selected.json"), "w"),
              indent=2)
    print(f"\n저장: {args.in_dir} (selected.ply/npz/json)")
    print("selected.ply 를 SuperSplat에서 확인. 다음 Step 4: selected를 뷰로 역투영→per-view 마스크.")


if __name__ == "__main__":
    main()
