#!/usr/bin/env python3
"""객체별 메쉬를 실내 씬 하나로 합친다.

왜 씬 단위인가:
  per-object 평가는 SAM3 인스턴스 = GT 객체를 전제한다. 실제로는 그렇지 않다.
    · obj8(꽃병) 은 GT 라벨 셋(id11 42.6% / id35 40.1% / id55 14.8%)에 걸쳐 있다
    · obj11(담요)의 prior 는 담요를 완성하며 밑의 소파 일부까지 생성한다
    · obj15/19 는 다른 쿠션에 가려 쪼개진 조각이다
  이 여분 기하는 인스턴스 라벨 밖이라 per-object 에서 오답으로 세지지만
  씬 안에서는 틀린 게 아니다. 씬 단위로 합치면 이 왜곡이 정의상 사라진다.

합치는 방식:
  삼각형 수프 union(단순 concatenate). 겹침을 평균내지 않는다 — 겹치는 곳에서
  두 표면이 어긋나 있으면 그건 우리 출력의 실제 성질이고, 숨기면 안 된다.
  (복셀 union 은 겹침을 감추면서 방 전체 그리드가 필요해 비싸다)

  python merge_scene.py --out ~/prior/scene_field.ply
  python merge_scene.py --mesh fuse_post.ply --out ~/prior/scene_base.ply
  python merge_scene.py --exclude 0,9,20,21,25,26,28,29,31,33 --out ...
"""
import argparse
import glob
import os

import numpy as np
import open3d as o3d

# 배치에서 융합이 폭주했거나(seen acc 44~2513mm) 베이스라인 평가부터
# 깨진(F@1cm≈0 또는 nan) 객체들. 씬에 넣으면 방 전체 지표를 오염시킨다.
DEFAULT_EXCLUDE = "0,9,20,21,25,26,28,29,31,33"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser(
        "~/RefineGS/output/replica_room0_v2/refinegs_full"))
    ap.add_argument("--iter", default=7000, type=int)
    ap.add_argument("--mesh", default="fused_field_post.ply",
                    help="객체 폴더에서 집을 파일명. baseline 씬을 만들려면 fuse_post.ply")
    ap.add_argument("--fallback", default="",
                    help="--mesh 가 없을 때 대신 쓸 파일명(예: fuse_post.ply). "
                         "비우면 그 객체를 건너뛴다")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help=f"제외할 gid(쉼표). 기본 '{DEFAULT_EXCLUDE}'. "
                         "전부 포함하려면 'none'")
    ap.add_argument("--only", default="", help="이 gid 들만 포함(쉼표)")
    ap.add_argument("--min_verts", default=100, type=int,
                    help="정점이 이보다 적은 메쉬는 건너뛴다(빈 껍데기 방지)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ex = set() if args.exclude.strip().lower() in ("none", "") else \
        {int(x) for x in args.exclude.split(",") if x.strip()}
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None

    gids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(args.root, "*"))
                  if os.path.basename(d).isdigit())

    merged = o3d.geometry.TriangleMesh()
    used, skipped, fell_back = [], [], []
    for g in gids:
        if g in ex or (only is not None and g not in only):
            skipped.append((g, "제외"))
            continue
        d = os.path.join(args.root, str(g), "train", f"ours_{args.iter}")
        p = os.path.join(d, args.mesh)
        if not os.path.isfile(p) and args.fallback:
            pf = os.path.join(d, args.fallback)
            if os.path.isfile(pf):
                p, _ = pf, fell_back.append(g)
        if not os.path.isfile(p):
            skipped.append((g, "파일 없음"))
            continue
        m = o3d.io.read_triangle_mesh(p)
        if len(m.vertices) < args.min_verts:
            skipped.append((g, f"정점 {len(m.vertices)}개"))
            continue
        merged += m
        used.append((g, len(m.vertices), len(m.triangles)))

    if not used:
        raise SystemExit("합칠 메쉬가 없습니다 — --root/--mesh 경로를 확인하세요")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(args.out, merged)
    assert ok, f"저장 실패: {args.out}"

    V = np.asarray(merged.vertices)
    print(f"[merge] {args.mesh}  객체 {len(used)}개 → {args.out}")
    print(f"  정점 {len(V):,}  면 {len(np.asarray(merged.triangles)):,}")
    print(f"  bbox  min {np.round(V.min(0), 2)}  max {np.round(V.max(0), 2)}  "
          f"크기 {np.round(V.max(0) - V.min(0), 2)}m")
    print("  포함: " + ", ".join(str(g) for g, _, _ in used))
    if fell_back:
        print(f"  ⚠ fallback({args.fallback}) 사용: " + ", ".join(map(str, fell_back)))
    if skipped:
        print("  건너뜀: " + ", ".join(f"{g}({r})" for g, r in skipped))
    # 가장 큰 객체가 방 전체를 덮고 있으면 오배치 잔재가 섞인 것이다
    big = max(used, key=lambda x: x[1])
    print(f"  최대 기여 객체 obj{big[0]} ({big[1]:,}정점, "
          f"{big[1]/len(V)*100:.0f}%) — 한 객체가 과반이면 오배치를 의심할 것")


if __name__ == "__main__":
    main()
