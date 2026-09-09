#!/usr/bin/env python3
"""씬 평가용 GT 라벨 목록을 객체별 eval 로그에서 모은다.

왜 필요한가:
  mesh_semantic.ply 에는 object_id 가 94종(벽·바닥·천장 포함) 있는데 우리가 합친 씬은
  24개 객체뿐이다. GT 전체와 비교하면 재구성하지 않은 70개가 전부 미달로 계상되어
  completion 이 1000mm 가 된다(실측). recall 계열이 통째로 무의미해진다.
  → 우리가 실제로 다루는 객체의 GT 라벨만 모아 --gt_labels 로 넘긴다.

  라벨은 객체별 평가가 이미 auto-match 로 정해놓았으므로 로그에서 읽으면 된다.

사용:
  python scene_gt_labels.py                      # 병합 기본 객체 집합
  python scene_gt_labels.py --gids 1,2,6,22      # 직접 지정
  → 쉼표 목록을 출력. eval_seen_unseen.py --gt_labels "$(python scene_gt_labels.py)"
"""
import argparse
import glob
import os
import re

# merge_scene.py 의 DEFAULT_EXCLUDE 와 같아야 한다
DEFAULT_EXCLUDE = {0, 9, 20, 21, 25, 26, 28, 29, 31, 33}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.expanduser("~/prior/logs"))
    ap.add_argument("--gids", default="", help="쉼표 목록. 비우면 로그에 있는 전부 - 제외목록")
    ap.add_argument("--max_dist", type=float, default=50.0,
                    help="recon→GT 거리 중앙값이 이 값(mm)을 넘는 객체는 매칭이 무의미하므로 제외")
    ap.add_argument("--quiet", action="store_true", help="라벨 목록만 출력")
    args = ap.parse_args()

    want = {int(x) for x in args.gids.split(",") if x.strip()} if args.gids else None
    rows, labels = [], {}
    for p in sorted(glob.glob(os.path.join(args.logdir, "eval_*.log"))):
        g = os.path.basename(p)[5:-4]
        if not g.isdigit():
            continue
        gi = int(g)
        if want is not None and gi not in want:
            continue
        if want is None and gi in DEFAULT_EXCLUDE:
            continue
        t = open(p, errors="ignore").read()
        m = re.findall(r"채택 라벨 \[([^\]]*)\]", t)
        d = re.findall(r"recon→GT 거리 중앙값 ([\d.]+)mm", t)
        if not m:
            rows.append((gi, [], None, "라벨 없음"))
            continue
        lab = [int(x) for x in m[-1].replace(" ", "").split(",") if x]
        dist = float(d[-1]) if d else None
        why = ""
        if dist is not None and dist > args.max_dist:
            why = f"매칭 무의미({dist:.0f}mm)"
        else:
            for L in lab:
                labels.setdefault(L, []).append(gi)
        rows.append((gi, lab, dist, why))

    if not args.quiet:
        print(f"{'obj':>5}{'거리mm':>9}  라벨", file=os.sys.stderr)
        for gi, lab, dist, why in sorted(rows):
            ds = f"{dist:.1f}" if dist is not None else "-"
            print(f"{gi:>5}{ds:>9}  {lab}{'  ← ' + why if why else ''}", file=os.sys.stderr)
        dup = {L: v for L, v in labels.items() if len(v) > 1}
        if dup:
            print(f"\n여러 객체가 같은 GT 라벨에 매칭됨(정상 — SAM3 인스턴스가 GT 를 쪼갬):",
                  file=os.sys.stderr)
            for L, v in sorted(dup.items()):
                print(f"  id{L} ← obj {v}", file=os.sys.stderr)
        print(f"\n채택 라벨 {len(labels)}종 / 객체 {len(rows)}개\n", file=os.sys.stderr)

    print(",".join(str(x) for x in sorted(labels)))


if __name__ == "__main__":
    main()
