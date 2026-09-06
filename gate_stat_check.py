#!/usr/bin/env python3
"""게이트 통계 후보를 배치 결과와 상관시켜 본다 — 임계를 정하기 전에 분리도부터.

배경:
  기존 게이트(min_unknown_frac)는 '생성 표면 껍질 중 unknown 비율'(ufrac)을 썼는데,
  |SG|<1.5vox 의 얇은 껍질에서 alpha<0.25 경계 복셀을 세는 통계라 실행마다
  38.4/38.8/40.0% 로 흔들렸다. 임계 0.20 근처 객체는 결과가 뒤집힌다.
  그래서 폐기했는데, 폐기 후 배치에서 obj16(0.914→0.646), obj8(0.070→0.025),
  obj10 이 악화됐다 — 게이트의 판단 자체는 옳았고 통계가 나빴던 것이다.

후보:
  '생성 내부(미관측)' = (SG<0) & (Wo==0) & ~FREE & ~OTH 의 부피 비율.
  prior 가 실제로 채우는 부피이므로 얇은 껍질보다 훨씬 안정적이다.
  로그에 이미 찍히고 있다.

사용:
  python gate_stat_check.py --logdir ~/prior/logs --csv <배치 CSV>

출력: 객체별 (통계값, unseen F@2 변화, seen F@1 변화) 와 임계별 혼동행렬.
"""
import argparse
import csv
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.expanduser("~/prior/logs"))
    ap.add_argument("--csv", default=os.path.expanduser(
        "~/RefineGS/output/replica_room0_v2/refinegs_full/_field_batch.csv"))
    ap.add_argument("--noise", type=float, default=0.0066,
                    help="unseen F@2 해석 임계(2σ). 이보다 작은 변화는 무변화로 본다")
    ap.add_argument("--thresholds", default="0.02,0.05,0.10,0.20,0.30")
    args = ap.parse_args()

    # 로그에서 '생성 내부(미관측) X%' 와 기존 ufrac 을 뽑는다
    stat, ufrac = {}, {}
    for p in glob.glob(os.path.join(args.logdir, "fuse_*.log")):
        g = os.path.basename(p)[5:-4]
        t = open(p, errors="ignore").read()
        m = re.findall(r"생성 내부\(미관측\)\s+([\d.]+)%", t)
        if m:
            stat[g] = float(m[-1])
        m2 = re.findall(r"생성 표면 중 unknown 비율\s+([\d.]+)%", t)
        if m2:
            ufrac[g] = float(m2[-1])

    # CSV 에서 A/B 지표를 뽑는다 (csv_all 이면 A 행과 B 행이 쌍으로 있다)
    rows = list(csv.DictReader(open(os.path.expanduser(args.csv))))
    by = {}
    for r in rows:
        g = re.sub(r"\D", "", r.get("tag", ""))
        by.setdefault(g, {})["A" if r["mesh"].startswith("fuse_post") else "B"] = r

    rec = []
    for g, ab in sorted(by.items(), key=lambda x: int(x[0]) if x[0] else 0):
        if "A" not in ab or "B" not in ab or g not in stat:
            continue
        try:
            f2a, f2b = float(ab["A"]["unseen_F2.0"]), float(ab["B"]["unseen_F2.0"])
            s1a, s1b = float(ab["A"]["seen_F1.0"]), float(ab["B"]["seen_F1.0"])
            aca = float(ab["A"]["seen_acc"])
        except (KeyError, ValueError):
            continue
        if aca != aca or s1a < 0.30:        # 베이스라인 파손 제외
            continue
        if float(ab["B"]["seen_acc"]) > 20:  # 폭주 제외
            continue
        rec.append((g, stat[g], ufrac.get(g, float("nan")), f2b - f2a, s1b - s1a))

    if not rec:
        raise SystemExit("매칭된 객체가 없습니다 — --logdir/--csv 경로와 CSV 헤더를 확인하세요")

    print(f"{'obj':>6}{'생성내부%':>10}{'ufrac%':>9}{'ΔunsF2':>9}{'ΔseenF1':>9}  판정")
    for g, s, u, df2, ds1 in sorted(rec, key=lambda x: x[1]):
        v = "개선" if df2 > args.noise else ("악화" if df2 < -args.noise else "무변화")
        print(f"{g:>6}{s:>10.2f}{u:>9.1f}{df2:>+9.3f}{ds1:>+9.3f}  {v}")

    print(f"\n임계별 혼동행렬 (prior 적용 = 통계 ≥ 임계)")
    print(f"{'임계%':>7}{'맞게적용':>9}{'헛되이적용':>11}{'맞게차단':>9}{'놓친차단':>9}")
    for t in [float(x) for x in args.thresholds.split(",")]:
        tp = sum(1 for r in rec if r[1] >= t and r[3] > args.noise)     # 적용, 개선
        fp = sum(1 for r in rec if r[1] >= t and r[3] < -args.noise)    # 적용, 악화
        tn = sum(1 for r in rec if r[1] < t and r[3] < -args.noise)     # 차단, 악화였을 것
        fn = sum(1 for r in rec if r[1] < t and r[3] > args.noise)      # 차단, 개선 놓침
        print(f"{t:>7.2f}{tp:>9}{fp:>11}{tn:>9}{fn:>9}")
    print("\n'헛되이적용' 과 '놓친차단' 이 동시에 0 이 되는 임계가 있으면 그 통계는 완벽히 분리한다.")
    print("없으면 두 오류의 비용을 비교해 고른다 — 악화(obj16: seen F@1 0.914→0.646)가")
    print("개선을 놓치는 것보다 비싸다면 임계를 높게 잡는다.")


if __name__ == "__main__":
    main()
