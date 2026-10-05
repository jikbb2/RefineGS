#!/usr/bin/env python3
"""배치 결과(_field_batch.csv) 요약 — 평가 실패와 방법 실패를 분리.

전체 평균은 무의미하다. 30개 중 상당수가 **GT 매칭이 깨져** baseline 자체가
GT와 안 맞거나(seen F@1cm≈0), SAM3 인스턴스가 더 큰 GT 객체의 일부라
completion 이 부당하게 나쁘다. 이걸 섞으면 평균이 nan 이 되거나 신호가 묻힌다.

분류는 **baseline 만 보고** 한다(우리 결과로 거르면 체리피킹이 된다):

  FAIL : baseline seen F@1cm < fail_f1        → GT 대응 실패. 지표 무의미, 전면 제외
  PART : baseline unseen completion > part_mm → GT 가 인스턴스보다 훨씬 큼(부분-전체 불일치).
                                                seen 지표만 유효
  OK   : 나머지                                → 주 표

  python summarize_field_batch.py output/.../\\_field_batch.csv
"""
import argparse
import collections
import csv
import math
import statistics as st


def f(r, k):
    try:
        v = float(r[k])
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--fail_f1", type=float, default=0.50,
                    help="baseline seen F@1cm 이 이 값 미만이면 GT 대응 실패로 제외")
    ap.add_argument("--part_mm", type=float, default=500.0,
                    help="baseline unseen completion 이 이 값 초과면 부분-전체 불일치")
    ap.add_argument("--degrade_x", type=float, default=3.0,
                    help="ours seen accuracy 가 baseline 의 이 배수 초과면 '관측 훼손' 플래그")
    args = ap.parse_args()

    by = collections.OrderedDict()
    for r in csv.DictReader(open(args.csv)):
        by.setdefault(r["tag"], []).append(r)

    groups = {"OK": [], "PART": [], "FAIL": []}
    for tag, rs in by.items():
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[1]                       # --csv_all → A(baseline), B(ours)
        if not math.isfinite(f(a, "seen_F1.0")) or f(a, "seen_F1.0") < args.fail_f1 \
                or not math.isfinite(f(a, "seen_acc")):
            groups["FAIL"].append((tag, a, b))
        elif f(a, "unseen_comp") > args.part_mm:
            groups["PART"].append((tag, a, b))
        else:
            groups["OK"].append((tag, a, b))

    cols = [("seen_acc", "seenAcc", "{:.2f}"), ("seen_F1.0", "seenF1", "{:.3f}"),
            ("unseen_comp", "unsComp", "{:.1f}"), ("unseen_P2.0", "unsP2", "{:.3f}"),
            ("unseen_R2.0", "unsR2", "{:.3f}"), ("unseen_F2.0", "unsF2", "{:.3f}"),
            ("free_pct", "free%", "{:.2f}")]

    def table(rows, title, keys):
        if not rows:
            return
        print(f"\n=== {title} ({len(rows)}개) ===")
        hdr = ["obj"] + [h for k, h, _ in cols if k in keys] + ["flag"]
        print("  " + "  ".join(f"{h:>16}" for h in hdr))
        for tag, a, b in rows:
            cells = [tag]
            for k, h, fm in cols:
                if k not in keys:
                    continue
                va, vb = f(a, k), f(b, k)
                cells.append((fm.format(va) if math.isfinite(va) else "nan") + "→"
                             + (fm.format(vb) if math.isfinite(vb) else "nan"))
            deg = (math.isfinite(f(a, "seen_acc")) and math.isfinite(f(b, "seen_acc"))
                   and f(b, "seen_acc") > args.degrade_x * f(a, "seen_acc"))
            cells.append("관측훼손" if deg else "")
            print("  " + "  ".join(f"{c:>16}" for c in cells))

    def agg(rows, keys, title):
        if not rows:
            return
        print(f"\n  --- {title} (중앙값, n={len(rows)}) ---")
        for k, h, fm in cols:
            if k not in keys:
                continue
            va = [f(a, k) for _, a, _ in rows if math.isfinite(f(a, k))]
            vb = [f(b, k) for _, _, b in rows if math.isfinite(f(b, k))]
            if not va or not vb:
                continue
            A, B = st.median(va), st.median(vb)
            print(f"  {h:>10}: {A:9.3f} → {B:9.3f}  ({B - A:+.3f})")

    allk = {k for k, _, _ in cols}
    seenk = {"seen_acc", "seen_F1.0", "free_pct"}

    table(groups["OK"], "OK — 주 표", allk)
    agg(groups["OK"], allk, "OK 그룹")
    table(groups["PART"], "PART — GT 가 인스턴스보다 큼(seen 지표만 유효)", seenk)
    agg(groups["PART"], seenk, "PART 그룹 seen 지표")
    table(groups["FAIL"], "FAIL — GT 대응 실패(제외)", {"seen_F1.0", "unseen_comp"})

    n = sum(len(v) for v in groups.values())
    print(f"\n  총 {n}개 → OK {len(groups['OK'])} / PART {len(groups['PART'])} "
          f"/ FAIL {len(groups['FAIL'])}")
    deg = [t for t, a, b in groups["OK"]
           if math.isfinite(f(a, "seen_acc")) and math.isfinite(f(b, "seen_acc"))
           and f(b, "seen_acc") > args.degrade_x * f(a, "seen_acc")]
    if deg:
        print(f"  ⚠ OK 그룹 내 '관측 훼손' 객체: {', '.join(deg)}  "
              f"→ 개별 확인 필요(생성 형상/스케일 오류 의심)")
    print("\n  ※ 분류는 baseline 만 보고 결정 — 우리 결과로 거르면 체리피킹이 된다.")


if __name__ == "__main__":
    main()