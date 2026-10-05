#!/usr/bin/env python3
"""mesh에서 작은 분리 컴포넌트(부유물) 제거. 면적 비율 임계 미만 컴포넌트 삭제.

실행: python clean_components.py --in a.ply --out b.ply --min_frac 0.02
"""
import argparse
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_frac", type=float, default=0.02,
                    help="전체 정점 대비 이 비율 미만 컴포넌트는 제거")
    a = ap.parse_args()

    m = trimesh.load(a.inp, process=False)
    comps = m.split(only_watertight=False)
    if not comps:
        m.export(a.out)
        print(f"  components=0 (그대로) verts={len(m.vertices)} -> {a.out}")
        return
    comps = sorted(comps, key=lambda c: len(c.vertices), reverse=True)
    thr = a.min_frac * len(m.vertices)
    keep = [c for c in comps if len(c.vertices) >= thr]
    out = trimesh.util.concatenate(keep) if keep else comps[0]
    out.export(a.out)
    print(f"  components {len(comps)} -> kept {len(keep)}  "
          f"verts {len(m.vertices)}->{len(out.vertices)}  -> {a.out}")


if __name__ == "__main__":
    main()
