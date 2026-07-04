#!/usr/bin/env python3
"""RefineGS — 조건부 novel-view 생성 어댑터 (task 5, model-agnostic).

[v2 패치]
  - --super_resolution: inference.py SR 경로 사용 + SR_predict_* 출력 우선 채택 (해상도 개선).
  - 출력 수집 glob 이 SR/일반 예측 모두 커버.

입력  : soft_in/  (render_hole_novel.py --soft_out 또는 GT-warp 합성본)
          view_<i>.jpg    조건용 RGB (실측 warp + coarse 렌더 합성 권장)
          weight_<i>.png  weight(높음=hole/refine 대상 — 어댑터 규약)
          poses.npz       카메라
출력  : gen_out/  gen_<i>.jpg + weight_<i>.png(soft, 학습 loss용) + poses.npz

백엔드(--backend):
  copy   : view 를 그대로 복사(identity). 주입 배관(task6) 검증용.
  see3d  : See3D inference.py 호출(디렉토리 배치, multi-view diffusion).
           warp=view, mask=binary(weight 임계: 흰=known/검=hole) → predict → gen.

실행(See3D env, See3D 디렉토리에서):
  python generate_novel_views.py --soft_in .../soft_in_adapter --out .../gen_out --backend see3d \
      --see3d_root /home/elicer/See3D --base_model_path <See3D ckpt> \
      --ref_views .../ref --hole_thr 0.3 --super_resolution

Deps: numpy, PIL. (see3d 백엔드는 See3D 레포/환경)
"""
import argparse, os, json, shutil, glob, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image


def load_poses(soft_in):
    p = os.path.join(soft_in, "poses.npz")
    if not os.path.exists(p):
        raise SystemExit(f"poses.npz 없음: {p} (render_hole_novel --soft_out 먼저)")
    return list(np.load(p, allow_pickle=True)["records"])


def finalize(out, soft_in, meta):
    shutil.copy(os.path.join(soft_in, "poses.npz"), os.path.join(out, "poses.npz"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------- copy (identity) ----------
def run_copy(args):
    recs = load_poses(args.soft_in)
    os.makedirs(args.out, exist_ok=True)
    meta = []
    for rec in recs:
        i = int(rec["idx"])
        vp = os.path.join(args.soft_in, f"view_{i:04d}.jpg")
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if not os.path.exists(vp):
            continue
        Image.open(vp).convert("RGB").save(os.path.join(args.out, f"gen_{i:04d}.jpg"), quality=95)
        if os.path.exists(wp):
            if args.invert_weight:
                w = np.asarray(Image.open(wp).convert("L"))
                Image.fromarray(255 - w).save(os.path.join(args.out, f"weight_{i:04d}.png"))
            else:
                shutil.copy(wp, os.path.join(args.out, f"weight_{i:04d}.png"))
        meta.append(dict(idx=i, gen=f"gen_{i:04d}.jpg", weight=f"weight_{i:04d}.png"))
        print(f"[copy] gen_{i:04d} ← view_{i:04d}")
    finalize(args.out, args.soft_in, meta)
    print(f"\n→ {len(meta)} (copy=identity{'/invert_weight' if args.invert_weight else ''}) → {args.out}")


# ---------- see3d (directory batch) ----------
def run_see3d(args):
    if not args.base_model_path or not args.ref_views:
        raise SystemExit("see3d 백엔드는 --base_model_path 와 --ref_views(관측 앵커 dir) 필요")
    recs = load_poses(args.soft_in)
    os.makedirs(args.out, exist_ok=True)
    warp_dir = tempfile.mkdtemp(prefix="see3d_warps_", dir=os.path.expanduser("~/tmp")
                                if os.path.isdir(os.path.expanduser("~/tmp")) else None)
    out_tmp = tempfile.mkdtemp(prefix="see3d_out_", dir=os.path.dirname(warp_dir))

    # 1) warp_*.png(=view) + mask_*.png(See3D: 흰=known/검=hole) 준비
    idxs = []
    for rec in recs:
        i = int(rec["idx"])
        vp = os.path.join(args.soft_in, f"view_{i:04d}.jpg")
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if not (os.path.exists(vp) and os.path.exists(wp)):
            continue
        Image.open(vp).convert("RGB").save(os.path.join(warp_dir, f"warp_{i:04d}.png"))
        w = np.asarray(Image.open(wp).convert("L")).astype(np.float32) / 255.0
        known = (w < args.hole_thr).astype(np.uint8) * 255
        Image.fromarray(known).save(os.path.join(warp_dir, f"mask_{i:04d}.png"))
        idxs.append(i)
    if not idxs:
        raise SystemExit("see3d 입력 0개 — soft_in 비었거나 weight 전부 0. MIN_WEIGHT/hole_thr 확인.")

    ref = args.ref_views if args.ref_views.endswith("/") else args.ref_views + "/"
    cmd = [sys.executable, os.path.join(args.see3d_root, "inference.py"),
           "--base_model_path", args.base_model_path,
           "--source_imgs_dir", ref,
           "--warp_root_dir", warp_dir,
           "--output_dir", out_tmp]
    if args.single_view:
        cmd.append("--single_view")
    if args.super_resolution:
        cmd.append("--super_resolution")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["LD_LIBRARY_PATH"] = ""
    print("See3D inference:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.see3d_root, env=env)

    # 2) 예측 수집: SR 출력(SR_predict_*) 우선, 없으면 일반(predict_*)
    def collect(pattern):
        d = {}
        for p in glob.glob(os.path.join(out_tmp, pattern)):
            m = re.search(r"warp_(\d+)", os.path.basename(p))
            if m:
                d[int(m.group(1))] = p
        return d
    preds = collect("SR_predict_*warp_*") if args.super_resolution else {}
    normal = collect("predict_*warp_*")
    for i, p in normal.items():
        preds.setdefault(i, p)          # SR 없는 인덱스는 일반 예측으로 보충
    meta = []
    for i in sorted(preds):
        Image.open(preds[i]).convert("RGB").save(os.path.join(args.out, f"gen_{i:04d}.jpg"), quality=95)
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if os.path.exists(wp):
            shutil.copy(wp, os.path.join(args.out, f"weight_{i:04d}.png"))
        meta.append(dict(idx=i, gen=f"gen_{i:04d}.jpg", weight=f"weight_{i:04d}.png",
                         src=os.path.basename(preds[i])))
        print(f"[see3d] gen_{i:04d} ← {os.path.basename(preds[i])}")
    if not meta:
        raise SystemExit(f"See3D 출력 0개 — {out_tmp} 확인(predict_*warp_* 없음).")
    finalize(args.out, args.soft_in, meta)
    print(f"\n→ {len(meta)} (see3d{'/SR' if args.super_resolution else ''}) → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="copy", choices=["copy", "see3d"])
    ap.add_argument("--see3d_root", default="/home/elicer/See3D")
    ap.add_argument("--base_model_path", default="", help="See3D diffusion 체크포인트 dir")
    ap.add_argument("--ref_views", default="", help="관측 앵커 이미지 dir (source_imgs_dir)")
    ap.add_argument("--hole_thr", type=float, default=0.3, help="weight>thr = hole(See3D 생성), 이하=known")
    ap.add_argument("--single_view", action="store_true")
    ap.add_argument("--super_resolution", action="store_true",
                    help="inference.py SR 경로 사용 + SR_predict_* 우선 채택 (해상도 개선)")
    ap.add_argument("--invert_weight", action="store_true",
                    help="weight=hole(warp_gt_to_pose) → 학습 weight=validity(1-hole)로 반전. "
                         "GT-warp 를 학습에 쓸 때(검은 영역 제외) 사용.")
    args = ap.parse_args()
    (run_copy if args.backend == "copy" else run_see3d)(args)


if __name__ == "__main__":
    main()
