#!/usr/bin/env python3
"""RefineGS — 조건부 novel-view 생성 어댑터 (task 5, model-agnostic).

입력  : soft_in/  (render_hole_novel.py --soft_out)
          view_<i>.jpg    조건용 RGB 렌더(현재 형상을 그 pose에서)
          weight_<i>.png  연속 soft weight(미관측·gen↑ = refine 대상)
          poses.npz       카메라
출력  : gen_out/  gen_<i>.jpg + weight_<i>.png(soft, 학습 loss용) + poses.npz

백엔드(--backend):
  copy   : view 를 그대로 복사(identity). 주입 배관(task6) 검증용.
  see3d  : See3D inference.py 호출(디렉토리 배치, multi-view diffusion).
           warp=view, mask=binary(weight 임계: 흰=known/검=hole) → predict → gen.
           ★조건뷰는 *실측 재학습된* 모델에서 렌더하는 게 좋음(조건 품질↑).

See3D 실행(see3d 백엔드 내부, See3D env 에서):
  python inference.py --base_model_path <ckpt> --source_imgs_dir <ref>/ \
      --warp_root_dir <tmp_warps> --output_dir <tmp_out>

실행:
  # 배관검증
  python generate_novel_views.py --soft_in .../soft_in --out .../gen_out --backend copy
  # 실제 생성(See3D env, See3D 디렉토리에서)
  python generate_novel_views.py --soft_in .../soft_in --out .../gen_out --backend see3d \
      --see3d_root /home/elicer/See3D --base_model_path <See3D ckpt> \
      --ref_views /home/elicer/See3D/dataset/refinegs_obj24/ref --hole_thr 0.3

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
            shutil.copy(wp, os.path.join(args.out, f"weight_{i:04d}.png"))
        meta.append(dict(idx=i, gen=f"gen_{i:04d}.jpg", weight=f"weight_{i:04d}.png"))
        print(f"[copy] gen_{i:04d} ← view_{i:04d}")
    finalize(args.out, args.soft_in, meta)
    print(f"\n→ {len(meta)} (copy=identity) → {args.out}. patch_train_novelview 로 주입 배관 검증 가능.")


# ---------- see3d (directory batch) ----------
def run_see3d(args):
    if not args.base_model_path or not args.ref_views:
        raise SystemExit("see3d 백엔드는 --base_model_path 와 --ref_views(관측 앵커 dir) 필요")
    recs = load_poses(args.soft_in)
    os.makedirs(args.out, exist_ok=True)
    warp_dir = tempfile.mkdtemp(prefix="see3d_warps_", dir=os.path.expanduser("~/tmp")
                                if os.path.isdir(os.path.expanduser("~/tmp")) else None)
    out_tmp = tempfile.mkdtemp(prefix="see3d_out_", dir=os.path.dirname(warp_dir))

    # 1) warp_*.jpg(=view) + mask_*.png(See3D: 흰=known/검=hole) 준비
    idxs = []
    for rec in recs:
        i = int(rec["idx"])
        vp = os.path.join(args.soft_in, f"view_{i:04d}.jpg")
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if not (os.path.exists(vp) and os.path.exists(wp)):
            continue
        Image.open(vp).convert("RGB").save(os.path.join(warp_dir, f"warp_{i:04d}.jpg"), quality=95)
        w = np.asarray(Image.open(wp).convert("L")).astype(np.float32) / 255.0
        known = (w < args.hole_thr).astype(np.uint8) * 255          # 흰=known(weight 낮음), 검=hole(미관측)
        Image.fromarray(known).save(os.path.join(warp_dir, f"mask_{i:04d}.png"))
        idxs.append(i)
    if not idxs:
        raise SystemExit("see3d 입력 0개 — soft_in 비었거나 weight 전부 0. MIN_WEIGHT/hole_thr 확인.")

    ref = args.ref_views if args.ref_views.endswith("/") else args.ref_views + "/"   # inference.py 가 문자열 concat
    cmd = [sys.executable, os.path.join(args.see3d_root, "inference.py"),
           "--base_model_path", args.base_model_path,
           "--source_imgs_dir", ref,
           "--warp_root_dir", warp_dir,
           "--output_dir", out_tmp]
    if args.single_view:
        cmd.append("--single_view")
    print("See3D inference:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.see3d_root)

    # 2) predict_warp_<i>*.jpg → gen_<i>.jpg
    meta = []
    preds = glob.glob(os.path.join(out_tmp, "predict_*warp_*"))
    for p in preds:
        m = re.search(r"warp_(\d+)", os.path.basename(p))
        if not m:
            continue
        i = int(m.group(1))
        Image.open(p).convert("RGB").save(os.path.join(args.out, f"gen_{i:04d}.jpg"), quality=95)
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if os.path.exists(wp):
            shutil.copy(wp, os.path.join(args.out, f"weight_{i:04d}.png"))
        meta.append(dict(idx=i, gen=f"gen_{i:04d}.jpg", weight=f"weight_{i:04d}.png"))
        print(f"[see3d] gen_{i:04d} ← predict {os.path.basename(p)}")
    if not meta:
        raise SystemExit(f"See3D 출력 0개 — {out_tmp} 확인(predict_*warp_* 없음).")
    finalize(args.out, args.soft_in, meta)
    print(f"\n→ {len(meta)} (see3d) → {args.out}")


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
    args = ap.parse_args()
    (run_copy if args.backend == "copy" else run_see3d)(args)


if __name__ == "__main__":
    main()
