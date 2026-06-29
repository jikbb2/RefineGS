#!/usr/bin/env python3
"""RefineGS — 조건부 novel-view 생성 어댑터 (task 5, model-agnostic).

입력  : soft_in/  (render_hole_novel.py --soft_out 출력)
          view_<i>.jpg    조건용 RGB 렌더(현재 base⊕gen 형상을 그 pose에서 본 것)
          weight_<i>.png  연속 soft weight(미관측·gen↑ = refine 대상)
          poses.npz       카메라(world_view/full_proj/FoVx/FoVy/W/H)
출력  : gen_out/  gen_<i>.jpg  (그 pose에서 *refine된* RGB) + poses.npz 복사 + meta.json

백엔드(--backend):
  copy        : view 를 그대로 복사(identity). ★See3D 없이 주입 파이프라인(task6) 먼저 검증용.
  see3d       : See3D inference 호출(조건=view+weight, 주변 관측뷰). ⚠️ CLI 미확정 → fill 필요.
  viewcrafter : ViewCrafter 호출(render-conditioned). ⚠️ 미확정.

설계 원칙(할루시네이션 억제): 생성기는 *백지*가 아니라 view(현재 형상 렌더)를 조건으로 받아 다듬는다.
weight 가 큰 곳(미관측)은 자유도↑, weight≈0(관측)은 view 를 거의 보존하도록 백엔드에 전달.

실행:
  # 1) 먼저 copy 로 주입 파이프라인 검증
  python generate_novel_views.py --soft_in ~/See3D/dataset/refinegs_obj24/soft_in \
      --out ~/See3D/dataset/refinegs_obj24/gen_out --backend copy
  # 2) 실제 생성(See3D CLI 확정 후)
  python generate_novel_views.py --soft_in ... --out ... --backend see3d \
      --see3d_root /home/elicer/See3D --ref_views <관측뷰dir>

Deps: numpy, PIL.  (see3d/viewcrafter 백엔드는 해당 레포 의존)
"""
import argparse, os, json, shutil
import numpy as np
from PIL import Image


def load_poses(soft_in):
    p = os.path.join(soft_in, "poses.npz")
    if not os.path.exists(p):
        raise SystemExit(f"poses.npz 없음: {p} (render_hole_novel --soft_out 먼저)")
    recs = np.load(p, allow_pickle=True)["records"]
    return list(recs)


# ---------- 백엔드 ----------
def backend_copy(view_path, weight_path, rec, args):
    """identity: 현재 렌더를 그대로 '생성물'로. 주입 파이프라인 플러밍 검증용."""
    return Image.open(view_path).convert("RGB")


def backend_see3d(view_path, weight_path, rec, args):
    """See3D inference 호출.
    ⚠️ 채울 부분: See3D inference.py 의 실제 CLI/함수 시그니처.
    일반 형태:
      conditioning = view(현재 형상 렌더) + mask(=weight 임계 또는 weight 자체) + 주변 관측뷰(ref_views)
      output       = 그 pose 의 inpaint/refine 된 RGB
    예시(자리표시 — 실제 인자명으로 교체):
      cmd = [sys.executable, f"{args.see3d_root}/inference.py",
             "--input", view_path, "--mask", weight_path,
             "--ref_dir", args.ref_views, "--out", tmp_out, ...]
      subprocess.run(cmd, check=True); return Image.open(tmp_out)
    """
    raise NotImplementedError(
        "see3d 백엔드 미구현 — See3D inference CLI(인자명) 공유 시 채움. "
        "우선 --backend copy 로 task6 주입을 검증하세요.")


def backend_viewcrafter(view_path, weight_path, rec, args):
    raise NotImplementedError("viewcrafter 백엔드 미구현 — 필요 시 ViewCrafter CLI 공유.")


BACKENDS = {"copy": backend_copy, "see3d": backend_see3d, "viewcrafter": backend_viewcrafter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_in", required=True, help="render_hole_novel --soft_out 디렉토리")
    ap.add_argument("--out", required=True, help="생성물 출력 디렉토리 gen_out")
    ap.add_argument("--backend", default="copy", choices=list(BACKENDS))
    ap.add_argument("--see3d_root", default="/home/elicer/See3D")
    ap.add_argument("--ref_views", default="", help="조건용 주변 관측뷰 디렉토리(see3d)")
    args = ap.parse_args()

    recs = load_poses(args.soft_in)
    os.makedirs(args.out, exist_ok=True)
    fn = BACKENDS[args.backend]

    meta = []
    for rec in recs:
        i = int(rec["idx"])
        vp = os.path.join(args.soft_in, f"view_{i:04d}.jpg")
        wp = os.path.join(args.soft_in, f"weight_{i:04d}.png")
        if not os.path.exists(vp):
            print(f"[skip] {vp} 없음"); continue
        img = fn(vp, wp, rec, args)
        op = os.path.join(args.out, f"gen_{i:04d}.jpg")
        img.save(op, quality=95)
        meta.append(dict(idx=i, gen=os.path.basename(op),
                         weight=os.path.basename(wp), stem=str(rec.get("stem", ""))))
        print(f"[{args.backend}] gen_{i:04d}  ← view_{i:04d}")

    # poses.npz 그대로 복사(주입이 카메라 복원에 사용) + meta
    shutil.copy(os.path.join(args.soft_in, "poses.npz"), os.path.join(args.out, "poses.npz"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n→ {len(meta)}개 생성 → {args.out}  (gen_*.jpg + weight 참조 + poses.npz)")
    if args.backend == "copy":
        print("copy 백엔드 = identity. 이제 patch_train_novelview 로 주입 파이프라인(task6) 검증 가능.")


if __name__ == "__main__":
    main()
