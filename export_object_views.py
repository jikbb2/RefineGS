#!/usr/bin/env python3
"""객체 단위 masked multi-view 를 3D 생성 모델(TRELLIS/Hunyuan3D) 조건 입력용으로 export.

VARCO 처럼 전체 RGB crop 을 넣으면 인접 객체(소파 등)까지 생성됨.
→ 객체 마스크로 잘라낸 cutout 여러 장(known view)을 조건으로 주면 해당 객체만 생성되고,
  multi-view 조건이라 비율·방향 왜곡과 할루시네이션이 줄어든다.

출력:
  <out>/views/<stem>.png     RGBA cutout(객체만, 배경 투명, bbox crop)
  <out>/poses.json           각 뷰의 COLMAP pose(정합 pose-init 용)

  python export_object_views.py --gid 6 --n_views 6 \
    --masks_root data/replica_room0_v2/masks --images data/replica_room0_v2/images \
    --colmap data/replica_room0_v2/sparse/0 \
    --stems ~/See3D/dataset/stage6/clean_stems/6.txt \
    --out ~/gen_input/obj6
"""
import os, json, glob, argparse
import numpy as np
import cv2
from PIL import Image
from warp_gt_to_pose import read_colmap, cam_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", required=True)
    ap.add_argument("--masks_root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--stems", default="")
    ap.add_argument("--n_views", type=int, default=6, help="조건 뷰 수(면적·각도 다양성)")
    ap.add_argument("--pad", type=float, default=0.08, help="bbox 여백 비율")
    ap.add_argument("--bg", default="transparent", choices=["white", "transparent"],
                    help="TRELLIS-2 는 RGBA alpha 를 마스크로 직접 사용 → transparent 권장")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outv = os.path.join(os.path.expanduser(args.out), "views")
    os.makedirs(outv, exist_ok=True)
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}

    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(args.masks_root, args.gid, "masks", "*")))
    stems = [s for s in stems if s in cams]

    def mask_of(s):
        mp = os.path.join(args.masks_root, args.gid, "masks", s + ".png")
        if not os.path.exists(mp):
            return None
        a = np.array(Image.open(mp))
        return (a[..., 3] if (a.ndim == 3 and a.shape[2] == 4)
                else np.array(Image.open(mp).convert("L"))) > 0

    # 면적 상위에서 시점 다양성 위해 카메라 방위각 기준 균등 선택
    cand = [(s, mask_of(s)) for s in stems]
    cand = [(s, m) for s, m in cand if m is not None and m.sum() > 500]
    areas = np.array([m.sum() for _, m in cand])
    order = np.argsort(-areas)[:max(args.n_views * 3, args.n_views)]  # 면적 상위 풀
    # 풀에서 카메라 방위각 균등 샘플
    az = []
    for i in order:
        s = cand[i][0]; c = cams[s]
        C = cam_center(c["R"], c["t"]); az.append(np.arctan2(C[1], C[0]))
    az = np.array(az)
    pick, used = [], []
    for target in np.linspace(-np.pi, np.pi, args.n_views, endpoint=False):
        j = order[np.argmin(np.abs(np.angle(np.exp(1j * (az - target)))))]
        if j not in used:
            used.append(j); pick.append(cand[j][0])
    print(f"조건 뷰 {len(pick)}장: {pick}")

    poses = {}
    for s in pick:
        m = mask_of(s)
        # RGB 로드(확장자 자동)
        src = None
        for ext in (".jpg", ".jpeg", ".png", ".JPEG"):
            p = os.path.join(args.images, s + ext)
            if os.path.exists(p):
                src = p; break
        if src is None:
            continue
        img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0

        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        py = int((y1 - y0) * args.pad); px = int((x1 - x0) * args.pad)
        y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
        x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
        crop = img[y0:y1, x0:x1]; mc = m[y0:y1, x0:x1]

        if args.bg == "transparent":
            rgba = np.dstack([crop, (mc * 255).astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(os.path.join(outv, s + ".png"))
        else:
            out = crop.copy(); out[~mc] = 255
            Image.fromarray(out).save(os.path.join(outv, s + ".png"))

        c = cams[s]
        w2c = np.eye(4); w2c[:3, :3] = c["R"]; w2c[:3, 3] = c["t"]
        poses[s] = {"w2c": w2c.tolist(),
                    "K": [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]],
                    "crop": [int(x0), int(y0), int(x1), int(y1)],
                    "W": c["W"], "H": c["H"]}

    json.dump(poses, open(os.path.join(os.path.expanduser(args.out), "poses.json"), "w"), indent=1)
    print(f"→ {outv}  ({len(poses)}장) + poses.json")


if __name__ == "__main__":
    main()
