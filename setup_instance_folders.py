#!/usr/bin/env python3
"""prepare_folder 직후 per-object 폴더를 학습 가능한 상태로 보정. (멱등)

각 data/<scene>/masks/<gid>/ 에 대해:
  (1) images/ 를 *마스크가 있는 프레임만* 실제 파일로 복사.
      소스는 항상 scene 레벨 data/<scene>/images (심볼릭→nice-slam 또는 실파일).
      - filterPLY 임계(len(images)/2)와 학습 뷰가 마스크 프레임 수에 맞아야 하므로 필수.
  (1b) GT depth 복사 → depths/<frame-stem>.png  (depth supervision; nice-slam: frameNNN.jpg <-> depthNNN.png)
  (2) sparse/0/points3D.ply (및 이전 points3d.ply) 삭제 → 로더가 filterPLY로 *객체* init 생성.

전제: 마스크는 RGBA(alpha=객체). cameras.py/dataset_readers.py 패치 적용.
소스를 per-object 폴더가 아니라 scene 레벨에서 읽으므로 몇 번 재실행해도 안전.

실행: python setup_instance_folders.py <scene>
"""
import os, sys, glob, shutil


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "replica_room0_v2"
    root = f"data/{scene}/masks"
    scene_img = os.path.realpath(f"data/{scene}/images")   # 원본 (frame*.jpg + depth*.png)
    if not os.path.isdir(scene_img):
        print(f"[ERROR] scene 이미지 소스 없음: data/{scene}/images (심볼릭 깨짐?)"); sys.exit(1)

    gdirs = sorted(d for d in glob.glob(root + "/*/") if os.path.isdir(d))
    n_obj = 0
    for gd in gdirs:
        masks = glob.glob(os.path.join(gd, "masks", "*.png"))
        if not masks:
            continue
        stems = sorted(os.path.splitext(os.path.basename(m))[0] for m in masks)

        # (1) images/ 재생성 — scene 소스에서 마스크 프레임만 실복사
        imgdir = os.path.join(gd, "images")
        if os.path.islink(imgdir):
            os.unlink(imgdir)
        elif os.path.isdir(imgdir):
            shutil.rmtree(imgdir)
        os.makedirs(imgdir)
        # (1b) depths/ 재생성
        depthdir = os.path.join(gd, "depths")
        shutil.rmtree(depthdir, ignore_errors=True); os.makedirs(depthdir)

        copied = dcopied = 0
        for st in stems:
            isrc = os.path.join(scene_img, st + ".jpg")
            if os.path.exists(isrc):
                shutil.copy(os.path.realpath(isrc), os.path.join(imgdir, st + ".jpg")); copied += 1
            dsrc = os.path.join(scene_img, st.replace("frame", "depth") + ".png")  # nice-slam 명명
            if os.path.exists(dsrc):
                shutil.copy(os.path.realpath(dsrc), os.path.join(depthdir, st + ".png")); dcopied += 1

        # (2) scene ply / 이전 객체 init 제거 → filterPLY 재실행
        for p in (os.path.join(gd, "sparse", "0", "points3D.ply"), os.path.join(gd, "points3d.ply")):
            if os.path.exists(p):
                os.remove(p)

        gid = os.path.basename(gd.rstrip("/"))
        print(f"  obj {gid:>3}: masks={len(stems)} images={copied} depths={dcopied}")
        n_obj += 1
    print(f"setup done: {n_obj} objects ({scene})  [scene_img={scene_img}]")


if __name__ == "__main__":
    main()
