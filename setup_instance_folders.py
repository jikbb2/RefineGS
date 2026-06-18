#!/usr/bin/env python3
"""prepare_folder 직후 per-object 폴더를 학습 가능한 상태로 보정.

각 data/<scene>/masks/<gid>/ 에 대해:
  (1) images/ 를 *마스크가 있는 프레임만* 실제 파일로 복사 (symlink/전체 → 실복사).
      - filterPLY 임계(len(images)/2)와 학습 뷰가 마스크 프레임 수에 맞아야 하므로 필수.
  (2) sparse/0/points3D.ply 삭제 → 로더가 filterPLY를 실행해 *객체* init 생성.

전제: 마스크는 RGBA(alpha=객체)  (amodal_mask.py가 그렇게 출력).
      cameras.py / dataset_readers.py 패치가 적용돼 있어야 객체만 학습됨.

실행: python setup_instance_folders.py <scene>
"""
import os, sys, glob, shutil


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "replica_room0_v2"
    root = f"data/{scene}/masks"
    gdirs = sorted(d for d in glob.glob(root + "/*/") if os.path.isdir(d))
    n_obj = 0
    for gd in gdirs:
        masks = glob.glob(os.path.join(gd, "masks", "*.png"))
        if not masks:
            continue
        stems = {os.path.splitext(os.path.basename(m))[0] for m in masks}

        # (1) images/ → 마스크 프레임만 실복사
        imgdir = os.path.join(gd, "images")
        # symlink/디렉토리 어느 쪽이든 target의 실파일 경로를 먼저 수집
        listing = glob.glob(os.path.join(os.path.realpath(imgdir), "*")) if os.path.exists(imgdir) else []
        srcmap = {os.path.splitext(os.path.basename(f))[0]: os.path.realpath(f) for f in listing}
        if os.path.islink(imgdir):
            os.unlink(imgdir)
        elif os.path.isdir(imgdir):
            shutil.rmtree(imgdir)
        os.makedirs(imgdir)
        copied = 0
        for st in stems:
            sf = srcmap.get(st)
            if sf and os.path.exists(sf):
                shutil.copy(sf, os.path.join(imgdir, st + ".jpg"))
                copied += 1

        # (2) scene points3D.ply 삭제 → filterPLY가 객체 init 생성
        ply = os.path.join(gd, "sparse", "0", "points3D.ply")
        if os.path.exists(ply):
            os.remove(ply)
        # 이전 실행에서 남은 객체 init도 제거(신선하게)
        old_obj = os.path.join(gd, "points3d.ply")
        if os.path.exists(old_obj):
            os.remove(old_obj)

        gid = os.path.basename(gd.rstrip("/"))
        print(f"  obj {gid:>3}: masks={len(stems)} images_copied={copied}")
        n_obj += 1
    print(f"setup done: {n_obj} objects ({scene})")


if __name__ == "__main__":
    main()
