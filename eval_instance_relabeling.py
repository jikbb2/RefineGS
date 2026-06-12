#!/usr/bin/env python3
#
# RefineGS - tools/eval_instance_relabeling.py
# ---------------------------------------------------------------------------
# Axis 1 (Robust Instance Re-labeling) 정량 평가.
# 예측 인스턴스 마스크(per-label per-frame) vs vMAP GT semantic_instance(uint16) 를
# 전 프레임 집계 → pred 라벨 ↔ GT 인스턴스 전역 매칭 → 지표 계산.
#
# 지표:
#   mIoU         : 매칭된 pred-GT 쌍의 평균 IoU
#   PQ / SQ / RQ : Panoptic Quality (IoU>0.5 매칭 기준)
#   fragmentation: GT 인스턴스 1개당 대응 pred 라벨 수 (1=이상적, ↑=과분할)
#   merge_error  : pred 라벨 1개당 유의 겹침 GT 수 (1=이상적, ↑=용접)
#   purity       : pred 라벨의 dominant-GT 픽셀 비율 평균 (뷰 간 일관성)
#
# 예측 구조 가정: <pred_root>/<label>/<mask_subdir>/<frame_stem>.png
# 프레임 매핑   : frame000010 → semantic_instance_10.png
#
# 사용:
#   python tools/eval_instance_relabeling.py \
#     --pred_root data/replica_room0/masks --mask_subdir masks \
#     --gt_dir <vmap>/room_0/imap/00/semantic_instance \
#     --ignore_ids 0 1
#   (순차 비교: --pred_root data/replica_room0/masks_seq)
# ---------------------------------------------------------------------------

import os
import re
import glob
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image


def frame_index(stem):
    """frame000010 → 10 (숫자만 추출)."""
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits != "" else None


def load_pred_mask(path):
    """RGBA(alpha=mask) 또는 grayscale 모두 처리 → bool (H,W)."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., -1]          # alpha 채널
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="<root>/<label>/<mask_subdir>/<frame>.png")
    ap.add_argument("--mask_subdir", default="masks", help="라벨 폴더 내 마스크 하위폴더 ('' 이면 라벨 폴더 직속)")
    ap.add_argument("--gt_dir", required=True, help="vMAP semantic_instance 폴더")
    ap.add_argument("--gt_prefix", default="semantic_instance_")
    ap.add_argument("--ignore_ids", type=int, nargs="*", default=[0], help="배경/void GT id")
    ap.add_argument("--iou_match", type=float, default=0.5, help="PQ TP 임계 IoU")
    ap.add_argument("--sig_overlap", type=float, default=0.25, help="merge_error 판정 IoU 임계")
    ap.add_argument("--report", default=None, help="per-label 표 저장 경로(csv, 선택)")
    args = ap.parse_args()

    ignore = set(args.ignore_ids)

    # 1) 예측 라벨 폴더 수집
    label_dirs = sorted([d for d in os.listdir(args.pred_root)
                         if os.path.isdir(os.path.join(args.pred_root, d))])
    # 라벨명 → 마스크 파일 목록
    pred = {}   # label -> {frame_idx: path}
    for lab in label_dirs:
        mdir = os.path.join(args.pred_root, lab, args.mask_subdir) if args.mask_subdir else os.path.join(args.pred_root, lab)
        if not os.path.isdir(mdir):
            continue
        fr = {}
        for p in glob.glob(os.path.join(mdir, "*.png")):
            stem = os.path.splitext(os.path.basename(p))[0]
            idx = frame_index(stem)
            if idx is not None:
                fr[idx] = p
        if fr:
            pred[lab] = fr
    print(f"예측 라벨 {len(pred)}개, 평가 프레임 합집합 "
          f"{len(set().union(*[set(v) for v in pred.values()]))}개")

    # 2) 전역 누적: inter[label][gt], pred_area[label], gt_area[gt]
    inter = defaultdict(lambda: defaultdict(np.int64))
    pred_area = defaultdict(np.int64)
    gt_area = defaultdict(np.int64)

    all_frames = sorted(set().union(*[set(v) for v in pred.values()]))
    gt_cache = {}
    for fidx in all_frames:
        gt_path = os.path.join(args.gt_dir, f"{args.gt_prefix}{fidx}.png")
        if not os.path.exists(gt_path):
            print(f"  [warn] GT 없음: {gt_path}")
            continue
        gt = np.array(Image.open(gt_path)).astype(np.int64)   # (H,W) uint16→int
        # gt_area: 이 프레임에 등장하는 (무시 제외) 인스턴스 픽셀 누적
        ids, cnts = np.unique(gt, return_counts=True)
        for gid, c in zip(ids, cnts):
            if gid in ignore:
                continue
            gt_area[int(gid)] += int(c)

        for lab, fr in pred.items():
            if fidx not in fr:
                continue
            pm = load_pred_mask(fr[fidx])
            if pm.shape != gt.shape:
                # 해상도 불일치 방어 (드물게)
                pm = np.array(Image.fromarray(pm.astype(np.uint8) * 255).resize(
                    (gt.shape[1], gt.shape[0]))) > 0
            a = int(pm.sum())
            if a == 0:
                continue
            pred_area[lab] += a
            gv = gt[pm]                      # pred 영역의 GT id들
            gids, gcnts = np.unique(gv, return_counts=True)
            for gid, c in zip(gids, gcnts):
                if gid in ignore:
                    continue
                inter[lab][int(gid)] += int(c)

    # 3) 매칭 + 지표
    rows = []          # (label, best_gt, iou, purity, n_sig_gt)
    matched_iou = []
    gt_to_preds = defaultdict(list)   # gt -> [labels matched]
    for lab in pred:
        pa = pred_area[lab]
        if pa == 0:
            continue
        # 최고 IoU GT
        best_gt, best_iou, best_inter = -1, 0.0, 0
        n_sig = 0
        for gid, ic in inter[lab].items():
            iou = ic / (pa + gt_area[gid] - ic + 1e-9)
            if iou >= args.sig_overlap:
                n_sig += 1
            if iou > best_iou:
                best_iou, best_gt, best_inter = iou, gid, ic
        purity = (max(inter[lab].values()) / pa) if inter[lab] else 0.0
        rows.append((lab, best_gt, best_iou, purity, n_sig))
        if best_gt != -1:
            matched_iou.append(best_iou)
            gt_to_preds[best_gt].append(lab)

    n_pred = len(rows)
    n_gt = len([g for g in gt_area if gt_area[g] > 0])

    # fragmentation: GT당 매칭 pred 수
    frag = np.mean([len(v) for v in gt_to_preds.values()]) if gt_to_preds else 0.0
    # merge_error: pred당 유의 겹침 GT 수
    merge = np.mean([r[4] for r in rows]) if rows else 0.0
    purity = np.mean([r[3] for r in rows]) if rows else 0.0
    miou = np.mean(matched_iou) if matched_iou else 0.0

    # PQ: IoU>iou_match 매칭을 TP (각 GT는 1개 pred에만)
    tp_pairs = {}
    for lab, gid, iou, _, _ in rows:
        if gid != -1 and iou >= args.iou_match:
            if gid not in tp_pairs or iou > tp_pairs[gid][1]:
                tp_pairs[gid] = (lab, iou)
    TP = len(tp_pairs)
    FP = n_pred - TP
    FN = n_gt - TP
    SQ = np.mean([v[1] for v in tp_pairs.values()]) if TP > 0 else 0.0
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN + 1e-9)
    PQ = SQ * RQ

    print("\n================ Axis 1 평가 결과 ================")
    print(f"pred 라벨 수            : {n_pred}")
    print(f"GT 인스턴스 수          : {n_gt}")
    print(f"mIoU (best-match)       : {miou:.4f}")
    print(f"PQ                      : {PQ:.4f}  (SQ {SQ:.4f} × RQ {RQ:.4f}, TP {TP}/FP {FP}/FN {FN})")
    print(f"fragmentation (GT당 pred): {frag:.3f}   (1=이상, ↑=과분할)")
    print(f"merge_error  (pred당 GT) : {merge:.3f}   (1=이상, ↑=용접)")
    print(f"purity                  : {purity:.4f}  (뷰 간 일관성, 1=완벽)")
    print("=================================================")

    if args.report:
        import csv
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pred_label", "best_gt", "iou", "purity", "n_sig_gt"])
            for r in sorted(rows, key=lambda x: -x[2]):
                w.writerow(r)
        print(f"per-label 표 → {args.report}")


if __name__ == "__main__":
    main()
