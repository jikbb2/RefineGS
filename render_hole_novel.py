#!/usr/bin/env python3
"""RefineGS — novel pose 에서 hole 렌더 + reachability 진단 (S2+S3).

학습 카메라가 아니라 *novel pose* 에서 label-buffer(색 교체 2차 패스)로 hole 을 그린다.
목적: '미관측이지만 도달가능(reachable)' 표면(객체 윗면·옆면 등)이 어떤 novel pose 에서
보이는지 확인 = See3D 가 실제로 supervise 할 수 있는 영역인지 진단.

벽-인접 prior-bound 영역(예: cabinet 뒷면)은 어떤 novel pose 에서도 front-most 로 안 떠야 정상.
free-standing 객체라면 옆/아랫면 hole 이 일부 novel pose 에서 보여야 함.

pose 소스:
  --path generate : render.py 와 동일한 generate_path(학습카메라→smooth novel 궤적). 안전, 1순위.
  --path orbit    : 객체 중심 orbit(실험적; world_view 직접 구성). generate 가 커버 부족할 때.

색 교체 수식 등은 render_hole_masks.py 와 동일.

실행(1순위):
  python render_hole_novel.py \
    -m output/replica_room0_v2/scene_b1_<gid> -s data/replica_room0_v2 \
    --iteration 1 --hole_npy ~/tmp/hole_label.npy \
    --path generate --n_frames 120 --max_views 24 \
    --thr 0.5 --dilate 3 --out_dir output/replica_room0_v2/scene_b1_<gid>/holes_novel

Deps: torch, torchvision, numpy, (선택)scipy + Split&Splat 코드베이스.
"""
import os
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

C0 = 0.28209479177387814


def dilate_mask(m, k):
    if k <= 0:
        return m
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(m, iterations=int(k))
    except Exception:
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        t = torch.nn.functional.max_pool2d(t, 2*k+1, 1, k)
        return (t[0, 0].numpy() > 0.5)


def freespace_filter(cams, center, radius, occluder_path, slack=1.5):
    """카메라→객체중심 사이를 occluder(벽)가 가로막으면 reject → free-space pose 만 남김.
    occluder_path: holistic base mesh(벽 포함, 예 fuse_cropped.ply).
    valid 조건: 첫 hit 가 객체 bounding-sphere 안(t_hit > dist - radius*slack).
               벽이 앞을 막으면 t_hit 가 작아 reject(=prior-bound 뒷면 카메라 제거)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(occluder_path)
    if len(mesh.triangles) == 0:
        print(f"[warn] occluder mesh 비어있음 {occluder_path} → 필터 skip"); return cams, None
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ctr = np.asarray(center, np.float32)
    kept, mask = [], []
    for c in cams:
        pos = c.camera_center.detach().cpu().numpy().astype(np.float32)
        d = ctr - pos; dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            mask.append(False); continue
        dir = d / dist
        ray = o3d.core.Tensor([[*pos, *dir]], dtype=o3d.core.Dtype.Float32)
        t_hit = float(rc.cast_rays(ray)["t_hit"].numpy()[0])
        ok = (not np.isfinite(t_hit)) or (t_hit > dist - radius * slack)
        mask.append(ok)
        if ok:
            kept.append(c)
    return kept, np.array(mask)


def obs_cone_filter(cams, train_cams, center, cone_deg):
    """관측 hemisphere 제약: novel 카메라의 (center→cam) 방향이 학습카메라 방향들과
    cone_deg 이내일 때만 유효. flush 벽 뒤처럼 '관측된 적 없는 방향' pose 를 제거.
    (벽 뒤는 어떤 학습카메라도 간 적 없음 → reject.)"""
    ctr = np.asarray(center, np.float64)
    tdir = []
    for c in train_cams:
        p = c.camera_center.detach().cpu().numpy().astype(np.float64)
        d = p - ctr; n = np.linalg.norm(d)
        if n > 1e-6:
            tdir.append(d / n)
    tdir = np.stack(tdir)                       # (T,3)
    cos_thr = np.cos(np.deg2rad(cone_deg))
    kept = []
    for c in cams:
        p = c.camera_center.detach().cpu().numpy().astype(np.float64)
        d = p - ctr; n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        if float((tdir @ (d / n)).max()) >= cos_thr:   # 가장 가까운 관측방향과의 각
            kept.append(c)
    return kept


def set_label_color(gaussians, label):
    fdc, frest = gaussians._features_dc, gaussians._features_rest
    saved = (fdc.detach().clone(), frest.detach().clone(), int(gaussians.active_sh_degree))
    dc = (label - 0.5) / C0
    with torch.no_grad():
        fdc[:, 0, 0] = dc; fdc[:, 0, 1] = dc; fdc[:, 0, 2] = dc
        frest.zero_()
    gaussians.active_sh_degree = 0
    return saved


def restore_color(gaussians, saved):
    fdc_s, frest_s, sh = saved
    with torch.no_grad():
        gaussians._features_dc.copy_(fdc_s)
        gaussians._features_rest.copy_(frest_s)
    gaussians.active_sh_degree = sh


def get_novel_cams(scene, mode, n_frames, center=None, radius_scale=1.0,
                   elev_min=0.0, elev_max=60.0, azim_min=0.0, azim_max=360.0, up_axis=1):
    """novel 카메라 목록.
    generate : 학습카메라 보간(관측 manifold 근처 — 새 각도 못 봄).
    orbit    : 객체 중심(center)을 바라보는 합성 pose. elev/azim/radius 로 시점 제어.
               center=객체 중심(보통 hole 점 centroid). up_axis: 월드 up 축(0=x,1=y,2=z).
    """
    train = scene.getTrainCameras()
    if mode == "generate":
        from utils.render_utils import generate_path
        return list(generate_path(train, n_frames=n_frames))
    elif mode == "orbit":
        import copy
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        cam_centers = np.stack([c.camera_center.detach().cpu().numpy() for c in train])
        ctr = np.asarray(center, np.float64) if center is not None else cam_centers.mean(0)
        # 반경 = 학습카메라들의 객체중심까지 평균 거리 × scale (현실적 관측거리)
        rad = float(np.linalg.norm(cam_centers - ctr, axis=1).mean()) * radius_scale
        up = np.zeros(3); up[up_axis] = 1.0
        base = train[0]
        proj = getProjectionMatrix(base.znear, base.zfar, base.FoVx, base.FoVy).transpose(0, 1).cuda()
        out = []
        for i in range(n_frames):
            f = i / max(n_frames - 1, 1)
            az = np.deg2rad(azim_min + (azim_max - azim_min) * f)
            el = np.deg2rad(elev_min + (elev_max - elev_min) * f)
            # 객체 중심 기준 카메라 위치 (up_axis 를 고도축으로)
            horiz = np.cos(el)
            offs = np.zeros(3)
            a = [j for j in range(3) if j != up_axis]      # 수평면 두 축
            offs[a[0]] = horiz * np.cos(az)
            offs[a[1]] = horiz * np.sin(az)
            offs[up_axis] = np.sin(el)
            cam_pos = ctr + rad * offs
            # look-at: 카메라가 ctr 를 바라보게 (Rc2w 의 열 = 카메라 축의 월드표현)
            fwd = ctr - cam_pos; fwd /= (np.linalg.norm(fwd) + 1e-9)
            right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
            down = np.cross(fwd, right)                      # y_cam = down (이미지 y 아래)
            Rc2w = np.stack([right, down, fwd], axis=1)      # 열: x_cam,y_cam,z_cam(월드)
            t = -Rc2w.T @ cam_pos                            # W2C translation
            # getWorld2View2 는 인자 R 을 전치해 W2C 회전으로 씀 → R=Rc2w 전달
            w2v = torch.tensor(getWorld2View2(Rc2w, t)).transpose(0, 1).float().cuda()
            cam = copy.copy(base)
            cam.world_view_transform = w2v
            cam.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            cam.camera_center = w2v.inverse()[3, :3]
            cam.image_name = f"orbit{i:04d}"
            out.append(cam)
        return out
    raise SystemExit(f"unknown path mode {mode}")


if __name__ == "__main__":
    parser = ArgumentParser(description="novel-pose hole render + reachability 진단")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--hole_npy", required=True)
    parser.add_argument("--path", default="generate", choices=["generate", "orbit"])
    parser.add_argument("--n_frames", default=120, type=int)
    parser.add_argument("--max_views", default=24, type=int, help="저장할 뷰 수(균등 subsample)")
    parser.add_argument("--thr", default=0.5, type=float)
    parser.add_argument("--dilate", default=0, type=int)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--soft_out", default="",
                        help="[권장] soft-weight 파이프라인 출력 디렉토리. valid novel pose마다 "
                             "view_<i>.jpg(조건용 RGB 렌더) + weight_<i>.png(연속 soft weight, 흰=미관측/refine) "
                             "저장 + poses.npz(카메라). 이진 hole 안 씀. 생성→soft-weighted 학습 주입 입력.")
    parser.add_argument("--see3d_out", default="",
                        help="[구식] 이진 hole 마스크 출력(warp+mask). soft_out 으로 대체됨.")
    parser.add_argument("--min_weight", default=0.01, type=float,
                        help="soft weight 평균이 이보다 작은 pose는 저장 skip(refine 거리 없는 뷰 제거)")
    parser.add_argument("--min_hole_frac", default=0.003, type=float,
                        help="see3d_out(구식) 전용 threshold")
    parser.add_argument("--quiet", action="store_true")
    # --- orbit 시점 제어 ---
    parser.add_argument("--elev_min", default=0.0, type=float, help="orbit 고도 시작(도). 음수=아래에서 위로(테이블 아랫면)")
    parser.add_argument("--elev_max", default=60.0, type=float, help="orbit 고도 끝(도). 양수=위에서 내려보기(윗면)")
    parser.add_argument("--azim_min", default=0.0, type=float)
    parser.add_argument("--azim_max", default=360.0, type=float)
    parser.add_argument("--radius_scale", default=1.0, type=float, help="관측거리 배율(작게=가까이)")
    parser.add_argument("--up_axis", default=1, type=int, help="월드 up 축 0=x/1=y/2=z (고도축)")
    parser.add_argument("--center", default="", help="orbit 중심 'x,y,z'. 비우면 hole 점 centroid 자동")
    parser.add_argument("--occluder_mesh", default="",
                        help="free-space 필터용 base mesh(벽 포함, 예 .../fuse_cropped.ply). "
                             "지정 시 카메라→객체 사이를 벽이 막는 pose reject → prior-bound 뒷면 제거")
    parser.add_argument("--obs_cone_deg", default=0.0, type=float,
                        help=">0 이면 관측 hemisphere 제약: (center→cam) 방향이 학습카메라 방향과 "
                             "이 각도(도) 이내인 pose 만 유효. flush 벽 뒤 pose 제거(예: 50)")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    label_np = np.load(args.hole_npy).astype(np.float32)
    n = gaussians.get_xyz.shape[0]
    if len(label_np) != n:
        raise SystemExit(f"hole_npy {len(label_np)} != gaussians {n}")
    label = torch.from_numpy(label_np).cuda()

    # orbit 중심/반경: hole(=미관측 gen) 점들의 월드 centroid + bounding 반경
    center, obj_radius = None, 0.3
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    hole_pts = xyz[label_np > 0.5]
    if args.center.strip():
        center = np.array([float(x) for x in args.center.split(",")])
    elif len(hole_pts) > 0:
        center = hole_pts.mean(0)
    if len(hole_pts) > 0 and center is not None:
        obj_radius = float(np.percentile(np.linalg.norm(hole_pts - center, axis=1), 95))
        print(f"orbit center: {np.round(center,3).tolist()}  obj_radius(95%): {obj_radius:.3f}  "
              f"({len(hole_pts)} hole pts)")

    cams = get_novel_cams(scene, args.path, args.n_frames, center=center,
                          radius_scale=args.radius_scale,
                          elev_min=args.elev_min, elev_max=args.elev_max,
                          azim_min=args.azim_min, azim_max=args.azim_max,
                          up_axis=args.up_axis)

    # free-space 필터: 벽이 카메라→객체를 막는 pose reject (큰 occluder/타객체)
    if args.occluder_mesh.strip() and center is not None:
        before = len(cams)
        cams, _ = freespace_filter(cams, center, obj_radius, args.occluder_mesh)
        print(f"free-space 필터: {len(cams)}/{before} pose 유효(벽 안 막힘).")
        if len(cams) == 0:
            raise SystemExit("유효 pose 0 — radius_scale 키우거나 elev 범위/occluder mesh 확인.")

    # 관측 hemisphere 필터: flush 벽 뒤처럼 관측된 적 없는 방향 pose reject
    if args.obs_cone_deg > 0 and center is not None:
        before = len(cams)
        cams = obs_cone_filter(cams, scene.getTrainCameras(), center, args.obs_cone_deg)
        print(f"obs-cone 필터(±{args.obs_cone_deg}°): {len(cams)}/{before} pose 유효(관측방향 근처). "
              f"reject = 관측된 적 없는 방향(예: 벽-인접 뒷면).")
        if len(cams) == 0:
            raise SystemExit("유효 pose 0 — obs_cone_deg 키우거나 객체가 prior-bound 인지 확인.")

    if args.max_views > 0 and len(cams) > args.max_views:
        idx = np.linspace(0, len(cams) - 1, args.max_views).astype(int)
        cams = [cams[i] for i in idx]

    os.makedirs(args.out_dir, exist_ok=True)
    see3d_dir = args.see3d_out.strip()
    if see3d_dir:
        os.makedirs(see3d_dir, exist_ok=True)
    soft_dir = args.soft_out.strip()
    if soft_dir:
        os.makedirs(soft_dir, exist_ok=True)
    print(f"path={args.path} novel views={len(cams)}  per-Gaussian hole frac={label.mean().item():.3f}")
    fracs = []
    n_saved = 0
    pose_records = []   # soft_out poses.npz 용
    for i, cam in enumerate(cams):
        stem = getattr(cam, "image_name", f"nv{i:04d}"); stem = os.path.splitext(stem)[0]
        with torch.no_grad():
            rgb = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        saved = set_label_color(gaussians, label)
        with torch.no_grad():
            lab = render(cam, gaussians, pipe, background)["render"][0].clamp(0, 1)
        restore_color(gaussians, saved)

        lab_np = lab.detach().cpu().numpy()          # 연속 soft weight (0~1)
        soft = lab.clamp(0, 1)                        # GPU tensor
        wmean = float(lab_np.mean())
        fracs.append(wmean)

        # QA: soft weight 자체 + 그 weight 강도로 red 오버레이(연속)
        torchvision.utils.save_image(soft.unsqueeze(0), os.path.join(args.out_dir, f"{stem}_weight.png"))
        ov = rgb.clone()
        ov[0] = torch.maximum(ov[0], soft); ov[1] = ov[1]*(1-0.5*soft); ov[2] = ov[2]*(1-0.5*soft)
        torchvision.utils.save_image(ov, os.path.join(args.out_dir, f"{stem}_overlay.png"))

        # [권장] soft-weight 출력: 조건용 view + 연속 weight + pose 기록
        if soft_dir and wmean >= args.min_weight:
            torchvision.utils.save_image(rgb, os.path.join(soft_dir, f"view_{n_saved:04d}.jpg"))
            torchvision.utils.save_image(soft.unsqueeze(0), os.path.join(soft_dir, f"weight_{n_saved:04d}.png"))
            pose_records.append(dict(
                idx=n_saved, stem=stem,
                world_view_transform=cam.world_view_transform.detach().cpu().numpy(),
                full_proj_transform=cam.full_proj_transform.detach().cpu().numpy(),
                FoVx=float(cam.FoVx), FoVy=float(cam.FoVy),
                width=int(cam.image_width), height=int(cam.image_height)))
            n_saved += 1

        # [구식] 이진 See3D 입력(backward-compat)
        if see3d_dir:
            hole = dilate_mask(lab_np >= args.thr, args.dilate)
            if float(hole.mean()) >= args.min_hole_frac:
                torchvision.utils.save_image(rgb, os.path.join(see3d_dir, f"warp_{n_saved:04d}.jpg"))
                known = torch.from_numpy((~hole).astype(np.float32))[None]
                torchvision.utils.save_image(known, os.path.join(see3d_dir, f"mask_{n_saved:04d}.png"))

        if i < 3 or (i+1) % 10 == 0:
            print(f"[{i+1}/{len(cams)}] {stem}: soft weight mean {wmean:.4f}")

    # soft_out poses.npz 저장 (학습 주입이 카메라 복원에 사용)
    if soft_dir and pose_records:
        np.savez(os.path.join(soft_dir, "poses.npz"),
                 records=np.array(pose_records, dtype=object))
        print(f"→ soft-weight 출력 {n_saved}뷰(view_*.jpg + weight_*.png) + poses.npz: {soft_dir}")

    fracs = np.array(fracs) if fracs else np.array([0.0])
    print(f"\nreachability 진단: soft weight mean {fracs.mean():.4f}  max {fracs.max():.4f}  "
          f"(weight>{args.min_weight} 뷰 {int((fracs>args.min_weight).sum())}/{len(fracs)})")
    print(f"→ QA: {args.out_dir} (*_weight / *_overlay)")
    print("해석: overlay 빨강 강도 = soft weight(미관측·gen↑). 옆/아랫면에 연속 weight가 실리면 refine 대상. "
          "weight 거의 0이면 그 방향엔 refine 거리 없음(관측 완전 or 도달불가).")
