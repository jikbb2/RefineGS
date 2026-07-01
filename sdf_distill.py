#!/usr/bin/env python3
"""RefineGS — 2DGS → implicit SDF distillation → watertight 메쉬 (TSDF 대체, C1).

2DGS Gaussian의 표면점+법선에 SDF MLP(IGR류)를 피팅 → zero-level-set marching cubes.
장점(TSDF 한계 극복): (a) SDF는 정의상 watertight, (b) MLP가 미관측 구멍을 매끄럽게 보간해 채움,
(c) 전역 연속장이라 국소 fusion floater·부호 불일치 없음.

입력: point_cloud.ply (refined whole-scene). x,y,z + rot(quat) → 법선(surfel 3번째 축) + opacity/scale(floater 제거) + f_dc(색).
손실: manifold(SDF=0) + normal(∇SDF=법선) + eikonal(|∇SDF|=1).

실행:
  python sdf_distill.py \
    --ply output/replica_room0_v2/scene_whole_orbit/point_cloud/iteration_7000/point_cloud.ply \
    --out output/replica_room0_v2/scene_whole_orbit/sdf_mesh.ply \
    --op_thr 0.3 --scale_thr 0.1 --n_pts 1500000 --iters 15000 --grid 512

Deps: torch, numpy, plyfile, scikit-image(marching_cubes), open3d(save/color KDTree via scipy).
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def quat_to_normal(q):
    """q:(N,4) wxyz(정규화) → 회전행렬 3번째 열(surfel 법선)."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R[:,2] = (2(xz+wy), 2(yz-wx), 1-2(xx+yy))
    n = np.stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)], 1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


class SDFNet(nn.Module):
    """IGR-style MLP (geometric init, skip connection). 표준 IGR 규약."""
    def __init__(self, d_hidden=256, n_layers=8, skip_in=(4,), pe_L=6, radius=0.5):
        super().__init__()
        self.pe_L = pe_L
        d_in = 3 + 3 * 2 * pe_L
        self.d_in = d_in
        dims = [d_in] + [d_hidden]*n_layers + [1]
        self.skip_in = set(skip_in)
        self.num_layers = len(dims)
        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            out_dim = dims[l+1] - d_in if (l+1) in self.skip_in else dims[l+1]
            lin = nn.Linear(dims[l], out_dim)
            if l == self.num_layers - 2:                       # 마지막: sphere 근사 init
                nn.init.normal_(lin.weight, mean=np.sqrt(np.pi)/np.sqrt(dims[l]), std=1e-4)
                nn.init.constant_(lin.bias, -radius)
            else:
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2)/np.sqrt(out_dim))
                nn.init.constant_(lin.bias, 0.0)
            self.layers.append(lin)
        self.act = nn.Softplus(beta=100)

    def pe(self, x):
        out = [x]
        for l in range(self.pe_L):
            for fn in (torch.sin, torch.cos):
                out.append(fn(2.0**l * np.pi * x))
        return torch.cat(out, -1)

    def forward(self, x):
        inp = self.pe(x)
        h = inp
        for l, lin in enumerate(self.layers):
            if l in self.skip_in:
                h = torch.cat([h, inp], -1) / np.sqrt(2)
            h = lin(h)
            if l < self.num_layers - 2:
                h = self.act(h)
        return h


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--op_thr", type=float, default=0.3, help="이하 opacity 제거(floater)")
    ap.add_argument("--scale_thr", type=float, default=0.1, help="이상 scale 제거(floater)")
    ap.add_argument("--n_pts", type=int, default=1500000, help="표면점 subsample")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_normal", type=float, default=1.0)
    ap.add_argument("--w_eik", type=float, default=0.1)
    ap.add_argument("--grid", type=int, default=512, help="marching cubes 그리드 해상도")
    args = ap.parse_args()

    v = PlyData.read(args.ply)["vertex"]; nm = v.data.dtype.names
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    op = sigmoid(np.asarray(v["opacity"]).astype(np.float64))
    sc = np.exp(np.column_stack([v["scale_0"], v["scale_1"]]).astype(np.float64)).max(1)
    quat = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).astype(np.float64)
    normals = quat_to_normal(quat)
    C0 = 0.28209479177387814
    rgb = np.clip(np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float64)*C0 + 0.5, 0, 1)

    keep = (op > args.op_thr) & (sc < args.scale_thr)
    xyz, normals, rgb = xyz[keep], normals[keep], rgb[keep]
    print(f"표면점 {len(xyz)} (floater 제거 후)")
    if len(xyz) > args.n_pts:
        idx = np.random.choice(len(xyz), args.n_pts, replace=False)
        xyz, normals, rgb = xyz[idx], normals[idx], rgb[idx]

    # 정규화 [-1,1] (여유 1.1)
    center = xyz.mean(0); scale = (np.abs(xyz - center).max()) * 1.1
    P = (xyz - center) / scale
    N = normals  # 방향 불변(스케일만)

    dev = "cuda"
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    Nt = torch.tensor(N, dtype=torch.float32, device=dev)
    net = SDFNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for it in range(args.iters):
        bi = torch.randint(0, len(Pt), (args.batch,), device=dev)
        pts = Pt[bi].clone().requires_grad_(True)
        nrm = Nt[bi]
        sdf = net(pts)
        g = grad(sdf, pts)
        # manifold: SDF=0, normal: grad=normal
        l_man = sdf.abs().mean()
        l_nrm = (1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)).mean()
        # eikonal: 표면 근처 + 균등 랜덤
        rp = torch.cat([Pt[torch.randint(0, len(Pt), (args.batch,), device=dev)]
                        + 0.02*torch.randn(args.batch, 3, device=dev),
                        torch.rand(args.batch, 3, device=dev)*2-1], 0).requires_grad_(True)
        ge = grad(net(rp), rp)
        l_eik = ((ge.norm(dim=-1) - 1)**2).mean()
        loss = l_man + args.w_normal*l_nrm + args.w_eik*l_eik
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"[{it}] man {l_man.item():.4f} nrm {l_nrm.item():.4f} eik {l_eik.item():.4f}")

    # marching cubes on grid
    print("SDF 그리드 평가 + marching cubes ...")
    G = args.grid
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    net.eval()
    vol = np.empty((G, G, G), np.float32)
    with torch.no_grad():
        gx, gy = np.meshgrid(lin, lin, indexing="ij")
        for k in range(G):
            pts = np.stack([gx, gy, np.full_like(gx, lin[k])], -1).reshape(-1, 3)
            s = net(torch.tensor(pts, dtype=torch.float32, device=dev)).cpu().numpy().reshape(G, G)
            vol[:, :, k] = s
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(2.0/(G-1),)*3)
    verts = verts - 1.0                      # [0,2]→[-1,1]
    verts = verts * scale + center           # 원좌표 복원

    # 색: 최근접 표면점
    from scipy.spatial import cKDTree
    _, ni = cKDTree(xyz).query(verts)
    vcol = (rgb[ni]*255).astype(np.uint8)

    dt = np.dtype([("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
    va = np.empty(len(verts), dt)
    va["x"],va["y"],va["z"] = verts[:,0],verts[:,1],verts[:,2]
    va["red"],va["green"],va["blue"] = vcol[:,0],vcol[:,1],vcol[:,2]
    fa = np.empty(len(faces), dtype=[("vertex_indices","i4",(3,))])
    fa["vertex_indices"] = faces
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(va,"vertex"), PlyElement.describe(fa,"face")], text=False).write(args.out)
    print(f"→ {args.out}  verts {len(verts)} faces {len(faces)} (watertight, 미관측 보간 채움)")


if __name__ == "__main__":
    main()
