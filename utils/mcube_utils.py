import numpy as np
import torch
import trimesh
from skimage import measure

def marching_cubes_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
):
    assert resolution % 512 == 0
    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]
                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()
                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                # Fix: use detach().clone() instead of torch.tensor()
                points = torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.detach().clone().float()

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.detach().cpu().numpy()
                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    verts = verts + np.array([x_min, y_min, z_min])
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

                print("finished one block")

    # Guard against empty mesh list
    if len(meshes) == 0:
        print("Warning: no surfaces found in any block!")
        return trimesh.Trimesh()

    combined = trimesh.util.concatenate(meshes)
    print(f"Combined mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    # merge_vertices can segfault on large meshes — skip if too large
    if len(combined.vertices) < 5_000_000:
        try:
            combined.merge_vertices(digits_vertex=6)
        except Exception as e:
            print(f"merge_vertices failed ({e}), skipping")
    else:
        print(f"Skipping merge_vertices (mesh too large: {len(combined.vertices)} verts)")

    # inverse contraction in chunks to avoid CUDA OOM
    if inv_contraction is not None:
        verts = torch.from_numpy(combined.vertices).float()
        chunk_size = 1_000_000
        out_verts = []
        for chunk in torch.split(verts, chunk_size):
            out_verts.append(inv_contraction(chunk.cuda()).cpu())
        combined.vertices = torch.cat(out_verts, dim=0).numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    return combined