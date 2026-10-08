#
# RefineGS - arguments/__init__.py
# ---------------------------------------------------------------------------
# Merge: a 2DGS base with the Split&Splat layer grafted on.
#   - [2DGS]    depth_ratio, lambda_dist, lambda_normal, opacity_cull, render_items
#   - [S&S]     composition, is_instance, depths, init_rec, train_test_exp,
#               optimizer_type, random_background
#   - [removed] exposure_lr_*, depth_l1_weight_* (inverse depth), antialiasing, max_num_splats
#
# Each field is tagged with where it comes from: [2DGS] or [S&S]. Untagged fields are common
# to both, from the shared 3DGS lineage.
# ---------------------------------------------------------------------------

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""                 # [S&S] depth directory
        self._resolution = -1
        self.is_instance = False          # [S&S] whether to mask the views
        self._composition = False         # [S&S] whether this is a composition model
        self.init_rec = False             # [S&S] apply mask dilation
        self._white_background = False
        self.train_test_exp = False       # [S&S] split the image left/right, for exposure evaluation
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']  # [2DGS] mesh render items
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0            # [2DGS] 0 = expected depth (smoother), 1 = median (sharper planes)
        self.debug = False
        # [removed] antialiasing -- specific to the 3DGS accelerated path, unused by the surfel rasteriser
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05            # [2DGS] default 0.05 (S&S tuned 0.025 for per-object)
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0            # [2DGS] depth distortion (LERF indoor: 100 advised)
        self.lambda_normal = 0.05         # [2DGS] normal consistency
        self.opacity_cull = 0.05          # [2DGS] densify_and_prune min_opacity

        self.densification_interval = 100
        self.opacity_reset_interval = 3000   # [2DGS] default 3000 (S&S tuned 1000 for per-object)
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000     # [2DGS] default 15000 (S&S tuned 10000 for per-object)
        self.densify_grad_threshold = 0.0002

        self.random_background = False    # [S&S]
        self.optimizer_type = "default"   # [S&S] sparse_adam is unused -- always default
        # [removed] exposure_lr_*       the exposure subsystem is gone
        # [removed] depth_l1_weight_*   inverse-depth supervision is gone; depth is supervised
        #                               directly instead
        # [removed] max_num_splats      the densify cap in train.py is gone
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)