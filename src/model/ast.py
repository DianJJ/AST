from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from toolz import valfilter

import numpy as np
from pytorch3d.loss import mesh_laplacian_smoothing as laplacian_smoothing
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, TexturesUV
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Compose, Resize, functional as Fvision, InterpolationMode
from .transformer import Transformer

from .encoder import Encoder, Encoder_16, GlobalEncoder
from .field import ProgressiveField, TextureField
from .generator import ProgressiveGenerator, SaliencyHead, SaliencyHead_16
from .loss import get_loss
from .position_encoding import PositionEmbeddingSine
from .renderer import Renderer, save_mesh_as_gif
from .tools import create_mlp, init_rotations, convert_3d_to_uv_coordinates, safe_model_state_dict, N_UNITS, N_LAYERS
from .tools import azim_to_rotation_matrix, elev_to_rotation_matrix, roll_to_rotation_matrix, cpu_angle_between
from utils import path_mkdir, use_seed
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.mesh import save_mesh_as_obj, repeat, get_icosphere, normal_consistency, normalize
from utils.metrics import MeshEvaluator, ProxyEvaluator
from utils.pytorch import torch_to
from utils.misc import NestedTensor

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# POSE & SCALE DEFAULT
N_POSES = 6
N_ELEV_AZIM = [1, 6]
SCALE_ELLIPSE = [1, 0.7, 0.7]
PRIOR_TRANSLATION = [0., 0., 2.732]

# NEIGHBOR REC LOSS DEFAULT (previously called swap loss)
MIN_ANGLE = 20
N_VPBINS = 5
MEMSIZE = 1024

HIDDEN_DIM = 128
DROPOUT = 0.1
NHEADS =4
DIM_FEEDFORWARD = 128
ENC_LAYERS = 1
DEC_LAYERS = 1
PRE_NORM = False
DEEP_SUPERVISION = False
NUMBER_OF_QUERIES = 642
NEED_ENCODER=False

class AST(nn.Module):
    name = 'ast'

    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.init_kwargs = deepcopy(kwargs)
        self.init_kwargs['img_size'] = img_size
        self._init_encoder(img_size, **kwargs.get('encoder', {}))
        self._init_transformer(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, nheads=NHEADS, dim_feedforward=DIM_FEEDFORWARD,
                          enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, pre_norm=PRE_NORM, deep_supvision=DEEP_SUPERVISION)
        self._init_queries(number_of_queries=NUMBER_OF_QUERIES, hidden_dim=HIDDEN_DIM)
        self._init_meshes(**kwargs.get('mesh', {}))
        self.renderer = Renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_saliency_model(img_size, **kwargs.get('saliency', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self._init_positional_encoding(HIDDEN_DIM // 2)
        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self.eval_mode = kwargs.pop('eval_mode', False)
        self._debug = False

    @property
    def n_features(self):
        return self.encoder.out_ch if self.shared_encoder else self.encoder.out_ch

    @property
    def tx_code_size(self):
        return self.txt_field.current_code_size

    @property
    def sh_code_size(self):
        return self.deform_field.current_code_size

    def _init_encoder(self, img_size, **kwargs):
        self.shared_encoder = kwargs.pop('shared', True)
        self.encoder_sz = kwargs.pop('size', 'large')
        if self.shared_encoder:
            self.encoder = Encoder(**kwargs)
            self.global_encoder = GlobalEncoder(img_size,**kwargs)
        else:
            self.encoder = Encoder(**kwargs)
            self.encoder_tx = Encoder_16(**kwargs)
            self.global_encoder = GlobalEncoder(img_size,**kwargs)

    def _init_transformer(self, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, nheads=NHEADS, dim_feedforward=DIM_FEEDFORWARD,
                          enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, pre_norm=PRE_NORM, deep_supvision=DEEP_SUPERVISION):
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supvision,
        )
        self.transformer_txt = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=4,
            dim_feedforward=128,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supvision,
        )
    def _init_queries(self, number_of_queries=NUMBER_OF_QUERIES, hidden_dim=HIDDEN_DIM):

        self.query_embed = nn.Embedding(number_of_queries, hidden_dim)
        self.tx_query_embed = nn.Embedding(1024, hidden_dim)

            #self.global_encoder = GlobleEncoder(img_size, **kwargs)

    def _init_positional_encoding(self, num_pos_feats=HIDDEN_DIM // 2):

        self.positional_encod = PositionEmbeddingSine(num_pos_feats, normalize=True)


    def _init_meshes(self, **kwargs):
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init', 'sphere')
        scale = kwargs.pop('scale', 1)
        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(4 if 'hr' in mesh_init else 3)
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_src.get_mesh_verts_faces(0)[0])[None])

        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))  # retrocompatibility
        dfield_kwargs = kwargs.pop('deform_fields', {})
        tgen_kwargs = kwargs.pop('texture_uv', {})
        tgen_coarse_kwargs = kwargs.pop('texture_uv_coarse', {})
        assert len(kwargs) == 0

        self.deform_field = ProgressiveField(inp_dim=self.n_features, name='deformation', **dfield_kwargs)
        self.deform_field_global = ProgressiveField(inp_dim=self.n_features, name='deformation2', **dfield_kwargs)
        self.txt_generator = ProgressiveGenerator(inp_dim=self.n_features, **tgen_coarse_kwargs)
        self.txt_field = TextureField(inp_dim=self.n_features, name='texture', **tgen_kwargs)


    def _init_rend_predictors(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.pose_step = True

        NF, NP = self.n_features, self.n_poses
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)

        # Translation
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))

        # Rotation
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')
        azim, elev, roll = [(e[1] - e[0]) / n for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        # In practice we extend the range a bit to allow overlap in case of multiple candidates
        if self.n_poses == 1:
            self.register_buffer('R_range', torch.Tensor([azim * 0.5, elev * 0.5, roll * 0.5]))
        else:
            self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range

        # Scale
        self.scale_regressor = create_mlp(NF, 3, NU, NL, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))

        # Pose probabilities
        if NP > 1:
            self.proba_regressor = create_mlp(NF, NP, NU, NL)

        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        return 1 if self.hard_select else self.n_poses

    @property
    def hard_select(self):
        if self.alternate_optim and not self._debug:
            return False if (self.training and self.pose_step) else True
        else:
            return False

    def _init_saliency_model(self, img_size, out=1,**kwargs):
        if len(kwargs) > 0:
            scly_kwargs = deepcopy(kwargs)
            self.slcy_generator = SaliencyHead()

    def _init_background_model(self, img_size, out=3,**kwargs):
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGenerator(inp_dim=self.n_features, img_size=img_size, out_dim=out, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.milestones = {
            'constant_txt': kwargs.pop('constant_txt', kwargs.pop('contant_text', 0)),  # retrocompatibility
            'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0),
            'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0),
            'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0),
            'freeze_shape': kwargs.pop('freeze_shape', 0),
            'coarse_shape': kwargs.pop('coarse_shape', True),
            'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt)),  # retrocompatibility
        }

        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        kwargs = deepcopy(kwargs)
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'bce_iou': kwargs.pop('bce_iou_weight', 0),
            'normal': kwargs.pop('normal_weight', 0),
            'laplacian': kwargs.pop('laplacian_weight', 0),
            'normal_refine': kwargs.pop('normal_weight_refine', 0),
            'laplacian_refine': kwargs.pop('laplacian_weight_refine', 0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'uniform': kwargs.pop('uniform_weight', 0),
            'neighbor': kwargs.pop('neighbor_weight', kwargs.pop('swap_weight', 0)),  # retrocompatibility
        }
        name = kwargs.pop('name', 'mse')
        perceptual_kwargs = kwargs.pop('perceptual', {})
        bce_iou_kwargs = kwargs.pop('bce_iou', {})
        self.nbr_memsize = kwargs.pop('nbr_memsize', kwargs.pop('swap_memsize', MEMSIZE))  # retro
        self.nbr_n_vpbins = kwargs.pop('nbr_n_vpbins', kwargs.pop('swap_n_vpbins', N_VPBINS))  # retro
        self.nbr_min_angle = kwargs.pop('nbr_min_angle', kwargs.pop('swap_min_angle', MIN_ANGLE))  # retro
        self.nbr_memory = {k: torch.empty(0) for k in ['sh', 'tx', 'S', 'R', 'T', 'sy', 'bg', 'img','mask_gt']}
        kwargs.pop('swap_equal_bins', False)  # retro
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')
        self.bce_iou_loss = get_loss('bce_iou')(**bce_iou_kwargs)
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

    @property
    def pred_saliency(self):
        return hasattr(self, 'slcy_generator')

    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        super().to(device)
        self.mesh_src = self.mesh_src.to(device)
        self.renderer = self.renderer.to(device)
        return self

    def forward(self, inp, debug=False):
        # XXX pytorch3d objects are not well handled by DDP so we need to manually move them to GPU
        # self.mesh_src, self.renderer = [t.to(inp['imgs'].device) for t in [self.mesh_src, self.renderer]]
        self._debug = debug
        imgs, K, B = inp['imgs'], self.n_candidates, len(inp['imgs'])
        mask_gt = inp['masks']
        perturbed = self.training and np.random.binomial(1, p=0.2)
        average_txt = self.is_live('constant_txt') or (perturbed and self.use_mean_txt and self.is_live('mean_txt'))
        meshes, meshes_coarse, (R, T), slcys, bkgs = self.predict_mesh_pose_slcy_bkg(imgs, average_txt)
        if self.alternate_optim:
            if self.pose_step:
                meshes, meshes_coarse, bkgs = meshes.detach(), meshes_coarse.detach() if self.is_live('coarse_shape') else None\
                    , bkgs.detach() if self.pred_background else None
            else:
                R, T = R.detach(), T.detach()
        #coarse meshes
        if self.is_live('coarse_shape'):
            meshes_coarse_to_render = repeat(meshes_coarse, len(T) // len(meshes_coarse))
            fgs_coarse, alpha_coarse = self.renderer(meshes_coarse_to_render, R, T).split([3, 1], dim=1)  # (K*B)CHW
            rec_img_coarse = fgs_coarse * alpha_coarse + (1 - alpha_coarse) * bkgs if self.pred_background else fgs_coarse
        else:
            alpha_coarse, rec_img_coarse = None, None
        meshes_to_render = repeat(meshes, len(T) // len(meshes))
        fgs, alpha = self.renderer(meshes_to_render, R, T).split([3, 1], dim=1)  # (K*B)CHW
        rec_img = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        losses, select_idx = self.compute_losses(meshes,  imgs, alpha, slcys, rec, rec_img,
                                                 R, T, meshes_coarse, alpha_coarse, rec_img_coarse, average_txt=average_txt, mask_gt=mask_gt)
        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
            self._debug = False
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec
        return out

    def predict_mesh_pose_slcy_bkg(self, imgs, average_txt=False):
        if self.shared_encoder:
            global_features = self.global_encoder(imgs)
            local_features = self.encoder(imgs)
            slcys, slcy = self.predict_saliency(local_features)
            hard_slcy=torch.where(slcy > 0.5, 1, 0).detach()
            tensor = torch.empty(local_features.shape[0], 1,64, 64)
            mask = torch.ones_like(tensor, device=imgs.device)
            mask = (mask == 0)
            fg_features_local = local_features
            positional_enc=self.positional_encod(NestedTensor(fg_features_local,mask))
            if self.is_live('coarse_shape'):
                meshes_coarse = self.predict_meshes_coarse(fg_features_local, global_features, mask,positional_enc,
                                             average_txt=average_txt)
            else:
                meshes_coarse = None

            meshes = self.predict_meshes(meshes_coarse, fg_features_local, global_features, mask, positional_enc,
                                                       features_tx=local_features,average_txt=average_txt)
            R, T = self.predict_poses(global_features)
            bkgs = self.predict_background(global_features) if self.pred_background else None
        else:
            global_features = self.global_encoder(imgs)
            local_features = self.encoder(imgs)
            local_features_tx = self.encoder_tx(imgs)
            slcys, slcy = self.predict_saliency(local_features)
            hard_slcy=torch.where(slcy > 0.5, 1, 0).detach()
            tensor = torch.empty(local_features.shape[0], 1,64, 64)
            mask = torch.ones_like(tensor, device=imgs.device)
            mask = (mask == 0)
            fg_features_local = local_features
            positional_enc=self.positional_encod(NestedTensor(fg_features_local,mask))
            positional_enc_tx = self.positional_encod(NestedTensor(local_features_tx, mask))
            if self.is_live('coarse_shape'):
                meshes_coarse = self.predict_meshes_coarse(fg_features_local, global_features, mask,positional_enc,positional_enc_tx,
                                             features_tx=local_features_tx,average_txt=average_txt)
            else:
                meshes_coarse = None

            meshes = self.predict_meshes(meshes_coarse, fg_features_local, global_features, mask, positional_enc,positional_enc_tx=positional_enc_tx,
                                                       features_tx=local_features_tx, average_txt=average_txt)
            with torch.no_grad():
                R, T = self.predict_poses(global_features)
            bkgs = self.predict_background(global_features) if self.pred_background else None
        return meshes, meshes_coarse, (R, T), slcys, bkgs


    def predict_meshes_coarse(self, features_local,features, mask, positional_enc, positional_enc_tx=None,features_tx=None, average_txt=False):
        if features_tx is None:
            features_tx = features_local
        if positional_enc_tx is None:
            positional_enc_tx=positional_enc
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        meshes = self.mesh_src.extend(len(features))  # XXX does a copy
        meshes.offset_verts_(self.predict_disp_verts_coarse(verts, features))
        return meshes

    def predict_meshes(self, meshes, features_local, features, mask, positional_enc, positional_enc_tx=None,features_tx=None, average_txt=False):
        if features_tx is None:
            features_tx = features
        if meshes is None:
            verts, faces = self.mesh_src.get_mesh_verts_faces(0)
            meshes = self.mesh_src.extend(len(features))
        else:
            verts, faces = meshes.get_mesh_verts_faces(0)
        with torch.no_grad():
            meshes.offset_verts_(self.predict_disp_verts(verts, features_local, mask,positional_enc,NEED_ENCODER))
        if self.is_live('coarse_shape'):
            meshes.textures = self.predict_textures(faces, features_tx, mask, positional_enc_tx,
                                                    average_txt=average_txt)
        else:
            meshes.textures = self.predict_textures(faces, features_tx, mask, positional_enc_tx,
                                                    average_txt=average_txt)
        with torch.no_grad():
            meshes.scale_verts_(self.predict_scales(features))
        return meshes


    def predict_disp_verts_coarse(self, verts, features):
        disp_verts = self.deform_field_global(verts, features)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def predict_disp_verts(self, verts, features_local, mask, positional_enc,need_encoder=True):
        features_local = self.transformer(features_local, mask, self.query_embed.weight,positional_enc,need_encoder)[0]
        disp_verts = self.deform_field(verts, features_local)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def predict_textures_coarse(self, faces, features, average_txt=False):
        B = len(features)
        maps = self.txt_generator(features)
        #("map shape",maps.shape)
        if average_txt:
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.2)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_textures(self, faces, features_local, mask, positional_enc, need_encoder=True, average_txt=False):
        B = len(features_local)

        maps = self.transformer_txt(features_local, mask, self.tx_query_embed.weight, positional_enc, need_encoder)[0]
        maps = self.txt_field(maps)
        if average_txt:
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.2)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    def predict_poses(self, features):
        B = len(features)

        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0)  # KBC
        R_pred = R_pred.tanh()[..., [1, 0, 2]]  # XXX for retro-compatibility
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)
        if self.n_poses > 1:
            weights = self.proba_regressor(features.view(B, -1)).permute(1, 0)
            self._pose_proba = torch.softmax(weights, dim=0)  # KB
            if self.hard_select:
                indices = self._pose_proba.max(0)[1]
                select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
                R, T = map(select_fn, [R, T])
        return R, T

    def predict_saliency(self, features):
        res = self.slcy_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res, res

    def predict_background(self, features):
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    def compute_losses(self, meshes, imgs, alpha, slcys, rec, rec_img,  R, T, meshes_coarse=None, alpha_coarse=None, rec_img_coarse=None, average_txt=False, mask_gt=None):
        K, B = self.n_candidates, len(imgs)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
            mask_gt = mask_gt.repeat(K, 1, 1, 1)
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        if self.training:
            update_3d, update_pose = (not self.pose_step, self.pose_step) if self.alternate_optim else (True, True)
        else:
            update_3d, update_pose = (False, False)

        if self.pred_background:
            if 'rgb' in losses:
                if self.is_live('coarse_shape'):
                    losses['rgb'] = self.loss_weights['rgb'] * (
                                self.criterion(rec_img, imgs).flatten(1).mean(1) +
                                self.criterion(rec_img_coarse, imgs).flatten(1).mean(1))

                else:
                    losses['rgb'] = self.loss_weights['rgb'] * (
                        self.criterion(rec_img, imgs).flatten(1).mean(1))

            if 'bce_iou' in losses:
                if self.is_live('coarse_shape'):
                    losses['bce_iou'] = self.loss_weights['bce_iou'] * (
                                self.bce_iou_loss(slcys, mask_gt) + self.bce_iou_loss(alpha, mask_gt)
                                + self.bce_iou_loss(alpha_coarse, mask_gt))
                else:
                    losses['bce_iou'] = self.loss_weights['bce_iou'] * (
                            self.bce_iou_loss(slcys, mask_gt) + self.bce_iou_loss(alpha, mask_gt))

            if 'perceptual' in losses:
                if self.is_live('coarse_shape'):
                    losses['perceptual'] = self.loss_weights['perceptual'] * (
                                self.perceptual_loss(rec_img, imgs) +
                                self.perceptual_loss(rec_img_coarse, imgs))
                else:
                    losses['perceptual'] = self.loss_weights['perceptual'] * (
                        self.perceptual_loss(rec_img * alpha, imgs * mask_gt))
        else:
            if 'rgb' in losses:
                if self.is_live('coarse_shape'):
                    losses['rgb'] = self.loss_weights['rgb'] * (self.criterion(rec_img*alpha, imgs*mask_gt).flatten(1).mean(1) +
                                                            self.criterion(rec_img_coarse*alpha_coarse, imgs*mask_gt).flatten(1).mean(1))

                else:
                    losses['rgb'] = self.loss_weights['rgb'] * (
                                self.criterion(rec_img*alpha, imgs*mask_gt).flatten(1).mean(1))

            if 'bce_iou' in losses:
                if self.is_live('coarse_shape'):
                    losses['bce_iou'] = self.loss_weights['bce_iou'] * (self.bce_iou_loss(slcys, mask_gt)+self.bce_iou_loss(alpha, mask_gt)
                                                                    +self.bce_iou_loss(alpha_coarse, mask_gt))
                else:
                    losses['bce_iou'] = self.loss_weights['bce_iou'] * (
                                self.bce_iou_loss(slcys, mask_gt) + self.bce_iou_loss(alpha, mask_gt))

            if 'perceptual' in losses:
                if self.is_live('coarse_shape'):
                    losses['perceptual'] = self.loss_weights['perceptual'] * (self.perceptual_loss(rec_img*alpha, imgs*mask_gt) +
                                                                            self.perceptual_loss(rec_img_coarse*alpha_coarse, imgs*mask_gt))
                else:
                    losses['perceptual'] = self.loss_weights['perceptual'] * (self.perceptual_loss(rec_img*alpha, imgs*mask_gt))

        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * (normal_consistency(meshes_coarse))
                #print("normal loss shape", losses['normal'].shape)
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * (laplacian_smoothing(meshes_coarse, method='uniform'))
            if 'normal_refine' in losses:
                losses['normal_refine'] = self.loss_weights['normal_refine'] * (normal_consistency(meshes))
                # print("normal loss shape", losses['normal'].shape)
            if 'laplacian_refine' in losses:
                losses['laplacian_refine'] = self.loss_weights['laplacian_refine'] * (
                    laplacian_smoothing(meshes, method='uniform'))
        if update_3d and 'neighbor' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            if self.is_live('coarse_shape'):
                verts, faces, textures = meshes_coarse.verts_padded(), meshes_coarse.faces_padded(), meshes_coarse.textures
            else:
                verts, faces, textures = meshes.verts_padded(), meshes.faces_padded(), meshes.textures

            scales = self._scales[:, None]
            if self.is_live('coarse_shape'):
                z_sh, z_tx = [m._latent for m in [self.deform_field_global, self.txt_field]]
            else:
                z_sh, z_tx = [m._latent for m in [self.deform_field, self.txt_field]]
                #z_sh, z_tx = [m._latent for m in [self.deform_field, self.txt_generator]]
            z_bg = self.bkg_generator._latent if self.pred_background else torch.empty(B, 1, device=dev)
            for n, t in [('sh', z_sh), ('tx', z_tx), ('bg',z_bg), ('S', scales), ('R', R), ('T', T), ('img', imgs), ('mask_gt',mask_gt)]:
                self.nbr_memory[n] = torch.cat([self.nbr_memory[n].to(dev), t.detach()])[-self.nbr_memsize:]
            min_angle, nb_vpbins = self.nbr_min_angle, self.nbr_n_vpbins
            with torch.no_grad():
                if self.is_live('coarse_shape'):
                    sim_sh = (z_sh[None] - self.nbr_memory['sh'][:, None]).pow(2).sum(-1)
                    sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).view(self.nbr_memory['tx'].shape[0],
                                                                            z_tx.shape[0], -1).pow(2).sum(-1)
                else:
                    sim_sh = (z_sh[None] - self.nbr_memory['sh'][:, None]).view(self.nbr_memory['sh'].shape[0],
                                                                                z_sh.shape[0], -1).pow(2).sum(-1)
                    sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).view(self.nbr_memory['tx'].shape[0],
                                                                            z_tx.shape[0], -1).pow(2).sum(-1)
                angles = cpu_angle_between(self.nbr_memory['R'][:, None], R[None]).view(sim_sh.shape)
                angle_bins = torch.randint(0, nb_vpbins, (B,), device=dev).float()
                # we create bins with equal angle range and sample from them
                bin_size = (180. - min_angle) / nb_vpbins  # we compute the size for uniform bins
                # invalid items are items whose angles are outside [min_angle, max_angle[
                min_angles, max_angles = [(angle_bins + k) * bin_size + min_angle for k in range(2)]
                invalid_mask = (angles < min_angles).float() + (angles >= max_angles).float()
                idx_sh, idx_tx = map(lambda t: (t + t.max() * invalid_mask).argmin(0), [sim_sh, sim_tx])
                #print("idx_sh", idx_sh.shape)

            v_src, f_src = self.mesh_src.get_mesh_verts_faces(0)
            nbr_list, select = [], lambda n, indices: self.nbr_memory[n][indices]
            sh_imgs, tx_imgs = select('img', idx_sh), select('img', idx_tx)
            sh_mask_gt, tx_mask_gt = select('mask_gt', idx_sh), select('mask_gt', idx_tx)
            loss = 0.
            with torch.no_grad():
                if self.shared_encoder:
                    tx_global_features = self.global_encoder(tx_imgs)
                    tx_local_features = self.encoder(tx_imgs)
                    tx_sy,tx_sy_ = self.predict_saliency(tx_local_features)
                    tx_hard_slcy = torch.where(tx_sy_ > 0.5, 1, 0).detach()
                    tx_tensor = torch.empty(tx_local_features.shape[0], 1, 64, 64)
                    tx_mask = torch.ones_like(tx_tensor, device=imgs.device)

                    tx_mask = (tx_mask == 0)
                    tx_fg_features_local = tx_local_features
                    tx_positional_enc = self.positional_encod(NestedTensor(tx_fg_features_local, tx_mask))
                    if self.is_live('coarse_shape'):
                        tx_verts = v_src + self.predict_disp_verts_coarse(v_src, tx_global_features).view(B, -1,
                                                                                  3)
                    else:
                        tx_verts = v_src + self.predict_disp_verts(v_src, tx_fg_features_local, tx_mask,tx_positional_enc,NEED_ENCODER).view(B, -1,
                                                                                                          3)

                    tx_S = self.predict_scales(tx_global_features)[:, None]
                    tx_R, tx_T = self.predict_poses(tx_global_features)
                    tx_bg = self.predict_background(tx_global_features) if self.pred_background else None
                else:
                    tx_global_features = self.global_encoder(tx_imgs)
                    tx_local_features = self.encoder(tx_imgs)
                    tx_sy,tx_sy_ = self.predict_saliency(tx_local_features)
                    tx_hard_slcy = torch.where(tx_sy_ > 0.5, 1, 0).detach()
                    tx_tensor = torch.empty(tx_local_features.shape[0], 1, 64, 64)
                    tx_mask = torch.ones_like(tx_tensor, device=imgs.device)

                    tx_mask = (tx_mask == 0)
                    tx_fg_features_local = tx_local_features
                    tx_positional_enc = self.positional_encod(NestedTensor(tx_fg_features_local, tx_mask))
                    if self.is_live('coarse_shape'):
                        tx_verts = v_src + self.predict_disp_verts_coarse(v_src, tx_global_features).view(B, -1,
                                                                                                          3)
                    else:
                        tx_verts = v_src + self.predict_disp_verts(v_src, tx_fg_features_local, tx_mask,tx_positional_enc,NEED_ENCODER).view(B, -1,
                                                                                                          3)
                    tx_S = self.predict_scales(tx_global_features)[:, None]
                    tx_R, tx_T = self.predict_poses(tx_global_features)
                    tx_bg = self.predict_background(tx_global_features) if self.pred_background else None
            tx_mesh = Meshes(tx_verts * tx_S, faces, textures)
            nbr_list.append([tx_mesh, tx_R, tx_T, tx_sy,tx_bg, tx_imgs, tx_mask_gt])

            for nbr_inp in nbr_list:
                nbr_mesh, R, T, slcys, bkgs, imgs, mask_gt = nbr_inp
                rec_sw, alpha_sw = self.renderer(nbr_mesh, R, T).split([3, 1], dim=1)
                rec_sw_img = rec_sw * alpha_sw + (1 - alpha_sw) * bkgs[:B] if bkgs is not None else rec_sw
                if self.pred_background:
                    if 'rgb' in losses:
                        loss += self.loss_weights['rgb'] * self.criterion(rec_sw_img, imgs).flatten(1).mean(1)
                    if 'perceptual' in losses:
                        loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw_img, imgs)
                else:
                    if 'rgb' in losses:
                        loss += self.loss_weights['rgb'] * self.criterion(rec_sw_img*alpha_sw, imgs*mask_gt).flatten(1).mean(1)
                    if 'perceptual' in losses:
                        loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw_img*alpha_sw, imgs*mask_gt)


            losses['neighbor'] = self.loss_weights['neighbor'] * loss
        # Pose priors
        if update_pose and 'uniform' in losses:
            losses['uniform'] = self.loss_weights['uniform'] * (self._pose_proba.mean(1) - 1 / K).abs().mean()
            #[6,32]
        dist = sum(losses.values())
        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()

            # For monitoring purpose only
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B

        else:
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = v.mean()

        losses['total'] = dist.mean()
        return losses, select_idx

    def iter_step(self):
        self.cur_iter += 1
        if self.alternate_optim and self.cur_iter % self.alternate_optim == 0:
            self.pose_step = not self.pose_step

    def step(self):
        self.cur_epoch += 1
        self.deform_field.step()
        self.deform_field_global.step()
        self.txt_generator.step()
        self.txt_field.step()
        if self.pred_background:
            self.bkg_generator.step()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch
        self.deform_field.set_cur_milestone(epoch)
        self.deform_field_global.set_cur_milestone(epoch)
        self.txt_generator.set_cur_milestone(epoch)
        self.txt_field.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state and name != 'T_init':
            #if name in state:
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    ########################
    # Visualization utils
    ########################

    def get_synthetic_textures(self, colored=False):
        verts = self.mesh_src.verts_packed()
        if colored:
            colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        else:
            colors = torch.ones(verts.shape, device=verts.device) * 0.8
        return TexturesVertex(verts_features=colors[None])

    def get_prototype(self):
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent = torch.zeros(1, NUMBER_OF_QUERIES, self.n_features, device=verts.device)
        meshes = self.mesh_src.offset_verts(self.deform_field(verts, latent).view(-1, 3))
        return meshes

    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_init[-1], elev=elev, azim=azim, device=mesh.device)
        return self.renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        path = path_mkdir(path or Path('.'))
        d, elev = self.T_init[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.renderer)

    ########################
    # Evaluation utils
    ########################

    @torch.no_grad()
    def quantitative_eval(self, loader, device, evaluator=None):
        self.eval()
        #print(loader.dataset.name)
        if loader.dataset.name in ['cub_200']:
            evaluator = ProxyEvaluator()
            for inp, _ in loader:
                mask_gt = inp['masks'].to(device)
                imgs = inp['imgs'].to(device)
                meshes, _, (R, T), slcys, bkgs = self.predict_mesh_pose_slcy_bkg(inp['imgs'].to(device))
                mask_pred = self.renderer(meshes, R, T, viz_purpose=True).split([3, 1], dim=1)[1]  # (K*B)CHW
                if mask_pred.shape != mask_gt.shape:
                    mask_pred = F.interpolate(mask_pred, mask_gt.shape[-2:], mode='bilinear', align_corners=False)
                mask_pred = (mask_pred > 0.5).float()
                evaluator.update(mask_pred, mask_gt)

        else:
            if loader.dataset.name == 'p3d_car':
                print_warning('make sure that the canonical axes of predicted shapes correspond to the GT shapes axes')
            if evaluator is None:
                evaluator = MeshEvaluator()
            for inp, labels in loader:
                if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                    break
                #test with mesh coarse
                meshes, _, (R, T), slcys, bkgs = self.predict_mesh_pose_slcy_bkg(inp['imgs'].to(device))
                if not torch.all(inp['poses'] == -1):
                    # we use x_pred @ R_pred + T_pred = x_gt @ R_gt + T_gt to align predicted mesh with GT mesh
                    verts, faces = meshes.verts_padded(), meshes.faces_padded()
                    R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].to(device).split([3, 1], dim=2))
                    verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                    meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
                evaluator.update(meshes, torch_to(labels, device))
        return OrderedDict(zip(evaluator.metrics.names, evaluator.metrics.values))

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')

        renderer = self.renderer
        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        ##the evaluation of texture
        psnr_values = []
        ssim_values = []
        lpips_values = []
        # Initialize the LPIPS loss function
        loss_fn_alex = lpips.LPIPS(net='alex')  # You can use other networks like 'vgg' or 'squeeze'
        # Move the model to the same device as the data (e.g., GPU)
        loss_fn_alex.to(device)

        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs'].to(device)
            mask_gt = inp['masks'].to(device)
            meshes, _, (R, T), slcys, bkgs = self.predict_mesh_pose_slcy_bkg(imgs)
            rec, alpha = renderer(meshes, R, T).split([3, 1], dim=1)  # (K*B)CHW
            if bkgs is not None:
                rec = rec * alpha + (1 - alpha) * bkgs
            B, NV = len(imgs), 50
            d, e = self.T_init[-1], np.mean(self.elev_range)

            for k in range(B):
                if loader.dataset.name in ['cub_200'] and self.eval_mode:
                    imgs = self.transform_img(imgs)
                    rec = self.transform_img(rec)
                    alpha = self.transform_img(alpha)
                img_ =  np.array(convert_to_img(imgs[k] * mask_gt[k]))
                rec_ = np.array(convert_to_img(rec[k] * alpha[k]))
                psnr_val = psnr(rec_, img_, data_range=255)
                psnr_values.append(psnr_val)
                ssim_val = ssim(rec_, img_, data_range=255, channel_axis=2, multichannel=True)
                ssim_values.append(ssim_val)
                lpips_val = 0
                lpips_values.append(lpips_val)
            for k in range(B):
                i = str(j*B+k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprecorigin_full.png')
                convert_to_img(alpha[k]).save(path / f'{i}_inpalpha_full.png')
                convert_to_img(slcys[k]).save(path / f'{i}_inprec_wscly.png')
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')

                mcenter = normalize(meshes[k])
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter.textures = self.get_synthetic_textures(colored=True)
                save_mesh_as_obj(mcenter, path / f'{i}_meshuv.obj')
                save_mesh_as_gif(mcenter, path / f'{i}_meshuv_raw.gif', dist=d, elev=e, renderer=renderer)
            mean_ssim = np.mean(ssim_values)
            mean_psnr = np.mean(psnr_values)
            with open(path/'image_similarity_results.txt', 'w') as file:
                file.write(f'Mean PSNR: {mean_psnr}\n')
                file.write(f'Mean SSIM: {mean_ssim}\n')

    def transform_img(self, image):
        transform = Compose([
            Resize((256, 256), interpolation=InterpolationMode.NEAREST)
        ])
        return transform(image)
