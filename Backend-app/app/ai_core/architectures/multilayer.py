"""
ai_core/architectures/multilayer.py
──────────────────────────────────
CLIP_CrisisKAN_TCG — Task-Conditioned Gated multi-layer visual fusion.

Three ViT stages (layers 3, 7, pooler) are projected independently.
Each task has its own gating network and its own cross-attention branch,
producing independent joint representations per task.

Set MODEL_VARIANT=multilayer in your environment to use this.
This architecture must match exactly the one used to produce multilayer/best.pt.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIP_CrisisKAN_TCG(nn.Module):

    def apply_self_attention(self, x):
        scores  = torch.matmul(x, x.transpose(-1, -2))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, x)

    def __init__(self, dim_visual_repr=768, dim_text_repr=768, dim_proj=128):
        super().__init__()
        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr   = dim_text_repr

        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.image_encoder = copy.deepcopy(clip.vision_model)
        self.text_encoder  = copy.deepcopy(clip.text_model)

        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout()

        self.text_map = nn.Sequential(
            copy.deepcopy(clip.text_projection),
            nn.ReLU(),
            nn.Linear(clip.projection_dim, dim_text_repr),
        )

        proj_dim = clip.projection_dim
        self.image_map_early = nn.Sequential(
            nn.Linear(768, proj_dim), nn.ReLU(), nn.Linear(proj_dim, dim_visual_repr)
        )
        self.image_map_mid = nn.Sequential(
            nn.Linear(768, proj_dim), nn.ReLU(), nn.Linear(proj_dim, dim_visual_repr)
        )
        self.image_map_late = nn.Sequential(
            copy.deepcopy(clip.visual_projection),
            nn.ReLU(),
            nn.Linear(proj_dim, dim_visual_repr),
        )
        del clip

        gate_in  = 3 * dim_visual_repr
        gate_hid = dim_visual_repr // 2

        def _gate():
            return nn.Sequential(
                nn.Linear(gate_in, gate_hid), nn.ReLU(),
                nn.Linear(gate_hid, 3), nn.Softmax(dim=-1),
            )

        self.gate_task1 = _gate()
        self.gate_task2 = _gate()
        self.gate_task3 = _gate()

        self.fused_bn_task1 = nn.BatchNorm1d(dim_visual_repr)
        self.fused_bn_task2 = nn.BatchNorm1d(dim_visual_repr)
        self.fused_bn_task3 = nn.BatchNorm1d(dim_visual_repr)

        self.proj_visual    = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text      = nn.Linear(dim_text_repr,   dim_proj)
        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn   = nn.BatchNorm1d(dim_proj)

        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text   = nn.Linear(dim_text_repr,   dim_proj)

        self.fc_as_self_attn = nn.Linear(2 * dim_proj, 2 * dim_proj)
        self.self_attn_bn    = nn.BatchNorm1d(2 * dim_proj)

        self.cls_task1 = nn.Linear(2 * dim_proj, 2)
        self.cls_task2 = nn.Linear(2 * dim_proj, 6)
        self.cls_task3 = nn.Linear(2 * dim_proj, 3)

    def _cross_attn_branch(self, img, txt, bn):
        img = F.relu(bn(img))
        f_sa = self.apply_self_attention(img)
        e_sa = self.apply_self_attention(txt)
        f_t  = F.relu(self.proj_visual_bn(self.proj_visual(img)))
        e_t  = F.relu(self.proj_text_bn(self.proj_text(txt)))
        alpha_v = torch.sigmoid(self.layer_attn_text(e_sa))
        alpha_e = torch.sigmoid(self.layer_attn_visual(f_sa))
        joint = torch.cat((alpha_v * f_t, alpha_e * e_t), dim=1)
        return self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint))))

    def forward(self, image, text):
        vis = self.image_encoder(pixel_values=image, output_hidden_states=True)

        v_early = self.image_map_early(vis.hidden_states[3][:, 0, :])
        v_mid   = self.image_map_mid(vis.hidden_states[7][:, 0, :])
        v_late  = self.image_map_late(vis.pooler_output)

        gate_in = torch.cat([v_early, v_mid, v_late], dim=1)
        stacked = torch.stack([v_early, v_mid, v_late], dim=1)

        img1 = (stacked * self.gate_task1(gate_in).unsqueeze(-1)).sum(1)
        img2 = (stacked * self.gate_task2(gate_in).unsqueeze(-1)).sum(1)
        img3 = (stacked * self.gate_task3(gate_in).unsqueeze(-1)).sum(1)

        txt = self.text_map(self.text_encoder(**text).pooler_output)

        f1 = self._cross_attn_branch(img1, txt, self.fused_bn_task1)
        f2 = self._cross_attn_branch(img2, txt, self.fused_bn_task2)
        f3 = self._cross_attn_branch(img3, txt, self.fused_bn_task3)

        return {
            "task1": self.cls_task1(f1),
            "task2": self.cls_task2(f2),
            "task3": self.cls_task3(f3),
        }