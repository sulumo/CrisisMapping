"""
ai_core/architectures/baseline.py
──────────────────────────────────
Baseline CLIP_CrisisKAN architecture — single pooler output, shared
cross-attention, three classifier heads receiving the same joint representation.

This matches the architecture used to produce baseline/best.pt.
Set MODEL_VARIANT=baseline in your environment to use this.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIP_CrisisKAN_Baseline(nn.Module):

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

        self.image_map = nn.Sequential(
            copy.deepcopy(clip.visual_projection),
            nn.ReLU(),
            nn.Linear(clip.projection_dim, dim_visual_repr),
        )
        self.text_map = nn.Sequential(
            copy.deepcopy(clip.text_projection),
            nn.ReLU(),
            nn.Linear(clip.projection_dim, dim_text_repr),
        )
        del clip

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

    def forward(self, image, text):
        img = self.image_map(self.image_encoder(pixel_values=image).pooler_output)
        txt = self.text_map(self.text_encoder(**text).pooler_output)

        f_sa = self.apply_self_attention(img)
        e_sa = self.apply_self_attention(txt)

        f_t  = F.relu(self.proj_visual_bn(self.proj_visual(img)))
        e_t  = F.relu(self.proj_text_bn(self.proj_text(txt)))

        alpha_v = torch.sigmoid(self.layer_attn_text(e_sa))
        alpha_e = torch.sigmoid(self.layer_attn_visual(f_sa))

        joint = torch.cat((alpha_v * f_t, alpha_e * e_t), dim=1)
        fused = self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint))))

        return {
            "task1": self.cls_task1(fused),
            "task2": self.cls_task2(fused),
            "task3": self.cls_task3(fused),
        }