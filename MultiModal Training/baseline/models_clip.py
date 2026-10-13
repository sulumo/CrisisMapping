import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from transformers import BertModel, BertConfig, AlbertModel, AlbertConfig, XLNetConfig, XLNetModel, ElectraConfig, ElectraModel
import os
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from transformers import CLIPModel, AutoConfig, AutoModel
import copy


class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)

    def load_best(self):
        print(f"loading best checkpoint from file {self.save_dir}/best.pt")
        state_dict = torch.load(os.path.join(self.save_dir, 'best.pt'))
        self.load_state_dict(state_dict, strict=False)


class CLIP_CrisisKAN(BaseModel):

    def apply_self_attention(self, input_tensor):
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, input_tensor)
        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=768, dim_text_repr=768, dim_proj=128):
        super(CLIP_CrisisKAN, self).__init__(save_dir)
        
        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr
        self.map_dim = dim_proj

        # -------------------------
        # CLIP encoders (frozen)
        # -------------------------
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = copy.deepcopy(self.clip.vision_model)

        self.text_encoder = copy.deepcopy(self.clip.text_model)

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)
        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout()

        self.image_map = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_visual_repr)
        )
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_text_repr)
        )

        del self.clip   # free memory

        # --------------------------------
        # Projections to common dimension
        # --------------------------------
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Guided cross attention masks
        self.layer_attn_visual = nn.Linear(self.dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(self.dim_text_repr, dim_proj)

        # Fusion + refinement block
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        final_dim = 2 * dim_proj

        # --------------------------------
        # MULTI-TASK CLASSIFIER HEADS
        # --------------------------------
        self.cls_task1 = nn.Linear(final_dim, 2)  # informative
        self.cls_task2 = nn.Linear(final_dim, 6)  # humanitarian
        self.cls_task3 = nn.Linear(final_dim, 3)  # damage


    def forward(self, x):
        image, text = x

        # -----------------------------
        # CLIP encodings (frozen)
        # -----------------------------
        image_features = self.image_encoder(pixel_values=image).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(**text).pooler_output
        text_features = self.text_map(text_features)

        # Self-attention inside each modality
        f_i_self_attn = self.apply_self_attention(image_features)
        e_i_self_attn = self.apply_self_attention(text_features)

        # Linear projections (Eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(self.proj_visual(image_features)))
        e_i_tilde = F.relu(self.proj_text_bn(self.proj_text(text_features)))

        # Guided cross-attention masks
        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))

        # Masked representations
        masked_v_i = alpha_v_i * f_i_tilde
        masked_e_i = alpha_e_i * e_i_tilde

        # Joint representation
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)

        # Refinement block
        fused = self.fc_as_self_attn(joint_repr)
        fused = self.self_attn_bn(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)


        # -----------------------------
        # MULTI-TASK OUTPUT
        # -----------------------------
        return {
            "task1": self.cls_task1(fused),
            "task2": self.cls_task2(fused),
            "task3": self.cls_task3(fused)
        }

