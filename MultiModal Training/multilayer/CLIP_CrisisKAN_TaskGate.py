import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, AlbertModel, AlbertConfig, XLNetConfig, XLNetModel, ElectraConfig, ElectraModel
import os
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from transformers import CLIPModel
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
    """
    CLIP_CrisisKAN with Task-Conditioned Gated Multi-Layer Visual Fusion (Option C).

    Architecture overview
    ---------------------
    Three ViT stages are extracted and projected independently:
        - Early : hidden_states[3]  — textures, edges
        - Mid   : hidden_states[7]  — structures, object parts
        - Late  : pooler_output     — semantics, CLIP-aligned

    Each task head gets its own dedicated gating network that learns an independent softmax weighting
    over the three stage projections. This means:
        - Task 1 (informative)   can learn to rely heavily on late-stage semantics
        - Task 2 (humanitarian)  can learn to emphasise mid + late structure
        - Task 3 (damage)        can learn to up-weight early textures / mid structure

    The gradient from each task loss flows directly through its own gate,
    giving each task explicit control over which visual stages it uses.

    Shared cross-modal pipeline
    ---------------------------
    After task-specific gating produces three fused visual features
    (one per task), each passes through a *shared* cross-attention block
    with the text encoder output. The cross-attention weights (alpha_v, alpha_e)
    are computed from the task-specific visual feature, so they too vary per task.
    The final joint representation is task-specific and fed to the corresponding
    classifier head.

    Parameter overhead vs baseline
    --------------------------------
    +3 gate networks    : 3 × (3*dim_v → dim_v//2 → 3)   ~small
    +3 stage projectors : already present from Option A/B
    +2 extra cross-attn branches : proj_visual, proj_text, layer_attn_* are shared
    Net addition        : ~3–4M trainable params on top of frozen encoders

    Layer index reference — CLIP-ViT-Base/Patch32 (12 transformer blocks):
        hidden_states[0]       : patch embedding output
        hidden_states[1..12]   : after each transformer block
        pooler_output          : hidden_states[12] post LayerNorm
    """

    # ------------------------------------------------------------------
    # Self-attention helper (unchanged from baseline)
    # ------------------------------------------------------------------
    def apply_self_attention(self, input_tensor):
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, input_tensor)
        return attn_output

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, save_dir, dim_visual_repr=768, dim_text_repr=768, dim_proj=128):
        super(CLIP_CrisisKAN, self).__init__(save_dir)

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr   = dim_text_repr
        self.map_dim         = dim_proj

        # ---------------------------------------------------------------
        # CLIP encoders (frozen)
        # ---------------------------------------------------------------
        self.clip          = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder  = copy.deepcopy(self.clip.text_model)

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)
        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout()

        # ---------------------------------------------------------------
        # Text projection (unchanged from baseline)
        # ---------------------------------------------------------------
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_text_repr)
        )

        # ---------------------------------------------------------------
        # Per-stage visual projections
        #
        # Each stage projected independently: raw ViT dim (768) → dim_visual_repr.
        # Late stage reuses pretrained CLIP visual_projection as warm-start init.
        # ---------------------------------------------------------------
        clip_proj_dim = self.clip.projection_dim  # 512 for ViT-B/32

        self.image_map_early = nn.Sequential(
            nn.Linear(768, clip_proj_dim),
            nn.ReLU(),
            nn.Linear(clip_proj_dim, self.dim_visual_repr)
        )
        self.image_map_mid = nn.Sequential(
            nn.Linear(768, clip_proj_dim),
            nn.ReLU(),
            nn.Linear(clip_proj_dim, self.dim_visual_repr)
        )
        self.image_map_late = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),  # pretrained init
            nn.ReLU(),
            nn.Linear(clip_proj_dim, self.dim_visual_repr)
        )

        del self.clip  # free memory

        # ---------------------------------------------------------------
        # Task-conditioned gating networks
        #
        # One gate per task. Each gate takes the concatenation of the
        # three projected stage features (B, 3*dim_visual_repr) and
        # produces three softmax weights (B, 3) for that task only.
        #
        # Architecture: two-layer MLP with bottleneck at dim_visual_repr//2.
        # All three gates share the same architecture but have entirely
        # separate parameters, so their gradients are decoupled.
        #
        # Gate weight interpretation:
        #   output[:, 0] → weight on early (layer 3)
        #   output[:, 1] → weight on mid   (layer 7)
        #   output[:, 2] → weight on late  (pooler)
        # ---------------------------------------------------------------
        gate_input_dim  = 3 * self.dim_visual_repr
        gate_hidden_dim = self.dim_visual_repr // 2

        def _make_gate():
            return nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 3),
                nn.Softmax(dim=-1)
            )

        self.gate_task1 = _make_gate()   # informative
        self.gate_task2 = _make_gate()   # humanitarian
        self.gate_task3 = _make_gate()   # damage severity

        # BN applied after gated fusion, one per task
        self.fused_bn_task1 = nn.BatchNorm1d(self.dim_visual_repr)
        self.fused_bn_task2 = nn.BatchNorm1d(self.dim_visual_repr)
        self.fused_bn_task3 = nn.BatchNorm1d(self.dim_visual_repr)

        # ---------------------------------------------------------------
        # Shared projections to cross-modal dimension
        #
        # proj_visual / proj_text are shared across tasks — the
        # task-specific information is already encoded in the gated
        # visual features entering these projections.
        # ---------------------------------------------------------------
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text   = nn.Linear(dim_text_repr,   dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn   = nn.BatchNorm1d(dim_proj)

        # ---------------------------------------------------------------
        # Guided cross-attention masks (shared)
        #
        # These are applied per-task using the task-specific visual
        # feature for the self-attention inputs, so the masks vary
        # across tasks despite the layers being shared.
        # ---------------------------------------------------------------
        self.layer_attn_visual = nn.Linear(self.dim_visual_repr, dim_proj)
        self.layer_attn_text   = nn.Linear(self.dim_text_repr,   dim_proj)

        # ---------------------------------------------------------------
        # Fusion + refinement block (shared)
        # ---------------------------------------------------------------
        self.fc_as_self_attn = nn.Linear(2 * dim_proj, 2 * dim_proj)
        self.self_attn_bn    = nn.BatchNorm1d(2 * dim_proj)

        final_dim = 2 * dim_proj

        # ---------------------------------------------------------------
        # Multi-task classifier heads (unchanged)
        # ---------------------------------------------------------------
        self.cls_task1 = nn.Linear(final_dim, 2)  # informative
        self.cls_task2 = nn.Linear(final_dim, 6)  # humanitarian
        self.cls_task3 = nn.Linear(final_dim, 3)  # damage severity

    # ------------------------------------------------------------------
    # Internal: shared cross-attention + refinement block
    #
    # Encapsulates the guided cross-attention pipeline so it can be
    # called once per task without duplicating code. Takes the
    # task-specific fused visual feature and the shared text feature.
    #
    # Args:
    #   image_features : (B, dim_visual_repr) — task-specific gated visual
    #   text_features  : (B, dim_text_repr)   — shared text encoding
    #   fused_bn       : the task-specific BN for the gated visual input
    #
    # Returns:
    #   fused : (B, 2*dim_proj) — joint representation ready for classifier
    # ------------------------------------------------------------------
    def _cross_attn_branch(self, image_features, text_features, fused_bn):
        # Apply task-specific BN to the incoming gated visual feature
        image_features = F.relu(fused_bn(image_features))

        # Self-attention within each modality
        f_i_self_attn = self.apply_self_attention(image_features)
        e_i_self_attn = self.apply_self_attention(text_features)

        # Project to cross-modal dimension (Eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(self.proj_visual(image_features)))
        e_i_tilde = F.relu(self.proj_text_bn(self.proj_text(text_features)))

        # Guided cross-attention masks
        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))    # text guides vision
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # vision guides text

        # Masked representations
        masked_v_i = alpha_v_i * f_i_tilde
        masked_e_i = alpha_e_i * e_i_tilde

        # Joint representation + refinement
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)  # (B, 2*dim_proj)
        fused = self.fc_as_self_attn(joint_repr)
        fused = self.self_attn_bn(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)

        return fused

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x):
        image, text = x

        # ---------------------------------------------------------------
        # Multi-layer visual encoding
        # ---------------------------------------------------------------
        vision_outputs = self.image_encoder(
            pixel_values=image,
            output_hidden_states=True
        )

        # CLS token at each stage — shape: (B, 768)
        f_early = vision_outputs.hidden_states[3][:, 0, :]   # beginning stage
        f_mid   = vision_outputs.hidden_states[7][:, 0, :]   # middle stage
        f_late  = vision_outputs.pooler_output                # ending stage

        # Project each stage to dim_visual_repr — shape: (B, dim_visual_repr)
        v_early = self.image_map_early(f_early)
        v_mid   = self.image_map_mid(f_mid)
        v_late  = self.image_map_late(f_late)

        # Shared inputs to all three gates
        gate_input = torch.cat([v_early, v_mid, v_late], dim=1)   # (B, 3*dim_visual_repr)
        stacked    = torch.stack([v_early, v_mid, v_late], dim=1)  # (B, 3, dim_visual_repr)

        # ---------------------------------------------------------------
        # Task-conditioned gating
        #
        # Each gate independently weights the three stage projections.
        # gates_tkN shape : (B, 3)
        # image_feats_tkN : (B, dim_visual_repr) — task-specific fused visual
        # ---------------------------------------------------------------
        gates_t1 = self.gate_task1(gate_input)                            # (B, 3)
        gates_t2 = self.gate_task2(gate_input)                            # (B, 3)
        gates_t3 = self.gate_task3(gate_input)                            # (B, 3)

        image_feats_t1 = (stacked * gates_t1.unsqueeze(-1)).sum(dim=1)   # (B, dim_visual_repr)
        image_feats_t2 = (stacked * gates_t2.unsqueeze(-1)).sum(dim=1)
        image_feats_t3 = (stacked * gates_t3.unsqueeze(-1)).sum(dim=1)

        # ---------------------------------------------------------------
        # Text encoding (shared across all tasks — unchanged)
        # ---------------------------------------------------------------
        text_features = self.text_encoder(**text).pooler_output  # (B, 768)
        text_features = self.text_map(text_features)             # (B, dim_text_repr)

        # ---------------------------------------------------------------
        # Per-task cross-attention + refinement
        #
        # Each task passes its own gated visual feature through the
        # shared cross-attention pipeline. The shared weights see
        # task-specific inputs, so the attention masks and projections
        # differ per task without requiring separate weight matrices.
        # ---------------------------------------------------------------
        fused_t1 = self._cross_attn_branch(image_feats_t1, text_features, self.fused_bn_task1)
        fused_t2 = self._cross_attn_branch(image_feats_t2, text_features, self.fused_bn_task2)
        fused_t3 = self._cross_attn_branch(image_feats_t3, text_features, self.fused_bn_task3)

        # ---------------------------------------------------------------
        # Multi-task output
        # Each head receives its own task-specific joint representation.
        # ---------------------------------------------------------------
        return {
            "task1": self.cls_task1(fused_t1),  # informative     (2 classes)
            "task2": self.cls_task2(fused_t2),  # humanitarian    (6 classes)
            "task3": self.cls_task3(fused_t3),  # damage severity (3 classes)
        }

    # ------------------------------------------------------------------
    # Diagnostic: return gate weights for all three tasks simultaneously
    #
    # Returns a dict of three (B, 3) tensors — one per task.
    # Columns: [w_early, w_mid, w_late]
    #
    # Usage (after training):
    #   with torch.no_grad():
    #       weights = model.get_gate_weights(image_batch)
    #       # weights['task1'] shape: (B, 3)
    #       # weights['task2'] shape: (B, 3)
    #       # weights['task3'] shape: (B, 3)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_gate_weights(self, image):
        vision_outputs = self.image_encoder(
            pixel_values=image,
            output_hidden_states=True
        )
        v_early = self.image_map_early(vision_outputs.hidden_states[3][:, 0, :])
        v_mid   = self.image_map_mid(vision_outputs.hidden_states[7][:, 0, :])
        v_late  = self.image_map_late(vision_outputs.pooler_output)

        gate_input = torch.cat([v_early, v_mid, v_late], dim=1)

        return {
            'task1': self.gate_task1(gate_input),  # (B, 3)
            'task2': self.gate_task2(gate_input),  # (B, 3)
            'task3': self.gate_task3(gate_input),  # (B, 3)
        }