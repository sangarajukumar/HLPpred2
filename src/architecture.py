import torch
import torch.nn as nn
import torch.nn.functional as F

class Hemopi_arch(nn.Module):
    """
    HLPpred 2.0 Classification Architecture:
    Multi-Scale CNN (MSCL) + Transformer Encoder + Late-stage Multimodal Fusion.
    """
    def __init__(self, 
                 feature_dim_list,
                 max_concatenated_len,
                 n_classes,
                 # Transformer param
                 d_model,
                 n_head,
                 num_transformer_layers,
                 # CNN param
                 kernel_size,
                 stride,
                 dilation,
                 fc_1,
                 norm_type,
                 dp_size,
                 prob_feature_dim,
                 merci_feature_dim
                 ):
        super(Hemopi_arch, self).__init__()
        self.num_of_features = len(feature_dim_list)
        self.max_concatenated_len = max_concatenated_len
        self.d_model = d_model
        self.fc_1_layers = fc_1
        self.norm_type = norm_type
        self.dp_size = dp_size
        self.prob_feature_dim = prob_feature_dim
        self.merci_feature_dim = merci_feature_dim
        
        self.projector_list = nn.ModuleList()
        for idx, feature_dim in enumerate(feature_dim_list):
            remaining_dim = 0
            if idx == 0:
                remaining_dim = self.d_model % len(feature_dim_list) 
            self.projector_list.append(
                nn.Linear(feature_dim, self.d_model // len(feature_dim_list) + remaining_dim)
            )
            
        self.position_embedding = nn.Embedding(self.max_concatenated_len + 1, self.d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_head,
                activation='gelu',
                dim_feedforward=self.d_model * 4,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        nn.init.xavier_normal_(self.cls_token)
        
        self.conv_ms_1_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_1_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_2_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_2_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_3_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.conv_ms_3_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_1 = nn.Linear(self.d_model + self.prob_feature_dim + self.merci_feature_dim, self.fc_1_layers)
        self.fc_2 = nn.Linear(self.fc_1_layers, n_classes)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.drop_out = nn.Dropout(self.dp_size)
    
    def forward(self, X, masks_X, X_probs, X_merci):
        batch_size = X[0].size(0)
        
        projected_X = [
            projector(x) for projector, x in zip(self.projector_list, X)
        ]
        
        concatenated_X = torch.cat(projected_X, dim=-1)
        
        padded_X = torch.zeros(
            (batch_size, self.max_concatenated_len, self.d_model), device=X[0].device
        )
        combined_masks = torch.zeros(
            (batch_size, self.max_concatenated_len + 1), device=X[0].device, dtype=torch.long
        )
        pos_ids = torch.zeros((batch_size, self.max_concatenated_len + 1), device=X[0].device, dtype=torch.long)
        
        for batch_idx, emb in enumerate(concatenated_X):
            padded_X[batch_idx, :emb.size(0), :] = emb
            combined_masks[batch_idx, emb.size(0):] = 1
            pos_ids[batch_idx, 1:] = torch.arange(1, self.max_concatenated_len + 1)
        
        padded_X = torch.cat([self.cls_token.expand(padded_X.size(0), -1, -1), padded_X], dim=1)
        pos_embeds = self.position_embedding(pos_ids)
        padded_X = padded_X + pos_embeds
        
        transformer_output = self.transformer_encoder(
            padded_X,
            src_key_padding_mask=combined_masks.bool(),
        )
        
        cls_embeds = transformer_output[:, 0, :]
        seq_embeds = transformer_output[:, 1:, :]
        
        seq_embeds = seq_embeds.permute(0, 2, 1)
        seq_combined = self.global_avg_pool(seq_embeds)
        seq_combined = seq_combined.squeeze(-1)
        
        if self.norm_type == 'norm':
            seq_outputs = self.layer_norm(seq_combined)
        elif self.norm_type == 'batch':
            seq_outputs = self.batch_norm(seq_combined)
            
        combined_output = seq_outputs + cls_embeds
        features_to_concat = [combined_output]

        if self.prob_feature_dim > 0 and X_probs.size(-1) > 0:
            features_to_concat.append(X_probs)

        if self.merci_feature_dim > 0 and X_merci.size(-1) > 0:
            features_to_concat.append(X_merci)

        if len(features_to_concat) > 1:
            final_output = torch.cat(features_to_concat, dim=-1)
        else:
            final_output = combined_output

        pre_output = self.fc_1(final_output)
        pre_output = self.drop_out(pre_output)
        outputs = self.fc_2(pre_output)
        
        return outputs


class Hemopi_Regression_arch(nn.Module):
    """
    HLPpred 2.0 Potency Regression Architecture:
    Multi-Scale CNN (MSCL) + Transformer Encoder + Late-stage Multimodal Fusion for pHC50.
    """
    def __init__(self, 
                 feature_dim_list,
                 max_concatenated_len,
                 d_model,
                 n_head,
                 num_transformer_layers,
                 kernel_size,
                 stride,
                 dilation,
                 fc_1,
                 norm_type,
                 dp_size,
                 prob_feature_dim,
                 merci_feature_dim
                 ):
        super(Hemopi_Regression_arch, self).__init__()
        self.num_of_features = len(feature_dim_list)
        self.max_concatenated_len = max_concatenated_len
        self.d_model = d_model
        self.fc_1_layers = fc_1
        self.norm_type = norm_type
        self.dp_size = dp_size
        self.prob_feature_dim = prob_feature_dim
        self.merci_feature_dim = merci_feature_dim
        
        self.projector_list = nn.ModuleList()
        for idx, feature_dim in enumerate(feature_dim_list):
            remaining_dim = 0
            if idx == 0:
                remaining_dim = self.d_model % len(feature_dim_list) 
            self.projector_list.append(
                nn.Linear(feature_dim, self.d_model // len(feature_dim_list) + remaining_dim)
            )
            
        self.position_embedding = nn.Embedding(self.max_concatenated_len + 1, self.d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_head,
                activation='gelu',
                dim_feedforward=self.d_model * 4,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        nn.init.xavier_normal_(self.cls_token)
        
        self.conv_ms_1_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_1_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_2_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_2_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_3_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.conv_ms_3_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.prob_feature_dim > 0:
            self.fc_1 = nn.Linear(self.d_model + self.prob_feature_dim + self.merci_feature_dim, self.fc_1_layers)
        else:
            self.fc_1 = nn.Linear(self.d_model + self.merci_feature_dim, self.fc_1_layers)
            
        self.fc_2 = nn.Linear(self.fc_1_layers, 1)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.drop_out = nn.Dropout(self.dp_size)
    
    def forward(self, X, masks_X, X_probs, X_merci):
        batch_size = X[0].size(0)
        
        projected_X = [
            projector(x) for projector, x in zip(self.projector_list, X)
        ]
        
        concatenated_X = torch.cat(projected_X, dim=-1)
        
        padded_X = torch.zeros(
            (batch_size, self.max_concatenated_len, self.d_model), device=X[0].device
        )
        combined_masks = torch.zeros(
            (batch_size, self.max_concatenated_len + 1), device=X[0].device, dtype=torch.long
        )
        pos_ids = torch.zeros((batch_size, self.max_concatenated_len + 1), device=X[0].device, dtype=torch.long)
        
        for batch_idx, emb in enumerate(concatenated_X):
            padded_X[batch_idx, :emb.size(0), :] = emb
            combined_masks[batch_idx, emb.size(0):] = 1
            pos_ids[batch_idx, 1:] = torch.arange(1, self.max_concatenated_len + 1)
        
        padded_X = torch.cat([self.cls_token.expand(padded_X.size(0), -1, -1), padded_X], dim=1)
        pos_embeds = self.position_embedding(pos_ids)
        padded_X = padded_X + pos_embeds
        
        transformer_output = self.transformer_encoder(
            padded_X,
            src_key_padding_mask=combined_masks.bool(),
        )
        
        cls_embeds = transformer_output[:, 0, :]
        seq_embeds = transformer_output[:, 1:, :]
        
        seq_embeds = seq_embeds.permute(0, 2, 1)
        seq_embeds_ms_1 = F.leaky_relu(self.conv_ms_1_1(seq_embeds))
        seq_embeds_ms_1 = F.leaky_relu(self.conv_ms_1_2(seq_embeds_ms_1))
        
        seq_embeds_ms_2 = F.leaky_relu(self.conv_ms_2_1(seq_embeds))
        seq_embeds_ms_2 = F.leaky_relu(self.conv_ms_2_2(seq_embeds_ms_2))
        
        seq_embeds_ms_3 = F.leaky_relu(self.conv_ms_3_1(seq_embeds))
        seq_embeds_ms_3 = F.leaky_relu(self.conv_ms_3_2(seq_embeds_ms_3))
        
        seq_avg_1 = self.global_avg_pool(seq_embeds_ms_1)
        seq_avg_2 = self.global_avg_pool(seq_embeds_ms_2)
        seq_avg_3 = self.global_avg_pool(seq_embeds_ms_3)
        
        seq_combined = seq_avg_1 + seq_avg_2 + seq_avg_3
        seq_combined = seq_combined.squeeze(-1)
        
        if self.norm_type == 'norm':
            seq_outputs = self.layer_norm(seq_combined)
        elif self.norm_type == 'batch':
            seq_outputs = self.batch_norm(seq_combined)
            
        combined_output = seq_outputs + cls_embeds
        features_to_concat = [combined_output]

        if self.prob_feature_dim > 0 and X_probs.size(-1) > 0:
            features_to_concat.append(X_probs)

        if self.merci_feature_dim > 0 and X_merci.size(-1) > 0:
            features_to_concat.append(X_merci)

        if len(features_to_concat) > 1:
            final_output = torch.cat(features_to_concat, dim=-1)
        else:
            final_output = combined_output

        pre_output = self.fc_1(final_output)
        pre_output = self.drop_out(pre_output)
        outputs = self.fc_2(pre_output)

        return outputs.squeeze(-1)
