import copy

import torch
import torch.nn as nn

import clip

class HyperCrossAttention(nn.Module):
    """Hyper Cross-Attention Aggregation for decoders (only shared parameters in sam_layers and mem_layers)."""

    def __init__(self, models, K, init_beta=0.1, mode="both"):
        """
        :param models: List of models to extract shared parameter keys from (for sam_layers and mem_layers).
        :param K: Number of decoders (i.e., number of clients/models).
        :param init_beta: Initial value for the beta parameters.
        """
        super(HyperCrossAttention, self).__init__()
        self.K = K

        if mode == "both":
            self.specific_layer_prefixes = [
                'sam_mask_decoder',  # Example prefix for SAM layers
                'obj_ptr_proj', 
                'memory_encoder', 
                'memory_attention', 
                'mask_downsample'  # MEM layers
            ]
        elif mode == "sam":
            self.specific_layer_prefixes = ['sam_mask_decoder']
        elif mode == "mem":
            self.specific_layer_prefixes = [
                                            'obj_ptr_proj', 
                                            'memory_encoder', 
                                            'memory_attention', 
                                            'mask_downsample'
                                            ]

        # Extract shared parameter keys across all models
        self.common_keys = self.get_common_keys(models)

        # Generate beta names by replacing '.' with '_' for compatibility with ParameterDict
        self.beta_names = {key.replace('.', '_') for key in self.common_keys}

        # Define learnable beta parameters for each shared layer
        self.beta = nn.ParameterDict()
        for name in self.beta_names:
            self.beta[name] = nn.Parameter(torch.ones(K) * init_beta)  # Initialize beta for each shared layer

    def get_common_keys(self, models):
        """
        Extract common parameter keys shared across all models for sam_layers and mem_layers.
        :param models: List of models to extract common parameter keys.
        :return: Set of shared parameter keys across all models.
        """
        # Get parameter names for each model that belong to sam_layers and mem_layers
        model_param_keys = []
        for model in models:
            param_keys = set()
            for name, _ in model.named_parameters():
                if any(prefix in name for prefix in self.specific_layer_prefixes):
                    param_keys.add(name)  # Only add sam_layers and mem_layers
            model_param_keys.append(param_keys)

        # Find the intersection of parameter keys across all models (shared keys)
        common_keys = set.intersection(*model_param_keys)

        return common_keys

    def forward(self, last_param_dict_list, delta_dict_list):
        """
        Forward method to apply cross-attention and beta-weighted updates to shared parameters in sam_layers and mem_layers.

        :param last_param_dict_list: A list of dictionaries containing the last set of parameters for each decoder.
        :param delta_dict_list: A list of dictionaries containing parameter deltas for each decoder.
        :return: Updated parameters after applying beta-weighted cross-attention.
        """
        # Initialize new parameter dictionaries as a copy of the original but the original one requires grad
        # We only need the values of the original parameters, not the gradients
        
        new_param_dict_list = copy.deepcopy(last_param_dict_list)  # Initialize new param dicts as a copy of the original
        assert self.K == len(last_param_dict_list), "Number of decoders does not match K"

        # Iterate through all layers to ensure that fine-tune layers are updated and non-fine-tune layers are kept intact
        for i in range(self.K):
            for name, param in last_param_dict_list[i].items():
                if name in self.common_keys:  # Fine-tune the shared layers (sam_layers and mem_layers)
                    # Clamp beta weights to [0, 1] for the current layer
                    layer_beta = torch.clamp(self.beta[name.replace('.', '_')], 0, 1)

                    # Stack the deltas across decoders for this shared parameter
                    cross_delta = torch.stack([delta_dict_list[j][name].reshape(-1) for j in range(self.K)])

                    # Get the delta for the i-th decoder
                    self_delta = delta_dict_list[i][name].reshape(1, -1)

                    # Apply cross-attention
                    cross_attn_delta = CrossAttention(self_delta, cross_delta, cross_delta)

                    # Apply beta-weighted update
                    beta = layer_beta[i]
                    ori_shape = delta_dict_list[i][name].shape  # Preserve original shape of the parameter
                    new_delta = delta_dict_list[i][name] + beta * cross_attn_delta.reshape(ori_shape)

                    # Update the parameter in the new parameter dictionary
                    new_param_dict_list[i][name] += new_delta
                else:
                    # Keep the non-fine-tuned layers unchanged
                    new_param_dict_list[i][name] = param

        return new_param_dict_list



class HyperCrossAttention_embedding_res(nn.Module):
    def __init__(self, models, K, site_texts, delta_dim = 1, init_beta=0.1, mode="both", device='cuda'):
        """
        :param models: List of models to extract shared parameter keys from (for sam_layers and mem_layers).
        :param K: Number of decoders (i.e., number of clients/models).
        :param site_texts: List of site texts for each client, used to generate task embeddings via CLIP.
        :param delta_dim: Dimension of the delta values.
        :param init_beta: Initial value for the beta parameters.
        """
        super(HyperCrossAttention_embedding_res, self).__init__()
        self.K = K
        self.site_text = site_texts
        self.delta_dim = delta_dim
        model_clip, _ = clip.load("ViT-B/32", device='cuda')
    
        # Generate task embeddings from the site texts
        self.task_embeddings = []
        for site_text in site_texts:
            tokenized_text = clip.tokenize(site_text).to(device)
            with torch.no_grad():
                embedding = model_clip.encode_text(tokenized_text).squeeze(0) 
            self.task_embeddings.append(embedding)
        
        # Stack task embeddings into a tensor of shape (K, embedding_dim)
        self.task_embeddings = torch.stack(self.task_embeddings).to(device)

        if mode == "both":
            self.specific_layer_prefixes = [
                'sam_mask_decoder',
                'obj_ptr_proj',
                'memory_encoder',
                'memory_attention',
                'mask_downsample'
            ]
        elif mode == "sam":
            self.specific_layer_prefixes = ['sam_mask_decoder']
        elif mode == "mem":
            self.specific_layer_prefixes = [
                'obj_ptr_proj', 
                'memory_encoder', 
                'memory_attention', 
                'mask_downsample'
            ]

        # Extract common parameter keys
        self.common_keys = self.get_common_keys(models)
        self.beta_names = {key.replace('.', '_') for key in self.common_keys}
        self.beta = nn.ParameterDict()
        for name in self.beta_names:
            self.beta[name] = nn.Parameter(torch.ones(K) * init_beta)
        
        embedding_dim = self.task_embeddings.size(1)

        # MLP to adjust task embeddings to match delta dimensions
        self.mlp_adjust = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, delta_dim)
        )

        # Gating mechanism
        self.site_gating = nn.Sequential(
            nn.Linear(delta_dim * 2, delta_dim),
            nn.Sigmoid()
        )

    def forward(self, last_param_dict_list, delta_dict_list):
        # Create a copy of the parameter dictionaries to avoid modifying the original
        new_param_dict_list = copy.deepcopy(last_param_dict_list)
        assert self.K == len(last_param_dict_list), "Number of decoders does not match K"

        # Adjust the task embeddings to match the delta dimensions
        adjusted_task_embeds = self.mlp_adjust(self.task_embeddings)  # (K, delta_dim)
        adjusted_task_embeds_pool = torch.mean(adjusted_task_embeds, dim=1, keepdim=True) 
        # print(adjusted_task_embeds.shape)  # 5,16
        # print(adjusted_task_embeds_pool.shape) # 5,1
        for i in range(self.K):
            for name, param in last_param_dict_list[i].items():
                if name in self.common_keys:
                    layer_beta = torch.clamp(self.beta[name.replace('.', '_')], 0, 1)

                    # Stack deltas from all clients for the specific parameter
                    cross_delta = torch.stack([delta_dict_list[j][name].reshape(-1) for j in range(self.K)]) # (K, delta_dim)
                    self_delta = delta_dict_list[i][name].reshape(1, -1) # (1, delta_dim)

                    # print(cross_delta.shape) # 5,16
                    # print(self_delta.shape)  # 1,16
                    # Pool the deltas using mean pooling


                    cross_delta_pooled = torch.mean(cross_delta, dim=1, keepdim=True)  # (K, 1)
                    self_delta_pooled = torch.mean(self_delta, dim=1, keepdim=True)  # (1, 1)


                    # print(cross_delta_pooled.shape) # 5,1
                    # print(self_delta_pooled.shape) # 1,1
                    # Element-wise multiplication of pooled deltas with adjusted task embeddings


                    cross_delta_with_indicator = torch.cat([cross_delta_pooled, adjusted_task_embeds_pool], dim=1)  # (K, delta_dim * 2)
                    self_delta_with_indicator = torch.cat([self_delta_pooled, adjusted_task_embeds_pool[i].unsqueeze(0)], dim=1) # (1, delta_dim * 2)
                    
                    
                    # print(cross_delta_with_indicator.shape) # 5,2
                    # print(self_delta_with_indicator.shape) # 1,2
                    # print(self.site_gating(cross_delta_with_indicator).shape)
                    # print(self.site_gating(self_delta_with_indicator).shape)

                    
                    cross_delta_with_embedding = self.site_gating(cross_delta_with_indicator) * cross_delta + cross_delta
                    self_delta_with_embedding = self.site_gating(self_delta_with_indicator) * self_delta + self_delta
                    
                    
                    # print(cross_delta_with_embedding.shape)
                    
                    
                    # print(self_delta_with_embedding.shape)
                    # Apply cross-attention
                    cross_attn_delta = CrossAttention(self_delta_with_embedding, cross_delta_with_embedding, cross_delta_with_embedding)

                    beta = layer_beta[i]
                    ori_shape = delta_dict_list[i][name].shape
                    new_delta = delta_dict_list[i][name] + beta * cross_attn_delta.reshape(ori_shape)

                    # Update the parameter in the new parameter dictionary
                    new_param_dict_list[i][name] += new_delta
                else:
                    # Keep the non-fine-tuned layers unchanged
                    new_param_dict_list[i][name] = param

        return new_param_dict_list
    
    def get_common_keys(self, models):
        """
        Extract common parameter keys shared across all models for sam_layers and mem_layers.
        :param models: List of models to extract common parameter keys.
        :return: Set of shared parameter keys across all models.
        """
        # Get parameter names for each model that belong to sam_layers and mem_layers
        model_param_keys = []
        for model in models:
            param_keys = set()
            for name, _ in model.named_parameters():
                if any(prefix in name for prefix in self.specific_layer_prefixes):
                    param_keys.add(name)  # Only add sam_layers and mem_layers
            model_param_keys.append(param_keys)

        # Find the intersection of parameter keys across all models (shared keys)
        common_keys = set.intersection(*model_param_keys)

        return common_keys
def CrossAttention(q, k, v):
    """
    Cross-attention mechanism to calculate attention-weighted deltas.
    
    :param q: Query tensor (self delta for a given decoder).
    :param k: Key tensor (cross deltas across decoders).
    :param v: Value tensor (same as keys for delta updates).
    :return: Attention-weighted output.
    """
    scale = q.size(-1) ** -0.5  # Scaling factor
    attn = (q @ k.transpose(-2, -1)) * scale  # Compute scaled dot-product attention
    attn = nn.Softmax(dim=-1)(attn)  # Apply softmax to get attention weights
    out = attn @ v  # Compute attention-weighted output

    return out

