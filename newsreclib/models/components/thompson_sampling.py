from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from newsreclib.data.components.batch import RecommendationBatch


class ThompsonSamplingMixin:
    """Mixin class that adds Thompson sampling functionality to recommendation models."""

    def __init__(
        self,
        ts_pseudocount: int = 10,
        ts_mode: str = None,  # "category", "embed", or "resample"
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ts_pseudocount = ts_pseudocount
        self.ts_mode = ts_mode
        assert ts_mode in ["category", "embed", "resample", None], f"Invalid ts_mode: {ts_mode}"
        print("TS PSEUDOCOUNT", self.ts_pseudocount)
        print("TS MODE", self.ts_mode)

    def _apply_thompson_sampling(
        self,
        scores: torch.Tensor,
        batch: RecommendationBatch,
        cand_embeds: torch.Tensor,
        cand_mask: torch.Tensor,
        hist_embeds: torch.Tensor,
        hist_mask: torch.Tensor,
        user_encoder_kwargs: dict
    ) -> torch.Tensor:
        """Apply Thompson sampling to the scores.

        Args:
            scores: Raw scores from the model [num_users, max_candidates]
            mask: Boolean mask indicating valid candidates [num_users, max_candidates]
            batch: The batch data containing candidate and history information

        Returns:
            torch.Tensor: Thompson sampled probabilities [num_users, max_candidates]
        """
        scores += (~cand_mask).float() * -1e9
        probs = F.softmax(scores, dim=-1)

        pseudocount = self.ts_pseudocount
        bonus = cand_mask.float()
        bonus[bonus == 0.0] = 1e-6

        if self.ts_mode == "resample":
            # Basic Thompson sampling without ICL
            alpha = (pseudocount * probs + bonus).float()
            beta = (pseudocount * (1 - probs) + bonus).float()
            prior = torch.distributions.Beta(alpha, beta)
            new_probs = prior.rsample()
            return new_probs

        if self.ts_mode == "category":
            # Category-based implementation
            # For candidate articles: shape [num_users, max_candidates]
            cand_category, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"], fill_value=-1)
            # For history articles: shape [num_users, max_history]
            hist_category, hist_mask = to_dense_batch(batch["x_hist"]["category"], batch["batch_hist"], fill_value=-1)

            num_users = probs.shape[0]
            max_candidates = probs.shape[1]
            sampled_probs = []

            for i in range(num_users):
                # Candidate info for user i
                cat_row = cand_category[i]       # candidate categories (dense)
                prob_row = probs[i]              # candidate probabilities (dense)
                bonus_row = bonus[i]
                mask_row = cand_mask[i]               # valid candidate positions

                # --- Initialize per-category Beta parameters for this user ---
                beta_params = {}
                # Consider only valid candidate positions
                valid_cat = cat_row[mask_row]
                unique_cats = torch.unique(valid_cat)
                for c in unique_cats:
                    # Select candidates belonging to category c
                    indices = (cat_row == c) & mask_row
                    sum_prob = prob_row[indices].sum()
                    sum_bonus = bonus_row[indices].sum()
                    init_alpha = pseudocount * sum_prob + sum_bonus
                    init_beta = pseudocount * ((1 - prob_row[indices]).sum()) + sum_bonus
                    beta_params[int(c.item())] = [init_alpha, init_beta]

                # --- Update Beta parameters using the user's history ---
                hist_row = hist_category[i]      # history categories for user i
                hist_mask_row = hist_mask[i]
                valid_hist = hist_row[hist_mask_row]
                for c in valid_hist:
                    cat_val = int(c.item())
                    # Only update if this category appears among the candidates
                    if cat_val in beta_params:
                        beta_params[cat_val][0] = beta_params[cat_val][0] + 1

                # --- Sample from each per-category Beta distribution ---
                sampled_values = {}
                for cat_val, (alpha_val, beta_val) in beta_params.items():
                    dist = torch.distributions.Beta(alpha_val, beta_val)
                    sampled_val = dist.rsample()
                    sampled_values[cat_val] = sampled_val

                # --- Assign the sampled Beta value to each candidate article ---
                new_prob_row = torch.zeros_like(prob_row)
                for j in range(max_candidates):
                    if mask_row[j]:
                        cat_val = int(cat_row[j].item())
                        # If for some reason the candidate's category is missing, default to 0
                        new_prob_row[j] = sampled_values.get(cat_val, torch.tensor(0.0, device=prob_row.device))
                    else:
                        new_prob_row[j] = 0.0

                sampled_probs.append(new_prob_row)

            new_probs = torch.stack(sampled_probs, dim=0)

        elif self.ts_mode == "embed":
            # Initialize alpha and beta parameters for each candidate article
            alpha = pseudocount * probs + bonus  # [num_users, max_candidates]
            beta = pseudocount * (1 - probs) + bonus  # [num_users, max_candidates]

            # Flatten arguments so that we can isolate effect of single history
            num_users, max_hist, embed_dim = hist_embeds.shape
            user_encoder_kwargs['hist_news_vector'] = hist_embeds
            for k, v in user_encoder_kwargs.items():
                if k == 'mask_hist':  # [num_users, max_history]
                    # Consider all articles so we don't get nans, mask later
                    v = torch.ones(v.shape[0] * v.shape[1], 1, dtype=torch.bool)
                    v = v.to(hist_embeds.device)  # [num_users * max_history, 1]
                elif k == 'mask_cand':  # [num_users, max_candidates]
                    v = v.repeat(max_hist, 1) # [num_users * max_history, max_candidates]
                elif k == 'cand_news_vector':  # [num_users, max_candidates, embed_dim]
                    v = v.repeat(max_hist, 1, 1)  # [num_users * max_history, max_candidates, embed_dim]
                elif k == 'hist_news_vector':  # [num_users, max_history, mebed_dim]
                    v = v.reshape(-1, v.shape[-1]).unsqueeze(1)  # [num_users * max_history, 1, embed_dim]
                elif k == 'projected_users':  # [num_users, embed_dim]
                    v = v.repeat(max_hist, 1)  # [num_users * max_history, embed_dim]
                else:  # ???
                    raise ValueError(f"{k}, {v.shape} unexpected")
                user_encoder_kwargs[k] = v

            # Process history through user encoder, reshape appropriately
            hist_features = self.user_encoder(**user_encoder_kwargs)
            if len(hist_features.shape) == 3:
                # [num_users * max_history, max_candidates, embed_dim]
                # Sum across candidate dimension and take mask-weighted mean
                mean_mask = cand_mask.sum(1).repeat(max_hist).unsqueeze(1)
                hist_features = hist_features.sum(1) / mean_mask

            # [num_users * max_history, embed_dim] -> [num_users, max_history, embed_dim]
            hist_features = hist_features.reshape(hist_embeds.shape[0], hist_embeds.shape[1], -1)

            # Compute similarities between candidates and history items for each user
            # [num_users, max_candidates, embed_dim] x [num_users, embed_dim, max_history]
            sims = torch.matmul(cand_embeds, hist_features.transpose(-2, -1))

            # Apply masks to zero out invalid candidates and history items
            sims = sims * hist_mask.unsqueeze(1)  # [num_users, max_candidates, max_history]
            sims = sims * cand_mask.unsqueeze(-1)  # Also mask out invalid candidates

            # Take mean similarity across history dimension
            sim_sums = sims.sum(dim=-1) / hist_mask.sum(dim=-1).unsqueeze(-1)
            sim_sums = F.softmax(sim_sums, dim=-1)  # [num_users, max_candidates]

            # Scale by pseudocount factor and update alpha/beta parameters
            pseudocount = hist_mask.sum(dim=-1).unsqueeze(-1)
            alpha += self.ts_pseudocount * pseudocount * sim_sums
            beta += self.ts_pseudocount * pseudocount * (1 - sim_sums)
            prior = torch.distributions.Beta(alpha, beta)
            return prior.rsample()

    def apply_thompson_sampling(
        self,
        scores: torch.Tensor,
        batch: RecommendationBatch,
        cand_embeds: torch.Tensor,
        cand_mask: torch.Tensor,
        hist_embeds: torch.Tensor,
        hist_mask: torch.Tensor,
        user_encoder_kwargs: dict
    ) -> torch.Tensor:
        """Wraps the forward pass with Thompson sampling if enabled.

        Args:
            scores: Raw scores from the model [num_users, max_candidates]
            mask: Boolean mask indicating valid candidates [num_users, max_candidates]
            batch: The batch data containing candidate and history information 
            cand_embeds: Candidate news embeddings [num_users, max_candidates, embed_dim]
            hist_embeds: History news embeddings [num_users, max_history, embed_dim]
            hist_mask: Boolean mask indicating valid history items [num_users, max_history]

        Returns:
            torch.Tensor: Original scores or Thompson sampled probabilities
        """
        if self.ts_mode is not None and self.ts_pseudocount != 0:
            return self._apply_thompson_sampling(
                scores, batch, 
                cand_embeds, cand_mask, 
                hist_embeds, hist_mask,
                user_encoder_kwargs
            )
        return scores 
