from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from newsreclib.data.components.batch import RecommendationBatch


class ThompsonSamplingMixin:
    """Mixin class that adds Thompson sampling functionality to recommendation models."""

    def __init__(
        self,
        ts_pseudocount: int = 0,
        ts_icl: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ts_pseudocount = ts_pseudocount
        self.ts_icl = ts_icl
        print("TS PSEUDOCOUNT", self.ts_pseudocount)
        print("TS ICL", self.ts_icl)

    def _apply_thompson_sampling(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        batch: RecommendationBatch,
    ) -> torch.Tensor:
        """Apply Thompson sampling to the scores.

        Args:
            scores: Raw scores from the model [num_users, max_candidates]
            mask: Boolean mask indicating valid candidates [num_users, max_candidates]
            batch: The batch data containing candidate and history information

        Returns:
            torch.Tensor: Thompson sampled probabilities [num_users, max_candidates]
        """
        scores += (~mask).float() * -1e9
        probs = F.softmax(scores, dim=-1)

        pseudocount = self.ts_pseudocount
        bonus = mask.float()
        bonus[bonus == 0.0] = 1e-6

        if self.ts_icl:
            # --- Prepare dense category information ---
            # For candidate articles: shape [num_users, max_candidates]
            cand_category, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"], fill_value=-1)
            # For history articles: shape [num_users, max_history]
            hist_category, hist_mask = to_dense_batch(batch["x_hist"]["category"], batch["batch_hist"], fill_value=-1)
            print("-" * 80)
            print(cand_category.shape, batch['x_cand']['category'].shape, batch['batch_cand'].shape)
            print(hist_category.shape, batch['x_hist']['category'].shape, batch['batch_hist'].shape)
            print("-" * 80)

            num_users = probs.shape[0]
            max_candidates = probs.shape[1]
            sampled_probs = []

            for i in range(num_users):
                # Candidate info for user i
                cat_row = cand_category[i]       # candidate categories (dense)
                prob_row = probs[i]              # candidate probabilities (dense)
                bonus_row = bonus[i]
                mask_row = mask[i]               # valid candidate positions

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
                print('=' *80)
                print(hist_row)
                print(hist_mask_row)
                print(valid_hist)
                print('unique cats', unique_cats)
                print('valid cats', valid_cat)
                print('=' *80)
                for c in valid_hist:
                    cat_val = int(c.item())
                    # Only update if this category appears among the candidates
                    if cat_val in beta_params:
                        print(cat_val, beta_params[cat_val][0])
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

                print("~" * 80)
                print(beta_params, new_prob_row)
                print()
                print(cat_row.unique(return_counts=True))
                print("~" * 80)
                print()
            new_probs = torch.stack(sampled_probs, dim=0)

        else:
            # Just initialize the TS and do not fit to icl examples
            alpha = (pseudocount * probs + bonus).float()
            beta = (pseudocount * (1 - probs) + bonus).float()

            prior = torch.distributions.Beta(alpha, beta)
            new_probs = prior.rsample()

        return new_probs

    def _wrap_forward(self, scores: torch.Tensor, mask: torch.Tensor, batch: RecommendationBatch) -> torch.Tensor:
        """Wraps the forward pass with Thompson sampling if enabled.

        Args:
            scores: Raw scores from the model
            mask: Boolean mask indicating valid candidates
            batch: The batch data containing candidate and history information

        Returns:
            torch.Tensor: Original scores or Thompson sampled probabilities
        """
        if self.ts_pseudocount != 0:
            return self._apply_thompson_sampling(scores, mask, batch)
        return scores 