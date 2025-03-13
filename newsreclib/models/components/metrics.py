from collections import defaultdict
from typing import Dict, Any
import json
import torch
from torchmetrics.classification import AUROC
from newsreclib.metrics.diversity import Diversity


class PerUserMetricsMixin:
    """Mixin class that adds per-user metrics computation functionality."""

    def __init__(self, save_metrics: bool = True, metrics_fpath: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_metrics = save_metrics
        self.metrics_fpath = metrics_fpath

    def compute_per_user_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_categories: torch.Tensor,
        target_sentiments: torch.Tensor,
        cand_indexes: torch.Tensor,
        user_ids: torch.Tensor,
        num_categ_classes: int,
        num_sent_classes: int,
        top_k_list: list,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each individual user.

        Args:
            preds: Model predictions
            targets: Ground truth labels
            target_categories: Category labels for candidates
            target_sentiments: Sentiment labels for candidates
            cand_indexes: Index mapping for candidates to users
            user_ids: User IDs
            num_categ_classes: Number of category classes
            num_sent_classes: Number of sentiment classes
            top_k_list: List of k values for top-k metrics

        Returns:
            Dictionary mapping user IDs to their metrics
        """
        per_user_metrics = defaultdict(dict)
        unique_users = torch.unique(cand_indexes)
        
        for user_idx in unique_users:
            user_mask = cand_indexes == user_idx
            user_preds = preds[user_mask]
            user_targets = targets[user_mask]
            user_target_categories = target_categories[user_mask]
            user_target_sentiments = target_sentiments[user_mask]
            
            # Compute recommendation metrics for this user
            user_rec_metrics = {
                "auc": AUROC(task="binary", num_classes=2)(user_preds, user_targets).item(),
            }
            
            # Add diversity metrics
            for k in top_k_list:
                # Create proper indexes tensor of type long
                indexes = torch.zeros(len(user_preds), dtype=torch.long)
                categ_div = Diversity(num_classes=num_categ_classes, top_k=k)(
                    user_preds, user_target_categories, indexes
                ).item()
                sent_div = Diversity(num_classes=num_sent_classes, top_k=k)(
                    user_preds, user_target_sentiments, indexes
                ).item()
                user_rec_metrics[f"categ_div@{k}"] = categ_div
                user_rec_metrics[f"sent_div@{k}"] = sent_div
            
            # Store metrics for this user
            user_id = user_ids[user_idx].item()
            per_user_metrics[user_id] = user_rec_metrics

        return per_user_metrics

    def save_per_user_metrics(self, metrics: Dict[str, Dict[str, float]], fpath: str) -> None:
        """Save per-user metrics to a JSON file.

        Args:
            metrics: Dictionary mapping user IDs to their metrics
            fpath: Path where to save the metrics
        """
        with open(fpath, 'w') as f:
            json.dump(metrics, f, indent=2) 
