from collections import defaultdict
from typing import Dict, Any
import json
import torch
from torchmetrics.classification import AUROC
from newsreclib.metrics.diversity import Diversity
from tqdm import tqdm


class PerUserMetricsMixin:
    """Mixin class that adds per-user metrics computation functionality."""

    def __init__(self, save_metrics: bool = True, metrics_fpath: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_metrics = save_metrics
        self.metrics_fpath = metrics_fpath
        # Initialize diversity metrics for categories and sentiments
        self.category_diversity = None
        self.sentiment_diversity = None

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
        
        # Process first 5% of total datapoints
        total_datapoints = len(preds)
        processed_points = 0
        max_points = total_datapoints
        
        for user_idx in tqdm(unique_users, desc="Computing per-user metrics"):
            user_mask = cand_indexes == user_idx
            num_user_points = user_mask.sum().item()
            
            # Check if we've exceeded our 5% threshold
            if processed_points + num_user_points > max_points:
                break
                
            user_preds = preds[user_mask]
            user_targets = targets[user_mask]
            user_target_categories = target_categories[user_mask]
            user_target_sentiments = target_sentiments[user_mask]
            
            # Create a fresh AUROC instance for this user
            try:
                auroc = AUROC(task="binary")
                user_auc = auroc(user_preds.float(), user_targets.long()).item()
            except Exception:
                user_auc = 0.5  # Default value if computation fails
            
            user_rec_metrics = {
                "auc": user_auc,
            }
            
            # Add diversity metrics
            for k in top_k_list:
                if k > len(user_preds):
                    continue
                
                # Initialize diversity metrics for this k
                category_diversity = Diversity(num_classes=num_categ_classes, top_k=k)
                sentiment_diversity = Diversity(num_classes=num_sent_classes, top_k=k)
                
                # Create sorted indices based on predictions
                _, indices = torch.sort(user_preds, descending=True)
                top_k_indices = indices[:k]
                
                # Update and compute category diversity
                category_diversity.update(
                    preds=user_preds[top_k_indices], 
                    target=user_target_categories[top_k_indices], 
                    indexes=torch.zeros(k, dtype=torch.long)  # Give each item a unique index
                )
                categ_div = category_diversity.compute().item()
                
                # Update and compute sentiment diversity
                sentiment_diversity.update(
                    preds=user_preds[top_k_indices], 
                    target=user_target_sentiments[top_k_indices], 
                    indexes=torch.zeros(k, dtype=torch.long)  # Give each item a unique index
                )
                sent_div = sentiment_diversity.compute().item()
                
                # Reset states
                category_diversity.reset()
                sentiment_diversity.reset()
                    
                user_rec_metrics[f"categ_div@{k}"] = categ_div
                user_rec_metrics[f"sent_div@{k}"] = sent_div
            
            # Store metrics for this user
            user_id = user_ids[user_idx].item()
            user_rec_metrics["num_articles"] = num_user_points
            per_user_metrics[user_id] = user_rec_metrics
            
            processed_points += num_user_points

        return per_user_metrics

    def save_per_user_metrics(self, metrics: Dict[str, Dict[str, float]], fpath: str) -> None:
        """Save per-user metrics to a JSON file.

        Args:
            metrics: Dictionary mapping user IDs to their metrics
            fpath: Path where to save the metrics
        """
        with open(fpath, 'w') as f:
            json.dump(metrics, f, indent=2) 

