from collections import defaultdict
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import torch
from torchmetrics.classification import AUROC
from newsreclib.metrics.diversity import Diversity
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG


class PerUserMetricsMixin:
    """Mixin class that adds functionality for computing and saving per-user metrics.
    
    This mixin provides methods to compute individual metrics for each user during testing.
    It requires the base class to have certain metric templates defined.
    
    Attributes:
        ind_test_rec_metrics_template: Template for recommendation metrics per user
        ind_test_categ_div_metrics_template: Template for category diversity metrics per user
    """

    def __init__(self, save_metrics: bool = True, metrics_fpath: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_metrics = save_metrics
        self.metrics_fpath = metrics_fpath
        # Initialize diversity metrics for categories and sentiments
        self.category_diversity = None
        self.sentiment_diversity = None

    def setup_per_user_metrics(self, top_k_list: List[int], num_categ_classes: int) -> None:
        """Sets up the metric templates needed for per-user metric computation.
        
        Args:
            top_k_list: List of positions at which to compute rank-based metrics
            num_categ_classes: Number of category classes
        """
        # Create base metric collections that will be cloned for each user
        rec_metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
            }
        )

        categ_div_metrics_dict = {}
        for k in top_k_list:
            categ_div_metrics_dict["categ_div@" + str(k)] = Diversity(
                num_classes=num_categ_classes, top_k=k
            )
        categ_div_metrics = MetricCollection(categ_div_metrics_dict)

        # Store templates that will be cloned per user
        self.ind_test_rec_metrics_template = rec_metrics
        self.ind_test_categ_div_metrics_template = categ_div_metrics

    def compute_per_user_metrics(
        self,
        user_ids: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_categories: torch.Tensor,
        cand_indexes: torch.Tensor,
        cand_news_size: torch.Tensor,
        hist_news_size: torch.Tensor
    ) -> Dict[str, MetricCollection]:
        """Computes metrics for buckets of history sizes.
        
        Args:
            user_ids: Tensor of user IDs
            preds: Model predictions
            targets: Ground truth labels
            target_categories: Category labels for candidates
            cand_indexes: Candidate indexes
            cand_news_size: Candidate news size
            hist_news_size: History news size
        Returns:
            Dictionary mapping history size buckets to their metric collections
        """
        per_bucket_metrics = {}
        
        # Create N buckets based on history size range
        N = 25
        min_hist = hist_news_size.min().item()
        max_hist = hist_news_size.max().item()
        bucket_size = (max_hist - min_hist) / N
        
        # Handle edge case where all histories are same size
        if bucket_size == 0:
            bucket_size = 1
        print(min_hist, max_hist, bucket_size, N)
        
        for i in range(N):
            bucket_start = min_hist + i * bucket_size
            bucket_end = min_hist + (i + 1) * bucket_size
            
            # Get indices for all users in this bucket
            if i == N - 1:  # Include the max value in the last bucket
                bucket_idx = torch.where((hist_news_size >= bucket_start) & (hist_news_size <= bucket_end))[0]
            else:
                bucket_idx = torch.where((hist_news_size >= bucket_start) & (hist_news_size < bucket_end))[0]
            
            # Skip empty buckets
            if len(bucket_idx) == 0:
                continue
            print(f"BUCKET {i}: {len(bucket_idx)}, {bucket_start}-{bucket_end}")
                
            # Clone metric templates for this bucket
            ind_test_rec_metrics = self.ind_test_rec_metrics_template.clone()
            ind_test_categ_div_metrics = self.ind_test_categ_div_metrics_template.clone()
            
            # Compute metrics for this bucket
            ind_test_rec_metrics(
                preds[bucket_idx],
                targets[bucket_idx],
                **{"indexes": cand_indexes[bucket_idx]}
            )
            ind_test_categ_div_metrics(
                preds[bucket_idx],
                target_categories[bucket_idx], 
                cand_indexes[bucket_idx]
            )
            
            # Combine metrics with proper prefixes
            bucket_metrics = {}
            bucket_range = f"{bucket_start:.1f}-{bucket_end:.1f}"
            for name, value in ind_test_rec_metrics.compute().items():
                bucket_metrics[f"test_hist_bucket_{bucket_range}/{name}"] = value
            for name, value in ind_test_categ_div_metrics.compute().items():
                bucket_metrics[f"test_hist_bucket_{bucket_range}/{name}"] = value
                
            # Store metrics for this bucket
            per_bucket_metrics[bucket_range] = bucket_metrics
            
        return per_bucket_metrics

    def log_per_user_metrics(self, per_user_metrics: Dict[str, Dict[str, MetricCollection]]) -> None:
        """Logs the computed per-user metrics.
        
        Args:
            per_user_metrics: Dictionary of per-user metrics to log
        """
        for user_metrics in per_user_metrics.values():
            self.log_dict(
                user_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )

