import logging
import os

import numpy as np

from crawler import Crawler

logger = logging.getLogger(__name__)


def log_time(elapsed_time: int, remaining_time: int) -> None:
    remaining_time_hours, remaining_time_remainder = divmod(remaining_time, 3600)
    remaining_time_minutes, remaining_time_seconds = divmod(remaining_time_remainder, 60)
    if remaining_time_hours > 0:
        remaining_time_str = f"{int(remaining_time_hours)}h {int(remaining_time_minutes)}m {int(remaining_time_seconds)}s"
    elif remaining_time_minutes > 0:
        remaining_time_str = f"{int(remaining_time_minutes)}m {int(remaining_time_seconds)}s"
    else:
        remaining_time_str = f"{int(remaining_time_seconds)}s"
    logger.info(
        f"Elapsed time this iter: {elapsed_time:.2f}s, remaining time: {remaining_time_str}"
    )


def eval_and_plot(args, crawler: Crawler) -> None:
    import scipy.stats as stats
    from matplotlib import pyplot as plt

    logger.info("Initializing seed docs")
    docids = []
    if args.seed_docs_file is None:
        raise ValueError("Seed docs file must be provided")
    with open(args.seed_docs_file, "r") as fin:
        for line in fin:
            docids.append(line.strip())

    crawler.wandb_logger = None
    results = crawler.get_scores_for_docs(docids)
    # Plot correlations between different rating methods
    annotation_labels = args.plots
    num_raters = len(annotation_labels)
    fig, axes = plt.subplots(num_raters, num_raters, figsize=(18, 18))
    fig.suptitle("Correlations between different rating methods")

    for i, rater1 in enumerate(annotation_labels):
        for j, rater2 in enumerate(annotation_labels):
            ax = axes[i, j]
            if i == j:
                # Plot score distribution
                scores = [doc.annotations[rater1] for doc in results]
                ax.hist(scores, bins=100, density=True, alpha=0.6, color="g")
                if "normalized" in rater1:
                    ax.set_xlim(-3, 3)
                ax.set_xlabel(rater1)
                ax.set_ylabel("Percentage (%)")
                ax.set_title(f"{rater1} Score Distribution")
            else:
                rater1_name = rater1
                rater2_name = rater2
                rater1_scores = [doc.annotations[rater1_name] for doc in results]
                rater2_scores = [doc.annotations[rater2_name] for doc in results]
                corr, _ = stats.spearmanr(rater1_scores, rater2_scores)
                logger.info(f"{rater1_name} vs {rater2_name}, spearman corr: {corr}")

                ax.scatter(rater1_scores, rater2_scores, label="Data points")

                # Fit line
                m, b = np.polyfit(rater1_scores, rater2_scores, 1)
                ax.plot(
                    rater1_scores, np.array(rater1_scores) * m + b, color="red", label="Fit line"
                )

                # Calculate mean y value for each bin
                bins = np.linspace(min(rater1_scores), max(rater1_scores), 100)
                bin_means = stats.binned_statistic(
                    rater1_scores, rater2_scores, statistic="mean", bins=bins
                )[0]
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.plot(bin_centers, bin_means, color="blue", label="Mean values")

                ax.set_xlabel(rater1_name)
                ax.set_ylabel(rater2_name)
                if rater2_name == "length":
                    ax.set_ylim(0, 30000)
                ax.set_title(f"Spearman corr: {corr:.2f}")
                ax.legend()

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(args.output_dir, "correlations.png"))
