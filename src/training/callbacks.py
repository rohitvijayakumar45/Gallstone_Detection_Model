class ExperimentTracker:
    """Dual MLflow/WandB tracker with optional dependency fallback."""

    def __init__(self, experiment_name, use_wandb=True, use_mlflow=True):
        self.mlflow = None
        self.wandb = None
        if use_mlflow:
            try:
                import mlflow

                mlflow.set_experiment(experiment_name)
                mlflow.start_run()
                self.mlflow = mlflow
            except Exception:
                self.mlflow = None
        if use_wandb:
            try:
                import wandb

                wandb.init(project=experiment_name)
                self.wandb = wandb
            except Exception:
                self.wandb = None

    def log_metrics(self, metrics, step):
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_image(self, tag, image):
        if self.wandb:
            self.wandb.log({tag: self.wandb.Image(image)})

    def log_confusion_matrix(self, cm):
        if self.wandb:
            self.wandb.log(
                {
                    "confusion_matrix": self.wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=cm["true"],
                        preds=cm["pred"],
                        class_names=["no_stone", "gallstone"],
                    )
                }
            )
