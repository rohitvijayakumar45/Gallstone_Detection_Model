import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    from src.models.rf_detr_trainer import RFDETRTrainer
    from pathlib import Path
    import torch

    # The winning hyperparameters from Trial 10 (Large model)
    config = {
        "lr": 5.61e-05,
        "lr_encoder": 7.97e-05,
        "weight_decay": 0.00157,
        "clip_max_norm": 0.319,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "epochs": 150,  # Long training
        "num_workers": 2, # Adjusted to 2 to prevent OOM
        "output_dir": "runs/detect/rfdetr_large_production_final",
        "dataset_file": "yolo",
        "eval_interval": 1,
        "patience": 20, # Higher patience for long run
        "progress_bar": "tqdm",
        "wandb": False,
        "mlflow": False,
    }

    print("============================================================")
    print("RF-DETR Production Training (Large)")
    print(f"Hyperparameters: {config}")
    print("============================================================")

    # Ensure output dir exists
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    trainer = RFDETRTrainer(config)
    
    # Train on the final resplit dataset
    # Note: Using 'large' model size
    try:
        trainer.train("dataset_final_resplit", model_size="large")
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nTIP: If you see 'DefaultCPUAllocator: not enough memory', reduce num_workers to 2 or 0.")
