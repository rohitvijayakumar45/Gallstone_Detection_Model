import logging
print("1")
from src.models.rf_detr_trainer import RFDETRTrainer
print("2")
import torch
print("3")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("4. Initializing trainer...")
    base_config = {
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-4,
        'lr_encoder': 1e-5,
        'weight_decay': 1e-4,
        'clip_max_norm': 0.1,
        'output_dir': 'runs/detect/rfdetr_test',
        'wandb': False,
        'patience': 10,
        'num_workers': 0,
    }
    trainer = RFDETRTrainer(base_config)
    print("5. Built trainer. Starting build_model...")
    
    try:
        model = trainer.build_model("small")
        print("6. Model built. Calling train...")
        
        train_args = {
            "dataset_dir": "dataset_final_resplit",
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "num_workers": 0,
            "wandb": False,
            "dataset_file": "yolo"
        }
        
        model.train(**train_args)
        print("7. Training completed successfully.")
    except Exception as e:
        print(f"Failed: {e}")
