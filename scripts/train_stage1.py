from __future__ import annotations

from pathlib import Path
import torch
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.instructions import InstructionDatasetPipeline
from backend_ai.training.config import TrainingConfig
from backend_ai.training.trainer import FodciTrainer

def main():
    print("Starting Stage 1 Training & Pipeline Validation...")
    
    # 1. Setup Config & Tokenizer
    dataset_config = DatasetConfig(input_dir="/home/ubuntu/backend-ai/training_data/fundamentals")
    tokenizer = FodciTokenizer()
    pipeline = InstructionDatasetPipeline(dataset_config, tokenizer)
    
    # 2. Load examples
    examples = list(pipeline.iter_training_examples())
    print(f"Loaded {len(examples)} training token sequences from Stage 1 dataset.")
    
    if not examples:
        raise ValueError("No training examples generated!")
        
    # 3. Setup Model & Trainer
    model_config = ModelConfig()
    model = FodciModel(model_config)
    
    training_config = TrainingConfig(
        epochs=3,
        batch_size=2,
        learning_rate=5e-4,
        output_dir=Path("/home/ubuntu/backend-ai/artifacts/checkpoints/stage_1"),
        checkpoint_interval=1,
    )
    
    trainer = FodciTrainer(
        model=model,
        train_dataset=examples,
        config=training_config,
        model_version="fodci-stage1-v1",
    )
    
    # 4. Run Training
    print("Running trainer.train()...")
    result = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Global steps: {result.global_step}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.2f}s")
    for metrics in result.history:
        print(f"Epoch {metrics.epoch}: train_loss = {metrics.train_loss:.4f}, perplexity = {metrics.train_perplexity:.2f}")

if __name__ == "__main__":
    main()
