from pathlib import Path
from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.instructions import InstructionDatasetLoader

def main():
    config = DatasetConfig(input_dir="/home/ubuntu/backend-ai/training_data/fundamentals")
    loader = InstructionDatasetLoader(config)
    result = loader.load()
    
    print(f"Total examples found: {len(result.examples)}")
    print(f"Total issues found: {len(result.issues)}")
    
    for issue in result.issues:
        print(f"Issue in {issue.source_path}: {issue.reason}")
        
    if len(result.examples) > 0:
        first = result.examples[0]
        print("\nFirst example preview:")
        print(f"ID: {first.example_id}")
        print(f"Instruction: {first.instruction[:50]}...")
        print(f"Input: {first.input_text}")
        print(f"Response: {first.response[:50]}...")
        
    if len(result.issues) == 0 and len(result.examples) == 20:
        print("\nVerification SUCCESSFUL!")
    else:
        print("\nVerification FAILED!")

if __name__ == "__main__":
    main()
