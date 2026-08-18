from __future__ import annotations

import json
from pathlib import Path
import torch
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer, EOS_ID

def main():
    print("Initializing Stage 1 Benchmark Suite...")
    
    tokenizer = FodciTokenizer()
    checkpoint_path = Path("/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-tiny-v1.pt")
    
    config = ModelConfig()
    model = FodciModel(config)
    
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)
            
    model.eval()
    
    benchmark_questions = [
        "What is a Backend in web development?",
        "What is the role of HTTP in backend engineering?",
        "What is the difference between GET and POST?",
        "What does HTTP status code 200 mean?",
        "What does HTTP status code 404 mean?",
        "What does HTTP status code 500 mean?",
        "What is a REST API?",
        "What is JSON?",
        "What is the difference between authentication and authorization?",
        "What is middleware?"
    ]
    
    results = []
    print("\nRunning Baseline Evaluation on Stage 1 Questions:")
    for i, q in enumerate(benchmark_questions):
        prompt = f"### Instruction\n{q}\n\n### Input\nNone\n\n### Response\n"
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            generated = x
            for _ in range(32):
                logits = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated = torch.cat([generated, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
                if next_token == EOS_ID:
                    break
        response_text = tokenizer.decode(generated[0].tolist()[len(input_ids):])
        results.append({"question": q, "response": response_text})
        print(f"[{i+1}] Q: {q}")
        print(f"    A: {response_text[:100]}...\n")
        
    output_path = Path("/home/ubuntu/backend-ai/artifacts/baseline_stage1_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Benchmark results saved to {output_path}")

if __name__ == "__main__":
    main()
