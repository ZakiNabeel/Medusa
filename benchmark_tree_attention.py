import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from medusa_tree_attention import medusa_generate_tree_attention  # <-- Fixed import here!

# --- Step 1: Define the Medusa wrapper ---
class GPT2Medusa(nn.Module):
    def __init__(self, base_model_name="gpt2", num_heads=3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_model = self.base_model.cuda()
        self.hidden_size = self.base_model.config.n_embd
        self.vocab_size = self.base_model.config.vocab_size
        self.num_heads = num_heads

        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            for _ in range(num_heads)
        ])
        
    @property
    def heads(self):
        return self.medusa_heads

# --- Step 2: Initialize Model and Load Weights ---
print("Loading model and weights...")
model_wrapper = GPT2Medusa(num_heads=3)

# Load your specific 5000-batch checkpoint
checkpoint_path = '/content/drive/MyDrive/medusa_project/trained_heads_5000.pt'
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']

# Strip 'heads.' prefix 
new_state_dict = {k.replace('heads.', ''): v for k, v in state_dict.items()}
model_wrapper.medusa_heads.load_state_dict(new_state_dict)
model_wrapper.medusa_heads.cuda()
model_wrapper.medusa_heads.eval()

base_model = model_wrapper.base_model
tokenizer = model_wrapper.tokenizer

# --- Step 3: Run the Generation Loop ---
prompt = "The future of artificial intelligence is"
max_new_tokens = 50

print(f"\nStarting generation for prompt: '{prompt}'")
print("-" * 50)

# Run with Entropy Pruning ON
results = medusa_generate_tree_attention(
    prompt=prompt,
    model=base_model,
    medusa_heads=model_wrapper, 
    tokenizer=tokenizer,
    max_new_tokens=max_new_tokens,
    verbose=True, 
    entropy_threshold=4.0, 
    enable_entropy_pruning=True 
)

print("\n" + "=" * 50)
print("GENERATION COMPLETE")
print("=" * 50)
print(f"Generated Text:\n{results['text']}\n")
print("--- Performance Metrics ---")
print(f"Total Tokens: {results['total_tokens']}")
print(f"Time Elapsed: {results['time_seconds']:.2f} seconds")
print(f"Tokens Per Second: {results['tokens_per_second']:.2f}")
print(f"Avg Step Time: {results['avg_step_time_ms']:.2f} ms")
print(f"Avg Accepted Per Step: {results['avg_accepted_per_step']:.2f}")
print(f"Acceptance Rate: {results['acceptance_rate'] * 100:.1f}%")
print(f"Avg Pruned Nodes Per Step: {results['avg_pruned_nodes_per_step']:.1f}")
print(f"Pruning Rate: {results['pruning_rate'] * 100:.1f}%")
print(f"Avg Entropy Kernel Time: {results['entropy_calc_avg_ms']:.3f} ms")
