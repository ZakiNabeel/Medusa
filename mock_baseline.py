import numpy as np
import time

# 1. Simulate the Data
# Let's pretend we have a vocabulary of 32,000 words
vocab_size = 32000
# Let's pretend the Medusa tree generated 10 candidate nodes (guesses)
num_nodes = 10 

# Fake draft tokens (the words Medusa guessed)
draft_tokens = np.random.randint(0, vocab_size, size=(num_nodes,))

# Fake base logits (the actual probabilities from the big model)
base_logits = np.random.rand(num_nodes, vocab_size)

# 2. The Sequential Verification (The part we want to speed up)
def sequential_verification(drafts, logits):
    is_match = np.zeros(num_nodes, dtype=np.int32)
    
    # Loop through every node in the tree sequentially
    for i in range(num_nodes):
        # Find the word the base model thought was most likely
        best_token = np.argmax(logits[i])
        # Did the draft guess correctly?
        if best_token == drafts[i]:
            is_match[i] = 1
            
    return is_match

# 3. Benchmark it
start_time = time.time()
matches = sequential_verification(draft_tokens, base_logits)
end_time = time.time()

print(f"Sequential Verification took: {(end_time - start_time) * 1000:.4f} ms")
print(f"Matches found: {matches}")