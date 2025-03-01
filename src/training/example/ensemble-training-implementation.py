import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import faiss
import torch.nn.functional as F

# Load models
def load_models():
    # Primary model
    primary_tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    primary_model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder2-3b", 
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    # Secondary model
    secondary_tokenizer = AutoTokenizer.from_pretrained("google/codegemma-2b")
    secondary_model = AutoModelForCausalLM.from_pretrained(
        "google/codegemma-2b", 
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    # Evaluator model
    evaluator_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    evaluator_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", 
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    # IR-NLP model
    ir_nlp_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    ir_nlp_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    
    return {
        "primary": (primary_model, primary_tokenizer),
        "secondary": (secondary_model, secondary_tokenizer),
        "evaluator": (evaluator_model, evaluator_tokenizer),
        "ir_nlp": (ir_nlp_model, ir_nlp_tokenizer)
    }

# Entropy calculation functions
def calculate_token_entropy(logits):
    """
    Calculate entropy from model logits for each token position
    """
    probs = F.softmax(logits, dim=-1)
    token_entropies = torch.zeros(probs.shape[0], probs.shape[1])
    
    for i in range(probs.shape[0]):  # Batch dimension
        for j in range(probs.shape[1]):  # Sequence dimension
            token_entropies[i, j] = entropy(probs[i, j].detach().cpu().numpy())
    
    return token_entropies

def calculate_sequence_entropy(token_entropies):
    """
    Calculate overall sequence entropy from token entropies
    """
    return token_entropies.mean(dim=1)

def evaluate_ensemble_uncertainty(primary_output, secondary_output):
    """
    Calculate disagreement entropy between primary and secondary models
    """
    # Convert outputs to embeddings for comparison
    primary_embedding = get_embedding(primary_output)
    secondary_embedding = get_embedding(secondary_output)
    
    # Calculate cosine similarity
    similarity = F.cosine_similarity(primary_embedding, secondary_embedding, dim=0)
    
    # Convert similarity to a disagreement score (higher means more disagreement)
    disagreement = 1 - similarity
    
    return disagreement.item()

# KNN implementation
class KNNIndex:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        # Using FAISS for efficient KNN search
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.examples = []
        self.metadata = []
    
    def add_example(self, text, embedding, metadata=None):
        """Add an example to the KNN index"""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embedding.shape[0]}")
        
        self.index.add(np.array([embedding.detach().cpu().numpy()]).astype('float32'))
        self.examples.append(text)
        self.metadata.append(metadata or {})
    
    def search(self, query_embedding, k=5):
        """Search for nearest neighbors"""
        query_embedding_np = np.array([query_embedding.detach().cpu().numpy()]).astype('float32')
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    'text': self.examples[idx],
                    'distance': distances[0][i],
                    'metadata': self.metadata[idx]
                })
        
        return results

# Embedding function
def get_embedding(text, models, max_length=512):
    """Get embedding for text using the IR-NLP model"""
    ir_model, ir_tokenizer = models["ir_nlp"]
    
    # Move model to appropriate device and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_model = ir_model.to(device)
    ir_model.eval()
    
    # Tokenize and get embedding
    inputs = ir_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        outputs = ir_model(**inputs)
    
    # Use mean pooling to get a single embedding vector
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Normalize embedding
    embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding[0]  # Return the first (and only) embedding

# Main ensemble training function
def ensemble_training_loop(prompt, models, knn_index, feedback_history=None):
    """
    Ensemble training with entropy calculations and KNN logic
    """
    feedback_history = feedback_history or []
    primary_model, primary_tokenizer = models["primary"]
    secondary_model, secondary_tokenizer = models["secondary"] 
    evaluator_model, evaluator_tokenizer = models["evaluator"]
    
    # Generate from primary model
    primary_inputs = primary_tokenizer(prompt, return_tensors="pt").to(primary_model.device)
    
    # Get logits for entropy calculation
    with torch.no_grad():
        primary_outputs = primary_model(**primary_inputs, output_hidden_states=True, return_dict=True)
        primary_logits = primary_outputs.logits
        
        # Generate actual output
        primary_generation = primary_model.generate(
            **primary_inputs, 
            max_length=256,
            do_sample=True,
            temperature=0.7
        )
    
    primary_text = primary_tokenizer.decode(primary_generation[0], skip_special_tokens=True)
    
    # Calculate entropy from primary model
    token_entropies = calculate_token_entropy(primary_logits)
    sequence_entropy = calculate_sequence_entropy(token_entropies)
    
    # Get embedding for primary output
    primary_embedding = get_embedding(primary_text, models)
    
    # Use KNN to find similar examples
    similar_examples = knn_index.search(primary_embedding, k=3)
    
    # Prepare context for secondary model with KNN results
    knn_context = ""
    for i, example in enumerate(similar_examples):
        knn_context += f"Reference example {i+1}:\n{example['text']}\n\n"
    
    # Create secondary prompt with KNN context
    secondary_prompt = f"""
    Task: Enhance and improve the following code.
    
    Reference examples:
    {knn_context}
    
    Original code:
    {primary_text}
    
    Enhanced code:
    """
    
    # Generate from secondary model
    secondary_inputs = secondary_tokenizer(secondary_prompt, return_tensors="pt").to(secondary_model.device)
    with torch.no_grad():
        secondary_generation = secondary_model.generate(
            **secondary_inputs,
            max_length=512,
            do_sample=True,
            temperature=0.5
        )
    
    secondary_text = secondary_tokenizer.decode(secondary_generation[0], skip_special_tokens=True)
    
    # Calculate ensemble uncertainty
    ensemble_uncertainty = evaluate_ensemble_uncertainty(primary_text, secondary_text)
    
    # Prepare evaluator prompt
    evaluator_prompt = f"""
    Task: Evaluate the quality of the enhanced code compared to the original.
    
    Original code:
    {primary_text}
    
    Enhanced code:
    {secondary_text}
    
    Evaluation criteria:
    1. Correctness: Is the code functionally correct?
    2. Efficiency: Is the enhanced code more efficient?
    3. Readability: Is the code more readable and maintainable?
    4. Best practices: Does it follow coding best practices?
    
    Provide a score for each criterion (1-10) and an overall score (1-10).
    """
    
    # Generate evaluation
    evaluator_inputs = evaluator_tokenizer(evaluator_prompt, return_tensors="pt").to(evaluator_model.device)
    with torch.no_grad():
        evaluator_generation = evaluator_model.generate(
            **evaluator_inputs,
            max_length=512,
            do_sample=False
        )
    
    evaluation_text = evaluator_tokenizer.decode(evaluator_generation[0], skip_special_tokens=True)
    
    # Extract scores (this is a simplistic implementation, would need regex in real use)
    # In practice, you'd want to parse the scores more robustly
    evaluation_score = 7.5  # Placeholder score
    
    # Update KNN index with this example if it's good
    if evaluation_score > 7.0:
        knn_index.add_example(
            secondary_text, 
            get_embedding(secondary_text, models),
            metadata={
                'score': evaluation_score,
                'entropy': sequence_entropy.item(),
                'uncertainty': ensemble_uncertainty
            }
        )
    
    # Store feedback for reinforcement learning
    feedback_entry = {
        'prompt': prompt,
        'primary_output': primary_text,
        'secondary_output': secondary_text,
        'evaluation': evaluation_text,
        'score': evaluation_score,
        'entropy': sequence_entropy.item(),
        'uncertainty': ensemble_uncertainty
    }
    feedback_history.append(feedback_entry)
    
    return {
        'primary_output': primary_text,
        'secondary_output': secondary_text,
        'evaluation': evaluation_text,
        'score': evaluation_score,
        'entropy': sequence_entropy.item(),
        'uncertainty': ensemble_uncertainty,
        'feedback_history': feedback_history,
        'similar_examples': similar_examples
    }

# Example usage
if __name__ == "__main__":
    models = load_models()
    knn_index = KNNIndex(embedding_dim=384)  # Dimension for all-MiniLM-L12-v2
    
    # Add some initial examples to KNN index
    initial_examples = [
        "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
    ]
    
    for example in initial_examples:
        embedding = get_embedding(example, models)
        knn_index.add_example(example, embedding)
    
    # Run a training iteration
    prompt = "Write a function to check if a string is a palindrome"
    result = ensemble_training_loop(prompt, models, knn_index)
    
    # Print results
    print(f"Primary model output entropy: {result['entropy']}")
    print(f"Ensemble uncertainty: {result['uncertainty']}")
    print(f"Evaluation score: {result['score']}")
