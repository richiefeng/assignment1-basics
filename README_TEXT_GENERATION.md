# Text Generation with Transformer Language Models

This document explains how to use the text generation capabilities implemented in our Transformer Language Model training system.

## 🎯 **Overview**

The text generation system provides comprehensive functionality for generating text completions from trained language models, including:

- **Temperature Scaling**: Control randomness in generation
- **Top-p (Nucleus) Sampling**: Focus on high-probability tokens
- **Batch Generation**: Generate multiple completions simultaneously
- **Detailed Statistics**: Track generation quality and probabilities

## 🚀 **Core Functions**

### **1. Basic Text Generation**

```python
from cs336_basics.utils import decode

generated_text = decode(
    model=model,
    prompt="The future of artificial intelligence",
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=1.0,
    top_p=1.0,
    device="cpu"
)
```

**Parameters:**
- `model`: Trained TransformerLM model
- `prompt`: Input text to complete
- `tokenizer`: BPE or character tokenizer
- `max_new_tokens`: Maximum number of new tokens to generate
- `temperature`: Controls randomness (0.1-2.0, lower = more focused)
- `top_p`: Nucleus sampling threshold (0.1-1.0, lower = more focused)
- `device`: Device to run generation on

### **2. Batch Text Generation**

```python
from cs336_basics.utils import batch_decode

prompts = [
    "The future of artificial intelligence",
    "Once upon a time",
    "The best way to learn programming is"
]

generated_texts = batch_decode(
    model=model,
    prompts=prompts,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    device="cpu"
)
```

**Benefits:**
- Process multiple prompts simultaneously
- More efficient than individual generation
- Consistent generation parameters across prompts

### **3. Detailed Generation with Statistics**

```python
from cs336_basics.utils import generate_text_with_controls

result = generate_text_with_controls(
    model=model,
    prompt="The future of artificial intelligence",
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.7,
    device="cpu",
    verbose=True
)

print(f"Generated text: {result['generated_text']}")
print(f"Average token probability: {result['avg_probability']:.4f}")
print(f"Generation steps: {result['generation_steps']}")
```

**Returns:**
- `prompt`: Original input prompt
- `generated_text`: Generated completion
- `full_text`: Complete text (prompt + completion)
- `generated_tokens`: List of generated token IDs
- `token_probabilities`: Probability of each generated token
- `generation_steps`: Number of generation steps
- `avg_probability`: Average token probability
- `min_probability`: Minimum token probability
- `max_probability`: Maximum token probability

## 🎛️ **Generation Controls**

### **Temperature Scaling**

Temperature controls the randomness of text generation:

```python
# Low temperature (0.1-0.5): More focused, deterministic
generated_text = decode(model, prompt, tokenizer, temperature=0.3)

# Medium temperature (0.5-1.0): Balanced creativity
generated_text = decode(model, prompt, tokenizer, temperature=0.8)

# High temperature (1.0-2.0): More creative, diverse
generated_text = decode(model, prompt, tokenizer, temperature=1.5)
```

**How it works:**
- Lower temperature concentrates probability mass on high-probability tokens
- Higher temperature spreads probability more evenly
- Formula: `softmax(logits / temperature)`

### **Top-p (Nucleus) Sampling**

Top-p sampling focuses generation on high-probability tokens:

```python
# Low top-p (0.1-0.3): Very focused, few token choices
generated_text = decode(model, prompt, tokenizer, top_p=0.2)

# Medium top-p (0.3-0.7): Balanced focus
generated_text = decode(model, prompt, tokenizer, top_p=0.5)

# High top-p (0.7-1.0): More diverse, many token choices
generated_text = decode(model, prompt, tokenizer, top_p=0.9)
```

**How it works:**
- Sorts tokens by probability in descending order
- Selects smallest set of tokens whose cumulative probability ≥ p
- Renormalizes probabilities of selected tokens
- Prevents generation of low-probability tokens

### **Combined Controls**

Temperature and top-p can be used together for fine-grained control:

```python
# Focused and creative
generated_text = decode(model, prompt, tokenizer, temperature=1.5, top_p=0.3)

# Focused and deterministic
generated_text = decode(model, prompt, tokenizer, temperature=0.5, top_p=0.3)

# Creative and diverse
generated_text = decode(model, prompt, tokenizer, temperature=1.5, top_p=0.9)
```

## 🔧 **Usage Examples**

### **Basic Text Completion**

```python
import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import decode

# Load your trained model
model = TransformerLM(...)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Create tokenizer
tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# Generate text
prompt = "The future of artificial intelligence"
generated_text = decode(
    model=model,
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"Full text: {prompt + generated_text}")
```

### **Creative Writing with High Temperature**

```python
# Generate creative story continuations
story_prompts = [
    "Once upon a time, there was a magical forest",
    "The spaceship drifted through the asteroid field",
    "In the year 2050, robots had become"
]

for prompt in story_prompts:
    generated = decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=1.5,  # High creativity
        top_p=0.8         # Focus on good tokens
    )
    print(f"\nPrompt: {prompt}")
    print(f"Story: {generated}")
```

### **Focused Technical Writing with Low Temperature**

```python
# Generate precise technical explanations
tech_prompts = [
    "The main advantage of transformer models is",
    "To implement gradient descent, you need to",
    "The key difference between supervised and unsupervised learning is"
]

for prompt in tech_prompts:
    generated = decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=60,
        temperature=0.3,  # Low randomness
        top_p=0.5         # Very focused
    )
    print(f"\nPrompt: {prompt}")
    print(f"Explanation: {generated}")
```

### **Batch Generation for Multiple Prompts**

```python
# Generate completions for multiple prompts efficiently
prompts = [
    "The weather today is",
    "My favorite programming language is",
    "The best way to learn machine learning is"
]

generated_texts = batch_decode(
    model=model,
    prompts=prompts,
    tokenizer=tokenizer,
    max_new_tokens=40,
    temperature=1.0,
    top_p=0.9
)

for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
    print(f"\n{i+1}. Prompt: {prompt}")
    print(f"   Generated: {generated}")
```

### **Detailed Generation Analysis**

```python
# Get detailed statistics about generation quality
result = generate_text_with_controls(
    model=model,
    prompt="The future of artificial intelligence",
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.7,
    verbose=True  # Print generation progress
)

# Analyze generation quality
if result['avg_probability'] > 0.1:
    print("High-quality generation")
elif result['avg_probability'] > 0.05:
    print("Medium-quality generation")
else:
    print("Low-quality generation - consider adjusting parameters")

# Check for repetition
generated_tokens = result['generated_tokens']
if len(set(generated_tokens)) < len(generated_tokens) * 0.8:
    print("Warning: Some token repetition detected")
```

## 📊 **Generation Quality Analysis**

### **Probability Analysis**

```python
result = generate_text_with_controls(model, prompt, tokenizer, verbose=False)

# Analyze token probabilities
probs = result['token_probabilities']
print(f"Average probability: {result['avg_probability']:.4f}")
print(f"Probability range: [{result['min_probability']:.4f}, {result['max_probability']:.4f}]")

# Check for very low probability tokens
low_prob_tokens = [p for p in probs if p < 0.01]
if low_prob_tokens:
    print(f"Warning: {len(low_prob_tokens)} tokens with very low probability")
```

### **Length Analysis**

```python
# Check generation length
if result['generation_steps'] < max_new_tokens:
    print("Generation stopped early - may have hit end token or context limit")
else:
    print("Generation completed to maximum length")

# Check for context length issues
if result['generation_steps'] > 0:
    avg_length = len(result['generated_text']) / result['generation_steps']
    print(f"Average characters per token: {avg_length:.2f}")
```

## 🚨 **Common Issues and Solutions**

### **1. Repetitive Text**

**Symptoms:** Same phrases or tokens repeated
**Solutions:**
- Increase temperature (1.2-1.5)
- Lower top-p (0.3-0.5)
- Increase max_new_tokens for more variety

### **2. Low-Quality Output**

**Symptoms:** Nonsensical text, very low probabilities
**Solutions:**
- Lower temperature (0.3-0.7)
- Increase top-p (0.8-1.0)
- Check model training quality

### **3. Generation Stops Early**

**Symptoms:** Output much shorter than max_new_tokens
**Solutions:**
- Check for end token in vocabulary
- Verify context length settings
- Check for numerical issues

### **4. Memory Issues**

**Symptoms:** Out of memory errors during generation
**Solutions:**
- Reduce max_new_tokens
- Use smaller batch sizes
- Generate on CPU if GPU memory is limited

## 🔍 **Advanced Techniques**

### **Beam Search (Future Enhancement)**

```python
# Planned feature for better generation quality
def beam_search_decode(model, prompt, tokenizer, beam_size=5, max_new_tokens=100):
    # Implementation for beam search decoding
    pass
```

### **Conditional Generation**

```python
# Generate text with specific constraints
def conditional_decode(model, prompt, tokenizer, required_tokens=None, forbidden_tokens=None):
    # Implementation for conditional generation
    pass
```

### **Style Transfer**

```python
# Generate text in different styles by adjusting parameters
def style_transfer_decode(model, prompt, tokenizer, style="formal"):
    if style == "formal":
        return decode(model, prompt, tokenizer, temperature=0.3, top_p=0.4)
    elif style == "creative":
        return decode(model, prompt, tokenizer, temperature=1.8, top_p=0.9)
    elif style == "technical":
        return decode(model, prompt, tokenizer, temperature=0.2, top_p=0.3)
```

## 📝 **Best Practices**

### **1. Parameter Selection**

- **Start with defaults**: temperature=1.0, top_p=1.0
- **For focused text**: temperature=0.3-0.7, top_p=0.3-0.7
- **For creative text**: temperature=1.2-1.8, top_p=0.7-0.9
- **For technical text**: temperature=0.2-0.5, top_p=0.3-0.6

### **2. Prompt Engineering**

- **Be specific**: "Write a technical explanation of" vs "Explain"
- **Set context**: "In the context of machine learning, explain"
- **Use examples**: "Similar to how transformers work, explain"

### **3. Quality Control**

- **Check probabilities**: High average probability indicates good quality
- **Monitor repetition**: Look for repeated phrases or tokens
- **Validate coherence**: Ensure generated text makes sense

### **4. Performance Optimization**

- **Batch generation**: Use batch_decode for multiple prompts
- **Device selection**: Use GPU for faster generation
- **Context length**: Balance between quality and memory usage

## 🎯 **Integration with Training**

### **Using Generated Text for Training**

```python
# Generate synthetic training data
synthetic_prompts = [
    "The benefits of using transformers include",
    "To implement attention mechanisms, you need to",
    "The key challenges in NLP are"
]

synthetic_data = []
for prompt in synthetic_prompts:
    generated = decode(model, prompt, tokenizer, max_new_tokens=50)
    synthetic_data.append(prompt + generated)

# Use synthetic data for fine-tuning
# (Implementation depends on your training pipeline)
```

### **Evaluating Model Quality**

```python
# Generate text and evaluate quality metrics
def evaluate_generation_quality(model, tokenizer, test_prompts):
    results = []
    for prompt in test_prompts:
        result = generate_text_with_controls(
            model, prompt, tokenizer, max_new_tokens=40, verbose=False
        )
        results.append(result)
    
    # Calculate average quality metrics
    avg_prob = sum(r['avg_probability'] for r in results) / len(results)
    avg_length = sum(r['generation_steps'] for r in results) / len(results)
    
    return {
        'average_probability': avg_prob,
        'average_length': avg_length,
        'total_samples': len(results)
    }
```

## 🔮 **Future Enhancements**

### **Planned Features**

1. **Beam Search**: Better generation quality through multiple candidate paths
2. **Conditional Generation**: Generate text with specific constraints
3. **Style Transfer**: Generate text in different writing styles
4. **Interactive Generation**: Real-time text generation with user feedback
5. **Quality Metrics**: Automatic evaluation of generation quality

### **Research Directions**

1. **Better Sampling Strategies**: Advanced sampling methods beyond temperature and top-p
2. **Controlled Generation**: More fine-grained control over output characteristics
3. **Multi-modal Generation**: Generate text with image or audio context
4. **Efficient Generation**: Faster generation for real-time applications

## 📚 **References**

- **Temperature Scaling**: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
- **Top-p Sampling**: Holtzman et al. (2020) - "The Curious Case of Neural Text Degeneration"
- **Beam Search**: Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
- **Transformer Architecture**: Vaswani et al. (2017) - "Attention Is All You Need"

## 🎉 **Conclusion**

The text generation system provides a comprehensive toolkit for generating high-quality text from trained Transformer Language Models. With proper parameter tuning and prompt engineering, you can achieve excellent results for various applications including:

- **Creative Writing**: Stories, poems, creative content
- **Technical Writing**: Documentation, explanations, tutorials
- **Content Generation**: Articles, summaries, translations
- **Research Applications**: Data augmentation, hypothesis generation

For questions or issues, refer to the individual function implementations or modify the generation logic as needed for your specific use case.

