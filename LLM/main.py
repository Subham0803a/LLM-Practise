from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, T5Tokenizer
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions
import torch
import os
import time
import math
import uvicorn

load_dotenv()

app = FastAPI()

# Load tokenizer - GPT2 Model uses Byte-Pair Encoding (BPE)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2") # For embeddings part

# Load tokenizer - BERT model using WordPiece
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load tokenizer - T5 model using SentencePiece
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Initialize GEMINI-CLIENT-SETUP
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"), 
    http_options=types.HttpOptions(api_version="v1")
)

MODEL_NAME = "gemini-2.5-flash"

# Model for Request Schemas
class TokenRequest(BaseModel):
    text: str

class TextRequest(BaseModel):
    prompt: str
    
class ContextRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

class TempRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    
# Basic Health Check Endpoint
@app.get("/")
def read_root():
    return {"message": "LLM Practise API"}

@app.get("/health")
def health_check():
    tokenizers_status = {
        "gpt2": gpt2_tokenizer is not None,
        "bert": bert_tokenizer is not None,
        "t5": t5_tokenizer is not None
    }
    return {"status": "online", "models_loaded": tokenizers_status}

"""
Basic Tokenization with GPT2 Tokenizer

Using GPT2 Tokenizer for demonstration for 
tokenization and detokenization by 
Convert text into tokens and token IDs
"""

# Endpoint to tokenize text -- PART-1
@app.post("/tokenize")
def tokenize_text(text: str):
    """
    Endpoint: /tokenize
    Purpose: Convert text into tokens and token IDs
    
    Parameters:
    - text: Input string to tokenize
    
    Returns:
    - original_text: The input text
    - tokens: List of subword tokens
    - token_ids: Numerical IDs for each token
    - token_count: Number of tokens generated
    """
    # Encode text to token IDs
    token_ids = gpt2_tokenizer.encode(text)
    
    # Convert token IDs back to tokens (subwords)
    tokens = gpt2_tokenizer.convert_ids_to_tokens(token_ids)
    
    return {
        "original_text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(token_ids)
    }

# Endpoint to detokenize token IDs back to text -- PART-2
@app.post("/detokenize")
def detokenize_ids(token_ids: list = Body(..., embed=True)):
    """
    Endpoint: /detokenize
    Purpose: Convert token IDs back to human-readable text
    
    Parameters:
    - token_ids: List of numerical token IDs
    
    Returns:
    - decoded_text: Original text reconstructed from tokens
    """
    # Decode token IDs back to text
    decoded_text = gpt2_tokenizer.decode(token_ids)
    
    return {
        "token_ids": token_ids,
        "decoded_text": decoded_text
    }
    
# -----------------------------------------------------------------------------

"""
Subword Tokenization Algorithms

Comparing all 3 tokenizers :- GPT2, BERT, T5 

BPE (GPT-2): Merges frequent character pairs iteratively
WordPiece (BERT): Splits words into subwords based on frequency
SentencePiece (T5): Uses a vocabulary-based approach for tokenization
"""

# Endpoint to compare tokenizers with diff algorithms -- PART-3
@app.post("/compare-tokenizers")
def compare_tokenizers(text: str):
    """
    Endpoint: /compare-tokenizers
    Purpose: Compare how different algorithms tokenize the same text
    
    How it works:
    1. BPE (GPT-2): Merges frequent character pairs iteratively
    2. WordPiece (BERT): Similar to BPE but uses likelihood maximization
    3. SentencePiece (T5): Language-agnostic, treats text as raw bytes
    
    Parameters:
    - text: Input string
    
    Returns:
    - Tokenization results from all three algorithms
    """
    # BPE Tokenization
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    gpt2_ids = gpt2_tokenizer.encode(text)
    
    # WordPiece Tokenization
    bert_tokens = bert_tokenizer.tokenize(text)
    bert_ids = bert_tokenizer.encode(text)
    
    # SentencePiece Tokenization
    t5_tokens = t5_tokenizer.tokenize(text)
    t5_ids = t5_tokenizer.encode(text)
    
    return {
        "original_text": text,
        "BPE (GPT-2)": {
            "algorithm": "Byte-Pair Encoding",
            "tokens": gpt2_tokens,
            "token_ids": gpt2_ids,
            "count": len(gpt2_tokens)
        },
        "WordPiece (BERT)": {
            "algorithm": "WordPiece",
            "tokens": bert_tokens,
            "token_ids": bert_ids,
            "count": len(bert_tokens)
        },
        "SentencePiece (T5)": {
            "algorithm": "Unigram/SentencePiece",
            "tokens": t5_tokens,
            "token_ids": t5_ids,
            "count": len(t5_tokens)
        }
    }

# Endpoint to show vocabulary size of different tokenizers with diff algorithms -- PART-4
@app.post("/vocabulary-info")
def get_vocabulary_info():
    """
    Endpoint: /vocabulary-info
    Purpose: Show vocabulary size of different tokenizers
    
    Why vocabulary size matters:
    - Larger vocab = fewer tokens per text but bigger model
    - Smaller vocab = more tokens but smaller embedding layer
    """
    return {
        "GPT-2": {
            "vocab_size": len(gpt2_tokenizer),
            "type": "BPE",
            "note": "~50,257 tokens"
        },
        "BERT": {
            "vocab_size": len(bert_tokenizer),
            "type": "WordPiece",
            "note": "~30,522 tokens"
        },
        "T5": {
            "vocab_size": len(t5_tokenizer),
            "type": "SentencePiece",
            "note": "~32,128 tokens"
        }
    }
    
# -------------------------------------------------------------------------------------

"""
Convert text to embeddings
Using Pre-trained (GPT2 Model) Tokenizers to Generate Embeddings
"""

# Endpoint for converting text to embeddings -- PART-5
@app.post("/get-embeddings")
def get_embeddings(text: str):
    """
    Endpoint: /get-embeddings
    Purpose: Convert text to embeddings (numerical vectors)
    
    Process:
    1. Tokenization: Text → Token IDs
    2. Embedding Lookup: Token IDs → Embedding Vectors
    3. Each token becomes a 768-dimensional vector in GPT-2
    
    Parameters:
    - text: Input string
    
    Returns:
    - Token-level embeddings and their shapes
    """
    # Step 1: Tokenize
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    tokens = gpt2_tokenizer.convert_ids_to_tokens(token_ids)
    
    # Step 2: Get embeddings
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        # Last hidden state contains embeddings for each token
        embeddings = outputs.last_hidden_state

    first_token_embedding = embeddings[0][0].tolist()[:10]  # Show first 10 dimensions
    
    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "embedding_dimension": embeddings.shape[-1],  # 768 for GPT-2
        "number_of_tokens": embeddings.shape[1],
        "first_token": tokens[0],
        "first_token_embedding_sample": first_token_embedding,
        "note": f"Each token has {embeddings.shape[-1]} dimensions (showing first 10)"
    }

# Endpoint to calculate semantic similarity between two texts -- PART-6
@app.post("/semantic-similarity")
def calculate_similarity(text1: str =  Body(..., embed=True), text2: str =  Body(..., embed=True)):
    """
    Endpoint: /semantic-similarity
    Purpose: Calculate how similar two texts are using embeddings
    
    How it works:
    1. Convert both texts to embeddings
    2. Calculate cosine similarity (ranges from -1 to 1)
    3. Higher score = more semantically similar
    
    JSON body:
    {
        "text1": "First input text",
        "text2": "Second input text"
    }
    
    Example:
    - "cat" and "kitten" → high similarity (~0.8)
    - "cat" and "car" → low similarity (~0.2)
    """
    # Get embeddings for both texts
    inputs1 = gpt2_tokenizer(text1, return_tensors="pt")
    inputs2 = gpt2_tokenizer(text2, return_tensors="pt")

    with torch.no_grad():
        # Get mean pooling of all token embeddings
        outputs1 = gpt2_model(**inputs1).last_hidden_state.mean(dim=1)
        outputs2 = gpt2_model(**inputs2).last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(outputs1, outputs2)
    similarity_score = cos_sim.item()
    
    return {
        "text1": text1,
        "text2": text2,
        "similarity_score": round(similarity_score, 4),
        "interpretation": "1.0=identical, 0.0=unrelated, -1.0=opposite"
    }

# Endpoint to explain embeddings -- PART-7
@app.get("/embedding-info")
def embedding_info():
    """
    Endpoint: /embedding-info
    Purpose: Explain what embeddings are
    
    Key Concepts:
    - Embedding Matrix: Vocabulary Size × Embedding Dimension
    - GPT-2: 50,257 tokens × 768 dimensions
    - Each row is a learnable vector for one token
    """
    vocab_size = len(gpt2_tokenizer)
    embedding_dim = gpt2_model.config.n_embd
    
    return {
        "vocabulary_size": vocab_size,
        "embedding_dimension": embedding_dim,
        "total_parameters_in_embeddings": vocab_size * embedding_dim,
        "explanation": {
            "what": "Embeddings map discrete tokens to continuous vectors",
            "why": "Neural networks need numerical inputs, not text",
            "how": "Each token ID looks up its vector in embedding matrix",
            "example": "Token 'cat' (ID: 3797) → [0.12, -0.45, ..., 0.78] (768 numbers)"
        }
    }
    
# -------------------------------------------------------------------------------------

"""
Testing Gemini Model Integration
Using Google Gemini API to generate text completions
"""

# Endpoint to demonstrate text completion with Gemini API -- PART-8
@app.post("/complete-text") 
def complete_text(request: TextRequest):
    """
    Next-Token(word) Prediction with Gemini API
    Real Gemini API call - Demonstrates text completion
    
    How it works:
    1. Takes your prompt text
    2. Gemini predicts/completes the next tokens
    3. Returns generated completion + full response
    """
    
    try:
        # 2. Attempt Gemini Call
        response = client.models.generate_content(
            model="gemini-2.5-flash", # gemini-1.5-flash or gemini-2.0-flash or gemini-2.5-flash
            contents=request.prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=0.7
            )
        )
        
        return {
            "status": "success",
            "original_prompt": request.prompt,
            "completion": response.text
            }

    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg:
            print("!!! DAILY QUOTA EMPTY: You must wait until midnight PT or use a new project key.")
        return {"status": "error", "message": error_msg}

# Endpoint to show pre-built examples -- PART-9
@app.get("/examples")
def text_completion_examples():
    """
    Pre-built examples showing text completion vs search
    """
    examples = {
        "text_completion": {
            "input": "The sky is",
            "expected_output": "blue"
        },
        "llm_generates": {
            "input": "Write a Python function to",
            "expected_output": "calculate fibonacci numbers"
        },
        "not_search": {
            "input": "Current president of USA is",
            "llm_generates": "Donald Trump (trained knowledge)",
            "not": "real-time web search"
        }
    }
    return examples

# -------------------------------------------------------------------------------------

"""
Testing Gemini Model Integration
Using Google Gemini API Tokens - Words/Parts of Words
APIs charges per token (input + output)

Example --> "Unhappiness" = 3 tokens: ["Un", "happi", "ness"]
"""

# function to count tokens using Gemini API
def counting_tokens(text: str) -> int:
    """Count tokens using Gemini's token counter"""
    result = client.models.count_tokens(
        model="gemini-2.5-flash",
        contents=text
    )
    return result.total_tokens

# Endpoint to analyze tokens and pricing -- PART-10
@app.post("/count-tokens")
def analyze_tokens(request: TokenRequest):
    """
    Analyzes text and provides 2026 pricing estimates.
    """
    input_tokens = counting_tokens(request.text)
    
    # 1 token is roughly 4 characters for English
    chars_per_token = len(request.text) / input_tokens if input_tokens > 0 else 0
    
    # Estimate output (Flash models are often used for concise summaries)
    estimated_output_tokens = int(input_tokens * 1.2)
    
    # Updated 2026 Pricing (Gemini 2.5 Flash)
    # Input: $0.15 per 1M tokens | Output: $0.60 per 1M tokens
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (estimated_output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost
    
    return {
        "analysis": {
            "input_tokens": input_tokens,
            "char_count": len(request.text),
            "efficiency": f"{chars_per_token:.2f} chars/token"
        },
        "pricing_2026_usd": {
            "input_cost": f"${input_cost:.8f}",
            "estimated_output_cost": f"${output_cost:.8f}",
            "total_estimated_session": f"${total_cost:.8f}"
        },
        "limits": {
            "model": MODEL_NAME,
            "context_window": "1.05M tokens",
            "max_output": "65k tokens"
        }
    }

# Endpoint to demonstrate tokenization examples -- PART-11
@app.post("/tokenize-example")
def tokenization_demo():
    """Demonstrates how the 2026 tokenizer handles different text types"""
    examples = ["chat", "unhappiness", "Pythonic logic", "def main():"]
    results = []
    
    for item in examples:
        count = counting_tokens(item)
        results.append({
            "text": item,
            "tokens": count,
            "bytes": len(item.encode('utf-8'))
        })
    
    return {"subword_examples": results}

# -------------------------------------------------------------------------------------

"""
Context Window and Token Limits
Context window = maximum tokens LLM can consider at once
Gemini 2.5 Flash: ~1.05 million tokens context window (~780K-800K words)

Gemini 1.5 Flash: 1 million tokens context window (~750K words). 
Older models: 4K-128K tokens
"""

# Endpoint to test context window limit utilization -- PART-12
@app.post("/test-context")
def test_context_window(request: ContextRequest):
    """
    Demonstrates real-time token tracking and context utilization.
    """
    # Count prompt tokens
    token_response = client.models.count_tokens(
        model="gemini-2.5-flash",
        contents=request.prompt
    )
    prompt_tokens = token_response.total_tokens
    
    # Context window for 2.5 Flash in 2026 is exactly 1,048,576
    MAX_CONTEXT = 1_048_576 
    context_remaining = MAX_CONTEXT - prompt_tokens
    
    # 3. Generate content with modern usage tracking
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=request.prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=request.max_tokens,
            temperature=0.7
        )
    )
    
    # Retrieve tokens from usage metadata
    output_tokens = response.usage_metadata.candidates_token_count
    
    return {
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "response_tokens": output_tokens,
            "total_used": prompt_tokens + output_tokens,
            "percent_of_window_used": f"{( (prompt_tokens + output_tokens) / MAX_CONTEXT ) * 100:.4f}%"
        },
        "capacity": {
            "remaining_tokens": context_remaining,
            "fits_heavy_novel": context_remaining > 500_000,
            "status": "✅ Safe" if prompt_tokens < 900_000 else "⚠️ Near Limit"
        },
        "response_text": response.text
    }

# Endpoint to show market comparison of context limits -- PART-13
@app.get("/context-limits")
def context_limits():
    """Updated context window benchmarks for early 2026"""
    return {
        "Gemini 3 Pro (Preview)": "2,097,152 tokens",
        "Gemini 2.5 Flash (GA)": "1,048,576 tokens",
        "Llama 4 Scout (Open)": "10,000,000 tokens",
        "GPT-5.2 (Premium)": "400,000 tokens",
        "Claude 5.1 Sonnet": "200,000 tokens"
    }
    
# -------------------------------------------------------------------------------------

"""
Temperature - Creativity Control
Temperature parameter controls randomness in text generation.

Low Temp (0.0): robot mode (safe, boring, consistent)
Mid Temp (1.0): normal mode (balanced, natural)
High Temp (2.0): crazy mode (creative, unpredictable)
"""

# Endpoint to compare temperature settings -- PART-14
@app.post("/temperature")
def generate_content(request: TempRequest):
    """
    Takes prompt + temp and returns prompt + response.
    
    1. ROBOT MODE (Deterministic, Factual, Boring)
       JSON Body: { "prompt": "Your prompt", "temperature": 0.0 }
       
    2. NORMAL MODE (Balanced, Natural, Default)
       JSON Body: { "prompt": "Your prompt", "temperature": 1.0 }
       
    3. CRAZY MODE (Creative, Unpredictable, Unique)
       JSON Body: { "prompt": "Your prompt", "temperature": 2.0 }
    """
    # Generate content using your specific input
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=request.prompt,
        config=types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=300
        )
    )

    # Return exactly what you asked for
    return {
        "prompt": request.prompt,
        "temperature": request.temperature,
        "response": response.text
    }

if __name__ == "__main__":
    # The uvicorn.run command starts the server
    uvicorn.run("main:app", host="localhost", port=6000, reload=True)
    # uvicorn.run("main:app", host="localhost", port=5000)
    