# Examples

This directory contains example scripts demonstrating how to use the Pokemon Card Generator.

## Quick Start

### 1. Basic Generation
```bash
python examples/quickstart.py
```

Demonstrates:
- Loading a trained model
- Generating cards
- Saving results

### 2. API Usage
```bash
# Start the API server first
python run_server.py

# Then run the example
python examples/api_usage_example.py --num-images 4 --seed 42
```

Demonstrates:
- Calling API endpoints
- Processing responses
- Saving generated cards

## Example Scripts

### `quickstart.py`
Basic example showing how to:
- Load a checkpoint
- Generate cards programmatically
- Save results

**Usage:**
```bash
python examples/quickstart.py
```

### `api_usage_example.py`
Complete API usage example showing:
- Getting API info
- Health checks
- Generating cards via API
- Saving results

**Usage:**
```bash
python examples/api_usage_example.py \
  --api-url http://localhost:8000 \
  --num-images 4 \
  --seed 42 \
  --output examples/output
```

## Output

All examples save results to `examples/output/` by default.

## Prerequisites

- Trained model checkpoint in `checkpoints/`
- API server running (for API examples)
- Required dependencies installed

## Tips

1. **Use seeds for reproducibility:**
   ```python
   torch.manual_seed(42)
   images = generator.generate(num_images=4)
   ```

2. **Batch generation is efficient:**
   ```python
   # Good: Generate in batches
   images = generator.generate(num_images=16)

   # Less efficient: Generate one at a time
   for i in range(16):
       img = generator.generate(num_images=1)
   ```

3. **Check model status before generating:**
   ```python
   response = requests.get(f"{api_url}/info")
   if response.json()['model_loaded']:
       # Generate cards
   ```

## Next Steps

- Explore the API documentation: http://localhost:8000/docs
- Try the web UI: http://localhost:8000/ui
- Read the main README for more examples
