# Example curl — generate sample_request.json first:
#   python examples/generate_sample_request.py
#
# Health check
curl -s http://localhost:8000/health | jq .

# Model info
curl -s http://localhost:8000/model/info | jq .

# Single forecast
curl -s -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json | jq .

# Batch forecast (two containers)
curl -s -X POST http://localhost:8000/forecast/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [<paste sample_request.json>, <paste sample_request.json>]}' | jq .
