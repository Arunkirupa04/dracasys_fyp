# Testing

## Run tests

```bash
cd forecast_service
pytest tests/ -v
```

## Test categories

| File | Scope | Requires TF |
|------|-------|-------------|
| `test_validation.py` | Input validation rules | No |
| `test_schemas.py` | Pydantic request schemas | No |

## Manual integration test

```bash
python run.py &
sleep 30
python examples/generate_sample_request.py
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## What to test before release

- [ ] `/health` returns healthy after startup
- [ ] `/forecast` with valid 200-step sample
- [ ] `/forecast` rejects < 200 steps (422)
- [ ] `/forecast` rejects wrong interval (400)
- [ ] Known container returns `scaler_mode: known`
- [ ] Unknown container returns `scaler_mode: new`
- [ ] Batch partial failure handling

## CI recommendation

- Always run unit tests (fast)
- Run integration test nightly or on artifact updates (slow, needs TF)

## Sample fixtures

Generate with:

```bash
python examples/generate_sample_request.py
```

Output: `examples/sample_request.json`
