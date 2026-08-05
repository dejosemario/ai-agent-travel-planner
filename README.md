# LLM Tool-Calling Demo (Python) — Flight, Hotel & Currency Tools

This project gives an LLM three function-calling tools:

1. **`get_flight_schedule(origin, destination)`** — round-trip flight schedule
   (outbound + return legs) with duration in hours and price in USD.
2. **`get_hotel_schedule(city, nights)`** — hotel booking with nightly rate
   and total price in USD.
3. **`convert_currency(amount, from_currency, to_currency)`** — currency
   conversion between USD and a handful of other currencies (NGN, KES, EUR,
   GBP), falling back to 1:1 for unrecognized codes.

The LLM is called via the **OpenAI Python SDK, pointed at OpenRouter** (base
URL `https://openrouter.ai/api/v1`), so any OpenRouter-hosted model works.
The script drives the full tool-calling loop itself (no server/endpoint is
started) — it prompts the model, executes whatever tools the model asks for,
feeds the results back, and repeats until the model returns a final answer,
which is printed to stdout.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)
```

`.env` variables:

| Variable             | Required | Default                  | Notes                                   |
|----------------------|----------|---------------------------|------------------------------------------|
| `OPENROUTER_API_KEY` | yes      | —                          | Your OpenRouter API key                  |
| `GEMINI_API_KEY`     | no       | —                          | Unused by this implementation (spec compliance only) |
| `LLM_MODEL_NAME`     | no       | `openai/gpt-4o-mini`       | Any OpenRouter model slug                |

## Run

```bash
python main.py
```

This prints the LLM's final answer — the total round-trip flight time for
Lagos → Nairobi → Lagos, and the total logistics cost (flights + hotel) in
USD for a 3-night stay — to standard output.

## How it works

- `tools.py` contains the pure, self-contained mock implementations of the
  three tools (no network calls, no external data source needed).
- `main.py`:
  1. Declares the three tools in OpenAI function-calling JSON-schema format.
  2. Sends the system + user prompt to the model with `tools` attached.
  3. If the model responds with `tool_calls`, executes each one locally via
     `tools.py` and appends a `role: "tool"` message with the JSON result for
     each call.
  4. Repeats step 2–3 until the model responds with a plain text answer
     (no more tool calls), then prints that answer and exits.
