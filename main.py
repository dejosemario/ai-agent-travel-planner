"""
Function-calling demo: flight schedule, hotel schedule, and currency
conversion tools wired to an LLM via OpenRouter (OpenAI SDK).

Run with:
    python main.py

Prints the LLM's final answer to standard output after driving the full
tool-calling conversation to completion. No server/endpoint is started.
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_IMPLEMENTATIONS

load_dotenv()

# Only OPENROUTER_API_KEY / GEMINI_API_KEY / LLM_MODEL_NAME are guaranteed to
# be supplied externally; everything else has a default.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    print(
        "Missing OPENROUTER_API_KEY. Copy .env.example to .env and set your key.",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ---- tool schemas (OpenAI function-calling format, supported by OpenRouter) ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_schedule",
            "description": (
                "Get the round-trip flight schedule (outbound and return legs) "
                "between two cities, including flight duration in hours and "
                "price in USD for each leg."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure city, e.g. 'Lagos'",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city, e.g. 'Nairobi'",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotel_schedule",
            "description": (
                "Get hotel booking details for a city for a given number of "
                "nights, including nightly rate and total price in USD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to book a hotel in, e.g. 'Nairobi'",
                    },
                    "nights": {
                        "type": "integer",
                        "description": "Number of nights to stay",
                    },
                },
                "required": ["city", "nights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Convert an amount of money from one currency to another "
                "(e.g. USD to NGN)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount to convert",
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "ISO currency code to convert from, e.g. 'USD'",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "ISO currency code to convert to, e.g. 'NGN'",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]


def execute_tool_call(tool_call) -> dict:
    name = tool_call.function.name
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse arguments: {e}"}

    try:
        return fn(**args)
    except Exception as e:  # noqa: BLE001 - surface any tool error back to the model
        return {"error": f"Tool execution failed: {e}"}


def main() -> None:
    user_prompt = (
        "I'm taking a flight from Lagos to Nairobi for a conference. I would "
        "like to know the total flight time back and forth, and the total "
        "cost of logistics for this conference if I'm staying for three days."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful travel-planning assistant. You have tools "
                "to look up flight schedules, hotel schedules, and to convert "
                "currency. Always call the appropriate tools to get real "
                "numbers rather than guessing or estimating. Once you have "
                "the flight and hotel data you need, give the user a clear "
                "final answer that states: (1) the total round-trip flight "
                "time, and (2) the total cost of logistics (flights + hotel) "
                "in USD for the length of their stay."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    max_turns = 8

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        # Store as a plain dict (rather than the SDK's pydantic object) so it
        # round-trips cleanly through the next request.
        messages.append(assistant_message.model_dump(exclude_unset=True))

        if not tool_calls:
            # No more tool calls requested -- this is the final answer.
            print(assistant_message.content)
            return

        # Execute every requested tool call and feed the results back in.
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        # loop again so the model can see the tool results and continue

    print(
        "Reached max conversation turns without a final answer from the model.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
