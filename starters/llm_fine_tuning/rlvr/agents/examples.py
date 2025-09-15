"""
OpenAI Agents SDK demo: function tool `get_weather` + minimal agent.

Setup (with astral-uv):
    uv venv && uv pip install -U pip
    uv add openai-agents pydantic
    # Or, if you prefer pip: `pip install openai-agents pydantic`

Environment:
    export OPENAI_API_KEY=...  # Required for the default OpenAI client

Run:
    uv run python agents_sdk_get_weather_demo.py
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field


class WeatherReport(BaseModel):
    """Structured output for a weather report.

    Attributes
    ----------
    city:
        Echo of the requested city (canonicalized).
    unit:
        "c" for Celsius or "f" for Fahrenheit.
    temperature:
        Air temperature in the requested unit.
    feels_like:
        Apparent temperature in the requested unit.
    condition:
        One of: "clear", "partly cloudy", "cloudy", "rain", "snow", "windy".
    humidity:
        Relative humidity percentage (0–100).
    wind_kph:
        Wind speed in kilometers per hour.
    observation_time:
        UTC timestamp when the reading was generated.
    """

    city: str
    unit: Literal["c", "f"]
    temperature: float
    feels_like: float
    condition: Literal["clear", "partly cloudy", "cloudy", "rain", "snow", "windy"]
    humidity: int = Field(ge=0, le=100)
    wind_kph: float = Field(ge=0)
    observation_time: datetime


@function_tool
def get_weather(
    city: str,
    unit: Literal["c", "f"] = "c",
    when: Literal["now", "today", "tomorrow"] = "now",
) -> str:
    """Return a deterministic, mock weather report for demos.

    The function is *offline* and *stable across runs* for a given `(city, date)`
    so it's ideal for showcasing **function-tool** calls without network flakiness.

    Args:
        city:
            Human-readable city name (e.g., "Vancouver").
        unit:
            Temperature unit: "c" for Celsius, "f" for Fahrenheit. Defaults to "c".
        when:
            Time window for the report: "now", "today", or "tomorrow". Defaults to "now".

    Returns:
        JSON string representing a `WeatherReport`.
    """
    canonical = city.strip()

    # City baselines (°C). Extend this mapping to taste.
    baselines: dict[str, float] = {
        "vancouver": 14.0,
        "new york": 12.0,
        "london": 11.0,
        "singapore": 28.0,
        "shanghai": 28.0,
        "auckland": 20.0,
        "tokyo": 17.0,
        "paris": 13.0,
        "san francisco": 16.0,
        "berlin": 12.0,
        "mexico city": 19.0,
    }

    key = canonical.lower()
    base_c = baselines.get(key, 15.0)

    # Reference date for deterministic seeding
    today = date.today()
    ref_date = today if when in ("now", "today") else today + timedelta(days=1)

    # Seeded pseudo-randoms derived from (city, date)
    seed = abs(hash(f"{key}|{ref_date.isoformat()}"))

    def prand(a: float, b: float, salt: int) -> float:
        # Deterministic pseudo-random in [a, b]
        return a + (seed ^ salt) % 10 / 9.0 * (b - a)

    temp_c = base_c + prand(-4.0, 4.0, 0xA5A5) - 0.5
    humidity = int(round(prand(40, 90, 0xB6B6)))
    wind_kph = round(prand(0.0, 30.0, 0xC7C7), 1)

    band = seed % 100
    if band < 20:
        condition = "clear"
    elif band < 45:
        condition = "partly cloudy"
    elif band < 65:
        condition = "cloudy"
    elif band < 85:
        condition = "rain"
    elif band < 95:
        condition = "windy"
    else:
        condition = "snow"

    feels_c = temp_c - 0.1 * wind_kph + 0.02 * (humidity - 50)

    def to_unit(tc: float, u: Literal["c", "f"]) -> float:
        return round(tc if u == "c" else (tc * 9 / 5 + 32), 1)

    report = WeatherReport(
        city=canonical,
        unit=unit,
        temperature=to_unit(temp_c, unit),
        feels_like=to_unit(feels_c, unit),
        condition=condition,  # type: ignore[arg-type]
        humidity=humidity,
        wind_kph=wind_kph,
        observation_time=datetime.now(timezone.utc),
    )

    # Agents SDK tools should return a string (or something that stringifies cleanly).
    return report.model_dump_json()


# --- Minimal agent wiring ----------------------------------------------------
weather_agent = Agent(
    name="Weather Helper",
    instructions=(
        "You answer weather questions. When the user asks about weather, "
        "call the `get_weather` tool. If it returns JSON, parse it and reply "
        "concisely with temperature, feels-like, condition, and units."
    ),
    tools=[get_weather],  # register the function tool
)


def main() -> None:
    """Run a single demo turn with the agent and print the final output."""
    # Example inputs that strongly encourage tool use
    user_inputs: list[str] = [
        "What's the weather in Vancouver today in celsius?",
        "NYC now, in Fahrenheit — include feels-like and wind, please.",
    ]

    for i, prompt in enumerate(user_inputs, start=1):
        print(f"\n=== Demo turn {i} ===")
        result = Runner.run_sync(weather_agent, prompt)
        print(result.final_output)


if __name__ == "__main__":
    main()
