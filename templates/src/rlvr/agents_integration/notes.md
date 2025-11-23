# Agent SDK output to HuggingFace

HuggingFace transformer requires the following format: `messages` and `tools`.

Messages in the usual OpenAI Chat Completion format, including a special "tool" role for tool outputs.

```json
[
  { "role": "user", "content": "Turn on the living room lights." },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "control_light",
          "arguments": { "room": "living room", "state": "on" }
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "control_light",
    "content": "The lights in the living room are now on."
  },
  { "role": "assistant", "content": "Done!" }
]
```

Tools as a list of dict of the following format:

```json
{
  "type": "function",
  "function": {
    "name": "control_light",
    "description": "Controls the lights in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": { "type": "string", "description": "The name of the room." },
        "state": {
          "type": "string",
          "description": "The desired state of the light (\"on\" or \"off\")."
        }
      },
      "required": ["room", "state"]
    },
    "return": {
      "type": "string",
      "description": "str: A message indicating the new state of the lights."
    }
  }
}
```

Test the format using `tokenizer.apply_chat_template`:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tools=(tools or None),  # type: ignore[arg-type]
    tokenize=False,
    add_generation_prompt=False,
)
```


Example output from Qwen3 format:

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "parameters": {"properties": {"city": {"description": "\nHuman-readable city name (e.g., \"Vancouver\").", "title": "City", "type": "string"}, "unit": {"default": "c", "description": "\nTemperature unit: \"c\" for Celsius, \"f\" for Fahrenheit. Defaults to \"c\".", "enum": ["c", "f"], "title": "Unit", "type": "string"}, "when": {"default": "now", "description": "\nTime window for the report: \"now\", \"today\", or \"tomorrow\". Defaults to \"now\".", "enum": ["now", "today", "tomorrow"], "title": "When", "type": "string"}}, "required": ["city", "unit", "when"], "title": "get_weather_args", "type": "object", "additionalProperties": false}, "description": "Return a deterministic, mock weather report for demos.\n\nThe function is *offline* and *stable across runs* for a given `(city, date)`\nso it's ideal for showcasing **function-tool** calls without network flakiness."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Weather in Auckland<|im_end|>
<|im_start|>user
<tool_response>
{"city":"Auckland","unit":"c","temperature":16.4,"feels_like":16.5,"condition":"cloudy","humidity":73,"wind_kph":3.3,"observation_time":"2025-09-12T18:01:56.767763Z"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Okay, let me process the tool response. The user asked for the weather in Auckland, so I need to present the data concisely. The temperature is 16.4°C, feels like 16.5°C. The condition is cloudy. Humidity is 73%, wind speed 3.3 kph. I should mention the time as well. Let me check the units again to make sure they're correct. Celsius for temperature and feels-like, and the condition. No need for extra info unless specified. Alright, putting it all together in a clear sentence.
</think>

The weather in Auckland is currently 16.4°C with a feels-like temperature of 16.5°C, with a cloudy condition and 73% humidity. Wind is blowing at 3.3 kph.<|im_end|>
```
