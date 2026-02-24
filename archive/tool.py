from openai import OpenAI
client = OpenAI(api_key="")

# Define tools with the calculator_add function
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_add",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number"},
                    "num2": {"type": "number"},
                },
                "required": ["num1", "num2"],
            },
        },
    }
]

# Simulate a conversation
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Add 8 and 12."}],
    tools=tools,
)

# Output tool calls (the function call details)
print(completion.choices[0].message.tool_calls)

# Example of calling the function with the extracted arguments
def calculator_add(num1, num2):
    return num1 + num2

# Simulating the tool being invoked
tool_calls = completion.choices[0].message.tool_calls
tool_call = tool_calls[0]
function_name = tool_call.function.name
arguments = eval(tool_call.function.arguments)
result = calculator_add(**arguments)
print(result)