import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr

# ENVIRONMENT
load_dotenv(override=True)
open_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

openai = OpenAI()
claude = anthropic.Anthropic()
deepseek = OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com/v1")


OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"
DEEPSEEK_MODEL = 'deepseek-coder'

system_message = (
    "You are an assistant that adds docstrings to functions and comments to code where necessary. "
    "Do not modify the code itself — only add docstrings and explanatory comments. "
    "The functionality of the code must remain exactly the same. "
    "Do not explain your reasoning or what additions you made — just add docstrings and comments to the code. "
    "Return only the modified version of the code, with no extra explanations or comments beyond what was asked."
)
def user_prompt_for(code):
    user_prompt = (
        "Add comments to the following code.\n"
        "DO NOT change the functionality of the code.\n\n"
        f"{code}"
    )
    return user_prompt

def messages_for(code):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(code)}
    ]
def stream_gpt(code):
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_for(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
        
def stream_deepseek(code):
    stream = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages_for(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
def stream_claude(code):
    result = claude.messages.stream(
        model= CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(code)}]
    )
    reply = ''
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```python\n','').replace('```','')

python_hard = """
def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value
        
def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters
n = 10000         # Number of random numbers
initial_seed = 42 # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
"""
def docstring(python, model):
    if model=="GPT":
        result = stream_gpt(python)
    elif model=="Claude":
        result = stream_claude(python)
    elif model == "DeepSeek":
        result = stream_deepseek(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far   
        
system_message_unit_test = (
    "You are an assistant that generates Python unit test code for the given code. "
    "Do not modify the original code — only write test cases that thoroughly cover the functionality. "
    "Use the standard unittest framework syntax. "
    "Return only the test code, no explanations or additional text. "
    "The tests should be clear, concise, and runnable as-is."
)
def user_prompt_unit_test(code: str) -> str:
    return (
        "Write Python unit tests for the following code using the unittest module. "
        "Do not change the original code. Only add test cases to verify its behavior.\n\n"
        f"{code}"
    )
def messages_for_test(code):
    return [
        {"role": "system", "content": system_message_unit_test},
        {"role": "user", "content": user_prompt_unit_test(code)}
    ]
def stream_gpt_test(code):
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_for_test(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
##############
def stream_deepseek_test(code):
    stream = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages_for_test(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
######################
def stream_claude_test(code):
    result = claude.messages.stream(
        model= CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message_unit_test,
        messages=[{"role": "user", "content": user_prompt_unit_test(code)}]
    )
    reply = ''
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```python\n','').replace('```','')
#####################
def unit_test(python, model):
    if model=="GPT":
        result = stream_gpt_test(python)
    elif model=="Claude":
        result = stream_claude_test(python)
    elif model == "DeepSeek":
        result = stream_deepseek_test(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far   
test_case = """def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b"""
system_message_explaio = (
    "You are a helpful assistant that explains Python code in clear, beginner-friendly terms. "
    "Break down the logic, describe the purpose of functions and classes, and clarify complex parts. "
    "Use simple language, bullet points, and examples where appropriate. "
    "Do not modify or rewrite the code. Just explain what it does and how it works."
)
def user_prompt_explaio(code: str) -> str:
    return (
        "Explain the following Python code in simple terms. "
        "Provide an overview of what the code does and explain any functions, loops, or logic used.\n\n"
        f"{code}"
    )
def messages_for_explaio(code):
    return [
        {"role": "system", "content": system_message_explaio},
        {"role": "user", "content": user_prompt_explaio(code)}
    ]
def stream_gpt_explain(code):
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_for_explaio(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
##############
def stream_deepseek_explain(code):
    stream = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages_for_explaio(code),
        stream=True
    )
    result = ''
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        result += fragment
        yield result.replace('```python\n','').replace('```','')
######################
def stream_claude_explain(code):
    result = claude.messages.stream(
        model= CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message_explaio,
        messages=[{"role": "user", "content": user_prompt_explaio(code)}]
    )
    reply = ''
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```python\n','').replace('```','')
#####################
def explain_code(code, model):
    if model=="GPT":
        result = stream_gpt_explain(code)
    elif model=="Claude":
        result = stream_claude_explain(code)
    elif model == "DeepSeek":
        result = stream_deepseek_explain(code)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far   
python_example = """
class ExpressionError(Exception):
    pass

class ExpressionEvaluator:
    def __init__(self, expression):
        self.expression = expression.replace(" ", "")

    def evaluate(self):
        try:
            return self._evaluate_expression(self.expression)
        except ZeroDivisionError:
            raise ExpressionError("Division by zero is not allowed.")
        except Exception as e:
            raise ExpressionError(f"Invalid expression: {e}")

    def _evaluate_expression(self, expr):
        if expr.isdigit():
            return int(expr)

        for op in ['+', '-', '*', '/']:
            depth = 0
            for i in range(len(expr) - 1, -1, -1):
                if expr[i] == ')':
                    depth += 1
                elif expr[i] == '(':
                    depth -= 1
                elif depth == 0 and expr[i] == op:
                    left = self._evaluate_expression(expr[:i])
                    right = self._evaluate_expression(expr[i + 1:])
                    return self._apply_operator(op, left, right)

        if expr[0] == '(' and expr[-1] == ')':
            return self._evaluate_expression(expr[1:-1])

        raise ExpressionError("Malformed expression")

    def _apply_operator(self, op, a, b):
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if b == 0:
                raise ZeroDivisionError()
            return a / b
        else:
            raise ExpressionError(f"Unsupported operator: {op}")
"""
MODEL_OPTIONS = ["GPT", "Claude", "DeepSeek"]
with gr.Blocks(title="Code Assistant", theme=gr.themes.Soft()) as ui:
    with gr.Tabs():
        # Docstring Adder Tab
        with gr.Tab("DocoBot"):
            gr.Markdown("### 📝 DocuBot: Auto-Generate Docstrings & Comments")
            with gr.Row():
                docu_code_input = gr.Code(
                    label="Input Code",
                    language="python",
                    lines=20,
                    value=python_hard                )
                docu_output = gr.Code(
                    label="Output Code with Docstrings",
                    language="python",
                    lines=20,
                    interactive=False
                )
            with gr.Row():
                docu_model_select = gr.Dropdown(
                    MODEL_OPTIONS,
                    value="DeepSeek",
                    label="Select Model"
                )
            with gr.Row():
                docu_convert_btn = gr.Button("Add Docstrings")
                docu_clear_btn = gr.Button("Clear")

            # Call docstring function on button click
            docu_convert_btn.click(
                fn=docstring,
                inputs=[docu_code_input, docu_model_select],
                outputs=[docu_output]
            )
            # Clear inputs and outputs
            docu_clear_btn.click(
                fn=lambda: ("", ""),
                inputs=[],
                outputs=[docu_code_input, docu_output]
            )

        # Unit Test Case Adder Tab
        with gr.Tab("TestoBot"):
            gr.Markdown("### 🧪 TestIT: Instantly Add Unit Tests to Your Code")

            with gr.Row():
                test_code_input = gr.Code(
                    label="Input Code",
                    language="python",
                    lines=20,
                    value=test_case                )
                test_output = gr.Code(
                    label="Output Code with Unit Tests",
                    language="python",
                    lines=20,
                    interactive=False
                )
            with gr.Row():
                test_model_select = gr.Dropdown(
                    MODEL_OPTIONS,
                    value="DeepSeek",
                    label="Select Model"
                )
            with gr.Row():
                test_convert_btn = gr.Button("Add Unit Tests")
                test_clear_btn = gr.Button("Clear")

            # Call unit_test function on button click
            test_convert_btn.click(
                fn=unit_test,
                inputs=[test_code_input, test_model_select],
                outputs=[test_output]
            )
            # Clear inputs and outputs
            test_clear_btn.click(
                fn=lambda: ("", ""),
                inputs=[],
                outputs=[test_code_input, test_output]
            )
        with gr.Tab("🧠 ExplaioBot"):
            gr.Markdown("## 🧠 Understand Your Code")
        
            with gr.Row():
                code = gr.Code(label="🧾 Your Code", lines=20,language='python', value=python_example)
                output = gr.Textbox(label="🧠 Explanation", lines=35)
        
            with gr.Row():
                model = gr.Dropdown(["GPT", "Claude", "DeepSeek"], value="DeepSeek", label="Select Model")
        
            with gr.Row():
                explain_btn = gr.Button("🔍 Explain Code")
        
            explain_btn.click(fn=explain_code, inputs=[code, model], outputs=[output])

ui.launch()