# Code Assistant 🤖

A powerful Gradio-based application that leverages multiple AI models to enhance your Python code through automated documentation, test generation, and code explanations.

## ✨ Features

### 📝 DocoBot - Documentation Generator
- **Automatic Docstring Generation**: Add comprehensive docstrings to your functions and classes
- **Smart Comments**: Insert explanatory comments where needed
- **Preserves Functionality**: Only adds documentation without modifying your code logic

### 🧪 TestoBot - Unit Test Generator  
- **Instant Test Creation**: Generate comprehensive unit tests using Python's unittest framework
- **Thorough Coverage**: Creates test cases that cover various scenarios and edge cases
- **Ready-to-Run**: Generates clean, executable test code

### 🧠 ExplaioBot - Code Explainer
- **Beginner-Friendly Explanations**: Break down complex code into simple terms
- **Logic Breakdown**: Understand the purpose and flow of functions and classes
- **Educational**: Perfect for learning and code reviews

## 🚀 Supported AI Models

- **GPT-4o-mini** (OpenAI)
- **Claude 3 Haiku** (Anthropic)  
- **DeepSeek Coder** (DeepSeek)

## 📋 Prerequisites

- Python 3.7+
- API keys for the AI services you want to use:
  - OpenAI API Key
  - Anthropic API Key  
  - DeepSeek API Key

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/code-assistant.git
cd code-assistant
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 📦 Dependencies

```txt
gradio
openai
anthropic
python-dotenv
```

## 🏃‍♂️ Usage

1. **Start the application:**
```bash
python doc_test_explain.py
```

2. **Open your browser** and navigate to the provided local URL (typically `http://127.0.0.1:7860`)

3. **Choose your tool:**
   - **DocoBot**: Paste your code, select a model, and click "Add Docstrings"
   - **TestoBot**: Input your functions, choose a model, and click "Add Unit Tests"  
   - **ExplaioBot**: Submit your code, pick a model, and click "Explain Code"

## 💡 Example Usage

### DocoBot Input:
```python
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
```

### DocoBot Output:
```python
def max_subarray_sum(n, seed, min_val, max_val):
    """
    Find the maximum sum of any contiguous subarray in a randomly generated array.
    
    Args:
        n (int): Number of random numbers to generate
        seed (int): Seed for the linear congruential generator
        min_val (int): Minimum value for random numbers
        max_val (int): Maximum value for random numbers
    
    Returns:
        int: Maximum sum of any contiguous subarray
    """
    # Generate random numbers using LCG
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    
    # Initialize maximum sum tracker
    max_sum = float('-inf')
    
    # Check all possible subarrays using brute force approach
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum
```

## 🛠️ Configuration

The application uses different models with specific configurations:

- **OpenAI**: GPT-4o-mini with streaming support
- **Claude**: Haiku model with 2000 token limit
- **DeepSeek**: Coder model optimized for programming tasks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
