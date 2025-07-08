# LLM Archaeology Agent

The agent should read LLM names from `llm_names.input`, which may contain naming errors, and then
1. Find the correct names of the LLMs
2. Find the release dates of them 
3. Group LLMs by company and return the output in the following Python dictionary format:

```python
LLMs = {
    "Company 1": [
        "LLM 1", #  "Release date",
        "LLM 2" : "Release date"
    ],
    "Company 2": [
        "LLM 3", # "Release date",
        "LLM 4" # "Release date"
    ]
}
```
4. Dump the result to a Python file whose name is set from command line argument `-o`. 

The input file is hardcoded as `llm_names.input`. Please analyze it to find ways to parse it.

Please use Gemini-2.5-Pro with web search tool and dynamic thinking. 
Assume my Gemini key is in .env. 
