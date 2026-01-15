tested tool call formats:
- JSON (less strict)
- MCP (super strict)
- XML
- Unstructured? (instructions)
- Function signatures?

testing philosophy:
- Multiple tools of varying usage complexity (some need to be very weird)
- Various eval prompts of varying complexity with expected tool calls
- Various models with a reasonable system prompt

Check in the .env file:
- Use cerebras zai-glm-4.7, gpt-oss-120b, qwen-3-235b-a22b-instruct-2507, zai-glm-4.6, llama-3.3-70b
- Use groq groq/compound, groq/compound-mini, meta-llama/llama-4-scout-17b-16e-instruct, moonshotai/kimi-k2-instruct-0905, 
- Use gemini gemini-2.5-flash
- Use cohere command-a-reasoning-08-2025 and command-a-03-2025

- I have installed a uv package called keycycle (that i made)
- Import multiproviderwrapper from it
- multiproviderwrapper gives an openai client. investigate the package multiproviderwrapper interface, found in the .venv folder for more details.
- Use this for any LLM calls.