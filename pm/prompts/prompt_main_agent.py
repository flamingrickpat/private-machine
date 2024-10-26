prompt_conscious_assistant_base = """
{character_card}
{tool_insert}
{thought_insert}
"""

tool_insert = """
You may answer in natural language or call a a tool. The tool calls are in valid JSON, and valid JSON only! 
If you do a JSON tool call, the next user response will be the result of that tool call.
Here are the possible schemas:

### BEGIN TOOL SCHEMAS
{tool_schemas}
### END TOOL SCHEMAS
"""

thought_insert = """
This is a complex query, you may use internal thought to think it over before an answer.
User ** (double asterisk) to start and end a thought. This will be invisible to the user. 
Take a deep breath and think it through before making the final response to the user in the rest of the message.
"""