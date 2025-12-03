# Prompts for LLM-based conversation processing
CONV_BOUNDARY_DETECTION_PROMPT = """
### Core Task
As a conversation analysis expert, you need to determine if the newly added message is a natural ending of an existing conversation "episode". The goal is to segment continuous conversation flow into meaningful, independently memorable segments (MemCell). Your core principle is **"Default to merge, split cautiously"**.

### Conversation Context
**Existing conversation history:**
```
{conversation_history}
```

**Time gap from previous message:**
`{time_gap_info}`

**Newly added messages:**
```
{new_messages}
```

### Decision Variables Explained
You need to output three key decision variables: `should_end`, `should_wait`, and `topic_summary`.

1.  **`should_end` (End current episode):**
    -   **When to set `true`?** Only when the new message clearly opens a new topic unrelated to previous history. A message that summarizes or concludes the current topic (e.g., "Migration complete") should be included in the current episode, not start a new one.
    -   **Trigger scenarios:**
        -   **Cross-day mandatory split:** As long as the new message and the previous message are on different dates (e.g., from "yesterday" to "today"), must split. This is the highest priority rule.
        -   **Topic switch:** Conversation suddenly shifts from "discussing technical details of Project A" to "where to go this weekend".
        -   **Task completion and new chapter:** A closing message for a task (e.g., "Repository migration done") belongs to that task's episode. Only split at the **next** message when it opens a completely unrelated new topic.
        -   **Long interruption followed by new topic:** Time gap exceeds 4 hours, and new message content has no obvious connection to historical conversation.

2.  **`should_wait` (Wait for more information):**
    -   **When to set `true`?** When the new message has **insufficient information** to determine if it continues the current topic. This is a **safe option** to avoid rashly splitting or merging when context is insufficient.
    -   **This is the default behavior, must be `true` in these situations:**
        -   **Non-text messages:** New message is only `[Image]`, `[Video]`, `[File]` placeholders without any text to judge intent.
        -   **Short responses without clear intent:** New message is extremely brief like "OK", "Got it", "Received", "lol", "😂".
        -   **System or non-conversation messages:** New message is system notification (invitations, joins, leaves), transfer tips, etc. **These messages don't provide enough information to judge episode boundaries, must wait for subsequent human messages to decide.**
        -   **Uncertain intermediate state:** Time gap between 30 minutes and 4 hours, and new message content is ambiguous, neither clearly continuing nor clearly starting new.

3.  **`topic_summary` (Episode topic summary):**
    -   **When to generate?** Only when `should_end` is set to `true`.
    -   **Content requirements:** Summarize the core content of **the episode about to end** in one precise, objective sentence. For example: "Finalized the technical solution for Project A" or "Discussed and confirmed weekend team activity arrangements".

### Decision Guidelines
- **Content over form:** Prioritize ensuring each episode contains complete, valuable core information. Don't be misled by simple greetings ("Hello") or closings ("Goodbye") - they should be included in the main episode they serve.
- **Merge is the default:** If uncertain, lean towards not splitting (`should_end: false`). Only split when clear splitting signals are present.
- **Focus on causal and process continuity:** If the new message is a direct goal or result of preceding actions (e.g., creating a group, adding members), followed by (first project instruction in the group), they should be merged into one complete episode. Don't break in the middle of continuous processes.
- **`should_end` and `should_wait` are mutually exclusive:** If `should_end` is `true`, `should_wait` must be `false`. If `should_wait` is `true`, `should_end` must be `false`.
- **Consider context continuity:** Treat continuous, short-time "back-and-forth" conversations serving the same small goal as a whole.
- **System message context relevance:** Don't treat system messages (e.g., "[User X] joined the group") as topic switch signals by default. Their meaning is determined by the next human message. For example, if the next message is "Welcome! We were just discussing...", should merge; if the next message opens a completely new topic, consider splitting.

### Output Format
Please return your analysis strictly in the following JSON format:
```json
{{
    "reasoning": "One sentence explaining why you made the should_end and should_wait decisions.",
    "should_end": boolean,
    "should_wait": boolean,
    "confidence": float,
    "topic_summary": "Only fill in when should_end is true. Summarize the topic of the ended episode, otherwise empty string."
}}
```"""

CONV_SUMMARY_PROMPT = """
You are an episodic memory summary expert. You need to summarize the following conversation.
"""
