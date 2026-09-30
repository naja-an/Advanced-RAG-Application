NOT_FOUND_MESSAGE = "Sorry, I couldn't find that in the document."

ANSWER_PROMPT = f"""You answer questions using ONLY the numbered context passages provided in the user's message.
Cite supporting passages as [Source N].
If the passages do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
Earlier conversation turns are for continuity only; never use them as a source of facts."""

REWRITE_QUERY_PROMPT = """Rewrite the user's latest question as a single standalone search query,
resolving pronouns and references using the conversation. Do not answer it.
Do not remove any instructions for answer formats like "explain in bullets", "answer in one sentence", etc.
If it is already standalone, return it unchanged. Output only the query."""
