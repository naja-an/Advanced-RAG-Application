import os

from langfuse import get_client
from langfuse.langchain import CallbackHandler


def create_langfuse_handler():
    """Create a Langfuse callback when tracing credentials are configured."""
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None
    return CallbackHandler()


def flush_langfuse() -> None:
    """Flush buffered Langfuse events before the process exits."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        get_client().flush()
