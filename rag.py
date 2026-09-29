import logging

from langchain.agents import create_agent
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import BM25Retriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import SummarizationMiddleware, ToolCallRequest, ToolErrorMiddleware, ToolRetryMiddleware, ModelRetryMiddleware
from document_processor import normalize_for_lexical_search
from langfuse import observe

logger = logging.getLogger(__name__)


AGENT_PROMPT = """
        You answer questions using the document retrieval tool.
        Always call the retrieval tool before answering.
        Use only information returned by the tool. Cite supporting sources as [Source N].
        If the retrieved documents do not contain the answer, say
        "Sorry, I couldn’t find that in the document."
        """


class RAGPipeline:
    def __init__(
        self,
        llm,
        vectorstore,
        docs: list[Document],
        vector_k: int = 20,
        lexical_k: int = 20,
        final_k: int = 5,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.docs = docs
        reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_k})
        bm25_retriever = BM25Retriever.from_documents(self.docs, preprocess_func=normalize_for_lexical_search)
        bm25_retriever.k = lexical_k

        base_ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever])
        reranker = CrossEncoderReranker(model=reranker, top_n=final_k)

        self.smart_hybrid_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=base_ensemble
                )
        retriever_tool = StructuredTool.from_function(
            coroutine=self.retrieve_documents,
            name="retrieve_documents",
            description="Retrieve relevant documents from the vectorstore based on the query. Always use this tool to retrieve documents or information related to the question before generating an answer.",
            response_format="content_and_artifact",
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[retriever_tool],
            system_prompt=AGENT_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                SummarizationMiddleware(model=self.llm, trigger=("tokens", 5000), keep=("messages", 20)),
                ToolErrorMiddleware(aon_error=aon_error),
                ToolRetryMiddleware(max_retries=2, on_failure="error"),
                ModelRetryMiddleware(max_retries=2, on_failure="error"),
            ]
        )

    @staticmethod
    def _message_text(message) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()

    @staticmethod
    def _retrieved_documents(messages) -> list[Document]:
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].type == "human"
            ),
            -1,
        )
        documents = []
        for message in messages[last_user_index + 1:]:
            if message.type == "tool" and message.name == "retrieve_documents":
                for artifact in getattr(message, "artifact", None) or []:
                    documents.append(
                        artifact
                        if isinstance(artifact, Document)
                        else Document.model_validate(artifact)
                    )
        return documents

    @observe(name="rag.retrieve_documents")
    async def retrieve_documents(
        self,
        query: str
    ) -> tuple[str, list[Document]]:
        """
        Retrieve relevant documents from the vectorstore based on the query.
        Does a hybrid search using both vector search and bm25 keyword search.
        Combines the results and reranks them using a cross-encoder model.
        Args:
            query (str): The user's query.
        Returns:
            List[Document]: A list of retrieved documents.
        """
        documents = await self.smart_hybrid_retriever.ainvoke(query)
        logger.debug("retrieval_completed document_count=%d", len(documents))
        content = "\n\n".join(
            f"[Source {index}]\n{document.page_content}"
            for index, document in enumerate(documents, start=1)
        )
        return content, documents

    @observe(name="rag.generate_answer")
    async def generate_answer(self, query: str, config: RunnableConfig | None = None) -> dict:
        """
        Generate an answer to the user's query based on the retrieved documents.
        Args:
            query (str): The user's query.
            config (dict): Configurations for the agent observability.
        Returns:
            dict: A dictionary containing the answer and the retrieved documents.
        """
        try:
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]}, config=config)

            documents = self._retrieved_documents(response["messages"])

            return {
                "answer": self._message_text(response["messages"][-1]),
                "documents": documents,
            }
        except Exception as e:
            logger.error(f"answer_generation_failed: {e}")
            raise RAGException(f"Error occured while response generation: {e}")


async def aon_error(exc: Exception, request: ToolCallRequest) -> str | None:
    tool_name = request.tool_call["name"]
    if isinstance(exc, ConnectionError):
        return f"Tool `{request.tool_call['name']}` encountered a connection error."
    elif isinstance(exc, ValueError):
        return f"`{request.tool_call['name']}` failed: {type(exc).__name__}. Fix the input and retry."
    logger.error(
        "tool_call_failed tool=%s error_type=%s",
        tool_name,
        type(exc).__name__,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return None


class RAGException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
