"""FastAPI endpoints for the RAG chatbot."""

import asyncio
import logging
import os
from time import perf_counter
from typing import Annotated
from uuid import UUID

from dotenv import load_dotenv
from fastapi import File, FastAPI, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from document_processor import (
	DocumentInputException,
	create_vectorstore,
	ingest_sources,
)
from evaluation import evaluate_answer
from models import (
	ChatRequest,
	ChatResponse,
	DocumentResponse,
	EvaluationResponse,
	ErrorResponse,
	SessionResponse,
	SourceResponse,
)
from observability import create_langfuse_handler
from rag import RAGPipeline, RAGException
from session_manager import SessionManager, SessionNotFoundError
from logging_config import configure_logging

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
	title="RAG Chatbot API",
	version="0.1.0",
	description="Session-based API for document-grounded conversations.",
)
session_manager = SessionManager()
UI_PATH = __file__.replace("api.py", "static/index.html")
langfuse_handler = create_langfuse_handler()


def _build_pipeline(documents: list[object]) -> RAGPipeline:
	llm = ChatGoogleGenerativeAI(
		model=os.getenv("RAG_MODEL", "gemini-3.5-flash-lite"),
		temperature=0,
		max_output_tokens=512,
	)
	embedding = GoogleGenerativeAIEmbeddings(
		model=os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-2-preview")
	)
	vectorstore = create_vectorstore(documents, embedding=embedding)
	return RAGPipeline(
		llm=llm,
		vectorstore=vectorstore,
		docs=documents,
	)


@app.get("/ui", include_in_schema=False)
async def test_ui() -> FileResponse:
	return FileResponse(UI_PATH)


def _session_or_404(session_id: UUID):
	try:
		return session_manager.get(session_id)
	except SessionNotFoundError as error:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Session {session_id} was not found",
		) from error


@app.get("/health", response_model=dict[str, str])
async def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session() -> SessionResponse:
	session = session_manager.create()
	logger.info("session_created session_id=%s", session.session_id)
	return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID) -> None:
	_session_or_404(session_id)
	session_manager.delete(session_id)
	logger.info("session_deleted session_id=%s", session_id)


@app.get(
	"/sessions/{session_id}/documents",
	response_model=list[DocumentResponse],
	responses={404: {"model": ErrorResponse}},
)
async def list_documents(session_id: UUID) -> list[DocumentResponse]:
	_session_or_404(session_id)
	return session_manager.list_documents(session_id)


@app.post(
	"/sessions/{session_id}/documents",
	response_model=list[DocumentResponse],
	status_code=status.HTTP_202_ACCEPTED,
	responses={404: {"model": ErrorResponse}},
)
async def add_documents(
	session_id: UUID,
	files: Annotated[list[UploadFile] | None, File()] = None,
	urls: Annotated[list[str] | None, Form()] = None,
) -> list[DocumentResponse]:
	session = _session_or_404(session_id)
	file_sources = [
		(upload.filename or "upload", await upload.read())
		for upload in (files or [])
		if upload.filename
	]
	urls = [url.strip() for url in (urls or []) if url.strip()]
	try:
		results = await asyncio.to_thread(
			ingest_sources,
			file_sources,
			urls,
			session.storage_dir,
		)
	except DocumentInputException as error:
		logger.warning(
			"document_ingestion_rejected session_id=%s error_type=%s",
			session_id,
			type(error).__name__,
		)
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(error),
		) from error

	responses = [
		DocumentResponse(
			document_id=result.document_id,
			name=result.name,
			source=result.source,
			status=("completed" if not result.error else "failed"),
			error=result.error,
		)
		for result in results
	]
	failed_count = sum(result.status == "failed" for result in responses)
	logger.info(
		"document_ingestion_completed session_id=%s source_count=%d succeeded=%d failed=%d",
		session_id,
		len(responses),
		len(responses) - failed_count,
		failed_count,
	)
	session_manager.add_documents(
		session_id,
		responses,
		[
			document
			for result in results
			for document in result.documents
		],
	)
	if session.loaded_documents:
		try:
			pipeline = await asyncio.to_thread(
				_build_pipeline,
				session.loaded_documents,
			)
			session_manager.set_pipeline(session_id, pipeline)
		except Exception as error:
			logger.exception("rag_pipeline_build_failed session_id=%s", session_id)
			raise HTTPException(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				detail=f"Unable to build the retrieval pipeline: {error}",
			) from error
	return responses


@app.post(
	"/sessions/{session_id}/chat",
	response_model=ChatResponse,
	responses={404: {"model": ErrorResponse}},
)
async def chat(session_id: UUID, request: ChatRequest) -> ChatResponse:
	started_at = perf_counter()
	session = _session_or_404(session_id)
	if session.pipeline is None:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Upload at least one document before starting a chat",
		)

	config = {
		"callbacks": [langfuse_handler] if langfuse_handler else [],
		"configurable": {
			"thread_id": str(session_id),
		},
		"metadata": {
			"langfuse_session_id": str(session_id),
		},
	}
	try:
		result = await session.pipeline.generate_answer(
			request.question,
			config=config,
		)
	except RAGException as error:
		logger.warning("chat_generation_failed session_id=%s", session_id)
		raise HTTPException(
			status_code=status.HTTP_502_BAD_GATEWAY,
			detail=str(error),
		) from error
	try:
		evaluation = await asyncio.to_thread(
			evaluate_answer,
			request.question,
			result["answer"],
			[document.page_content for document in result["documents"]],
		)
	except Exception as error:
		logger.exception("answer_evaluation_unavailable session_id=%s", session_id)
		evaluation = {
			"answer_relevance": None,
			"faithfulness": None,
			"error": f"Evaluation unavailable: {type(error).__name__}",
		}
	evaluation["latency_ms"] = round((perf_counter() - started_at) * 1000, 1)
	logger.info(
		"chat_completed session_id=%s source_count=%d duration_ms=%d",
		session_id,
		len(result["documents"]),
		int(evaluation["latency_ms"]),
	)

	return ChatResponse(
		question=request.question,
		query_used_for_retrieval=request.question,
		answer=result["answer"],
		evaluation=EvaluationResponse(**evaluation),
		sources=[
			SourceResponse(
				metadata=document.metadata,
				content=document.page_content,
			)
			for document in result["documents"]
		],
	)

if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8000)
