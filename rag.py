import os
import math
import tempfile
import numpy as np
import streamlit as st
import re

from uuid import uuid4
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "./chroma_db"
COLLECTION_NAME = "real_estate_agent_db"

_vector_store = None
_llm = None
_current_model = None
_bm25_index = None
_bm25_docs: List[Document] = []


def format_indian_currency(num: float) -> str:
    is_negative = num < 0
    num = abs(num)
    s = str(int(num))
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        remaining_reversed = remaining[::-1]
        groups = [remaining_reversed[i:i + 2] for i in range(0, len(remaining_reversed), 2)]
        formatted_remaining = ",".join(groups)[::-1]
        formatted = f"{formatted_remaining},{last_three}"
    return f"-₹{formatted}" if is_negative else f"₹{formatted}"


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.corpus_size, 1)
        self.doc_freqs: List[dict] = []
        self.idf: dict = {}
        self.doc_len: List[int] = []
        df: dict = {}
        for doc in corpus:
            freq: dict = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            self.doc_freqs.append(freq)
            self.doc_len.append(len(doc))
            for term in freq:
                df[term] = df.get(term, 0) + 1
        for term, n in df.items():
            self.idf[term] = math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1)

    def score(self, query_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_len[doc_idx]
        freq = self.doc_freqs[doc_idx]
        for term in query_terms:
            if term not in freq:
                continue
            idf = self.idf.get(term, 0)
            tf = freq[term]
            score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return score

    def get_top_n(self, query_terms: List[str], n: int = 5) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query_terms, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


def rerank_docs(query: str, docs: List[Document], embeddings, top_k: int = 5) -> List[Document]:
    if not docs:
        return docs
    q_emb = np.array(embeddings.embed_query(query))
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        return docs[:top_k]

    contents = [doc.page_content[:512] for doc in docs]
    d_embs = np.array(embeddings.embed_documents(contents))

    d_norms = np.linalg.norm(d_embs, axis=1)
    dot_products = np.dot(d_embs, q_emb)

    norms_product = d_norms * q_norm
    norms_product[norms_product == 0] = 1.0
    scores = dot_products / norms_product

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


def _get_embeddings():
    if not hasattr(_get_embeddings, "_inst"):
        _get_embeddings._inst = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _get_embeddings._inst


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=VECTORSTORE_DIR,
        )
    return _vector_store


def get_llm(model: str = "llama-3.3-70b-versatile"):
    global _llm, _current_model
    if _llm is None or model != _current_model:
        _llm = ChatGroq(model=model, temperature=0.1, max_tokens=1024)
        _current_model = model
    return _llm


def clear_database():
    global _bm25_index, _bm25_docs
    get_vector_store().reset_collection()
    _bm25_index = None
    _bm25_docs = []
    return "Property data cleared."


def _rebuild_bm25(docs: List[Document]):
    global _bm25_index, _bm25_docs
    _bm25_docs = docs
    tokenized = [tokenize_text(d.page_content) for d in docs]
    _bm25_index = BM25(tokenized)


def process_inputs(urls=None, pdf_files=None):
    vs = get_vector_store()
    documents = []

    if urls:
        yield "🌐 Reading property URLs..."
        loader = UnstructuredURLLoader(urls=urls)
        documents.extend(loader.load())

    if pdf_files:
        yield "📄 Processing PDFs..."
        for f in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                path = tmp.name
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = f.name
                if "Whitefield" in f.name:
                    d.metadata["price"] = 48000000
            documents.extend(docs)
            os.remove(path)

    if not documents:
        yield "⚠️ No data found."
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = f"Property Document: {chunk.metadata.get('source', 'Unknown')}\n\n{chunk.page_content}"

    ids = [str(uuid4()) for _ in chunks]
    vs.add_documents(chunks, ids=ids)

    yield "🔍 Building BM25 keyword index..."
    existing = _bm25_docs + chunks
    _rebuild_bm25(existing)

    yield f"✅ Indexed {len(chunks)} chunks. Property data ready."


def _reciprocal_rank_fusion(dense_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
    scores, seen = {}, {}

    def _key(doc: Document) -> str:
        return doc.page_content[:100]

    for rank, doc in enumerate(dense_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        seen[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        seen[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [seen[key] for key, _ in ranked]


def hybrid_retrieve(query: str, chroma_filter: Optional[Dict[str, Any]] = None, k: int = 10) -> List[Document]:
    vs = get_vector_store()

    # 1. Dense retrieval (Chroma handles the filter natively here)
    dense = vs.similarity_search(query, k=k, filter=chroma_filter)

    # 2. BM25 retrieval
    bm25_results: List[Document] = []
    if _bm25_index and _bm25_docs:
        tokens = tokenize_text(query)
        top_idxs = _bm25_index.get_top_n(tokens, n=k * 3)

        for i, score in top_idxs:
            doc = _bm25_docs[i]


            passed_filter = True
            if chroma_filter and "price" in chroma_filter and "$lt" in chroma_filter["price"]:
                max_price = chroma_filter["price"]["$lt"]
                doc_price = doc.metadata.get("price")
                if doc_price is not None and doc_price >= max_price:
                    passed_filter = False

            if passed_filter:
                bm25_results.append(doc)
                if len(bm25_results) >= k:
                    break

    # 3. RRF & Rerank
    fused = _reciprocal_rank_fusion(dense, bm25_results)
    reranked = rerank_docs(query, fused[:15], _get_embeddings(), top_k=5)
    return reranked


@dataclass
class ROIResult:
    purchase_price: float
    down_payment: float
    loan_amount: float
    monthly_mortgage: float
    monthly_rental: float
    monthly_expenses: float
    monthly_cash_flow: float
    annual_cash_flow: float
    cash_on_cash_return: float
    cap_rate: float
    gross_rent_multiplier: float
    five_year_equity: float
    five_year_appreciation: float
    total_roi_5yr: float


def calculate_roi(purchase_price: float, down_payment_pct: float = 20.0, interest_rate: float = 8.5,
                  loan_term_years: int = 20, monthly_rental: float = 0.0, vacancy_rate: float = 5.0,
                  annual_expenses: float = 0.0, appreciation_rate: float = 5.0) -> ROIResult:
    down = purchase_price * down_payment_pct / 100
    loan = purchase_price - down
    r = interest_rate / 100 / 12
    n = loan_term_years * 12
    mortgage = loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r else loan / n

    eff_rental = monthly_rental * (1 - vacancy_rate / 100)
    monthly_exp = annual_expenses / 12
    cash_flow = eff_rental - mortgage - monthly_exp
    annual_cf = cash_flow * 12

    coc = (annual_cf / down * 100) if down else 0
    noi = (eff_rental * 12) - annual_expenses
    cap = (noi / purchase_price * 100) if purchase_price else 0
    grm = (purchase_price / (monthly_rental * 12)) if monthly_rental else 0

    fv = purchase_price * (1 + appreciation_rate / 100) ** 5
    appreciation_gain = fv - purchase_price
    total_cf = annual_cf * 5

    if r > 0:
        months = min(60, n)
        equity_paid = loan * (((1 + r) ** months - 1) / ((1 + r) ** n - 1))
    else:
        equity_paid = mortgage * min(60, n)

    total_roi = ((appreciation_gain + total_cf + equity_paid) / down * 100) if down else 0

    return ROIResult(
        purchase_price=purchase_price, down_payment=down, loan_amount=loan, monthly_mortgage=mortgage,
        monthly_rental=monthly_rental, monthly_expenses=monthly_exp, monthly_cash_flow=cash_flow,
        annual_cash_flow=annual_cf, cash_on_cash_return=coc, cap_rate=cap, gross_rent_multiplier=grm,
        five_year_equity=equity_paid, five_year_appreciation=appreciation_gain, total_roi_5yr=total_roi,
    )


def format_roi_report(r: ROIResult) -> str:
    cash_flow_status = "🟢 **Great News!** This property generates positive cash flow. Your rent covers all expenses and puts money in your pocket." if r.monthly_cash_flow >= 0 else "🔴 **Warning:** This property has a negative cash flow. Your monthly expenses are higher than your rent. You will need to pay out-of-pocket to cover the difference."
    eff_rental = r.monthly_cash_flow + r.monthly_mortgage + r.monthly_expenses

    return f"""
### 📝 Plain-English Breakdown
{cash_flow_status}

#### 1. The Initial Purchase
* **Property Price:** {format_indian_currency(r.purchase_price)}
* **Your Cash Invested (Down Payment):** {format_indian_currency(r.down_payment)}
* **Bank Loan Amount:** {format_indian_currency(r.loan_amount)}

#### 2. Monthly Money In vs. Money Out
* **Money In (Rent after assumed vacancies):** {format_indian_currency(eff_rental)}
* **Money Out (Bank Mortgage):** {format_indian_currency(r.monthly_mortgage)}
* **Money Out (Taxes, Insurance, Maint.):** {format_indian_currency(r.monthly_expenses)}
* **Net Monthly Cash Flow:** **{format_indian_currency(r.monthly_cash_flow)}**

#### 3. Where Will You Be In 5 Years?
* **Property Value Increase (Appreciation):** {format_indian_currency(r.five_year_appreciation)}
* **Debt Paid Off by Tenants (Equity):** {format_indian_currency(r.five_year_equity)}
* **Cash Collected (5 Years of Cash Flow):** {format_indian_currency(r.annual_cash_flow * 5)}
* **Total 5-Year Profit:** **{format_indian_currency(r.five_year_appreciation + r.five_year_equity + (r.annual_cash_flow * 5))}**
"""


class RouteQuery(BaseModel):
    intent: str = Field(
        description="Must be 'roi' if the user is asking to calculate returns, cash flow, or profits. Must be 'rag' if they are asking about property details, zoning, or general queries."
    )
    purchase_price: Optional[float] = Field(
        description="Extracted purchase price in raw numbers (e.g., 4.8 crore -> 48000000).")
    monthly_rental: Optional[float] = Field(
        description="Extracted monthly rental in raw numbers (e.g., 1.2 lakh -> 120000).")
    down_payment: Optional[float] = Field(
        description="Extracted down payment. If it's a percentage, extract the number (e.g. 25). Do NOT confuse this with the rental amount.")
    max_price_filter: Optional[float] = Field(
        description="If the user asks for properties 'under' a certain price, extract that max price here.")
    property_type: Optional[str] = Field(description="Commercial or Residential, if explicitly stated.")


def generate_answer(query: str, model: str, persona: str) -> Tuple[str, List[Document]]:
    llm = get_llm(model)

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert real estate data extraction router. Convert Indian units (lakhs = 100,000, crores = 10,000,000) into raw floats."),
        ("human", "{query}"),
    ])

    structured_llm = llm.with_structured_output(RouteQuery)
    router_chain = router_prompt | structured_llm

    try:
        route_data = router_chain.invoke({"query": query})
    except Exception as e:
        route_data = RouteQuery(intent="rag", purchase_price=None, monthly_rental=None, down_payment=None,
                                max_price_filter=None, property_type=None)

    if route_data.intent == "roi" and route_data.purchase_price:
        price = route_data.purchase_price
        if price < 100000:
            return "❌ **Invalid property price detected.** Please provide a realistic property value.", []

        rental = route_data.monthly_rental if route_data.monthly_rental else price * 0.003

        down_pct = 20.0
        if route_data.down_payment:
            if route_data.down_payment <= 100:
                down_pct = route_data.down_payment
            else:
                down_pct = (route_data.down_payment / price) * 100

        annual_exp = price * 0.01
        result = calculate_roi(purchase_price=price, down_payment_pct=down_pct, monthly_rental=rental,
                               annual_expenses=annual_exp)
        return format_roi_report(result), []

    chroma_filter = {}
    if route_data.max_price_filter:
        chroma_filter["price"] = {"$lt": route_data.max_price_filter}

    docs = hybrid_retrieve(query, chroma_filter=chroma_filter if chroma_filter else None, k=10)

    tone = {
        "Investor": "Focus on ROI, risks, appreciation, and numbers.",
        "Homebuyer": "Focus on comfort, amenities, and lifestyle.",
        "Legal Expert": "Focus on zoning, contracts, and compliance.",
    }[persona]

    def format_docs(documents: List[Document]) -> str:
        if not documents:
            return "No relevant contextual property data discovered."
        return "\n\n".join(d.page_content for d in documents)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Real Estate AI Agent.\n"
         f"Tone/Persona Style: {tone}\n"
         "Instructions: Answer the user's query factually using ONLY the provided context below. "
         "If the context is insufficient, state it. Do not invent facts.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
            {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    answer = rag_chain.invoke(query)
    return answer, docs