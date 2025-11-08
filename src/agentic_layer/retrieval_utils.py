"""检索工具函数

提供多种检索策略的实现：
- Embedding 向量检索
- BM25 关键词检索
- RRF 融合检索
"""

import re
import time
import jieba
import numpy as np
from typing import List, Tuple
from core.nlp.stopwords_utils import filter_stopwords as filter_chinese_stopwords
from .vectorize_service import get_vectorize_service


def build_bm25_index(candidates):
    """构建 BM25 索引（支持中英文）"""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        return None, None, None, None
    
    # 确保 NLTK 数据已下载
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # 提取文本并分词（支持中英文）
    tokenized_docs = []
    for mem in candidates:
        text = getattr(mem, "episode", None) or getattr(mem, "summary", "") or ""
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if has_chinese:
            tokens = list(jieba.cut(text))
            processed_tokens = filter_chinese_stopwords(tokens)
        else:
            tokens = word_tokenize(text.lower())
            processed_tokens = [
                stemmer.stem(token)
                for token in tokens
                if token.isalpha() and len(token) >= 2 and token not in stop_words
            ]
        
        tokenized_docs.append(processed_tokens)
    
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs, stemmer, stop_words


async def search_with_bm25(
    query: str,
    bm25,
    candidates,
    stemmer,
    stop_words,
    top_k: int = 50
) -> List[Tuple]:
    """BM25 检索（支持中英文）"""
    if bm25 is None:
        return []
    
    try:
        from nltk.tokenize import word_tokenize
    except ImportError:
        return []
    
    # 分词查询（支持中英文）
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    
    if has_chinese:
        tokens = list(jieba.cut(query))
        tokenized_query = filter_chinese_stopwords(tokens)
    else:
        tokens = word_tokenize(query.lower())
        tokenized_query = [
            stemmer.stem(token)
            for token in tokens
            if token.isalpha() and len(token) >= 2 and token not in stop_words
        ]
    
    if not tokenized_query:
        return []
    
    # 计算 BM25 分数
    scores = bm25.get_scores(tokenized_query)
    
    # 排序并返回 Top-K
    results = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return results


def reciprocal_rank_fusion(
    results1: List[Tuple],
    results2: List[Tuple],
    k: int = 60
) -> List[Tuple]:
    """RRF 融合两个检索结果"""
    doc_rrf_scores = {}
    doc_map = {}
    
    # 处理第一个结果集
    for rank, (doc, score) in enumerate(results1, start=1):
        doc_id = id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 处理第二个结果集
    for rank, (doc, score) in enumerate(results2, start=1):
        doc_id = id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 转换为列表并排序
    fused_results = [
        (doc_map[doc_id], rrf_score)
        for doc_id, rrf_score in doc_rrf_scores.items()
    ]
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    return fused_results


async def lightweight_retrieval(
    query: str,
    candidates,
    emb_top_n: int = 50,
    bm25_top_n: int = 50,
    final_top_n: int = 20
) -> Tuple:
    """轻量级检索（Embedding + BM25 + RRF 融合）"""
    start_time = time.time()
    
    metadata = {
        "retrieval_mode": "lightweight",
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    if not candidates:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    # 构建 BM25 索引
    bm25, tokenized_docs, stemmer, stop_words = build_bm25_index(candidates)
    
    # Embedding 检索
    emb_results = []
    try:
        vectorize_service = get_vectorize_service()
        query_vec = await vectorize_service.get_embedding(query)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm > 0:
            scores = []
            for mem in candidates:
                try:
                    doc_vec = np.array(mem.extend.get("embedding", []))
                    if len(doc_vec) > 0:
                        doc_norm = np.linalg.norm(doc_vec)
                        if doc_norm > 0:
                            sim = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
                            scores.append((mem, float(sim)))
                except:
                    continue
            
            emb_results = sorted(scores, key=lambda x: x[1], reverse=True)[:emb_top_n]
    except Exception as e:
        pass
    
    metadata["emb_count"] = len(emb_results)
    
    # BM25 检索
    bm25_results = []
    if bm25 is not None:
        bm25_results = await search_with_bm25(
            query, bm25, candidates, stemmer, stop_words, top_k=bm25_top_n
        )
    
    metadata["bm25_count"] = len(bm25_results)
    
    # RRF 融合
    if not emb_results and not bm25_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    elif not emb_results:
        final_results = bm25_results[:final_top_n]
    elif not bm25_results:
        final_results = emb_results[:final_top_n]
    else:
        fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=60)
        final_results = fused_results[:final_top_n]
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    return final_results, metadata

