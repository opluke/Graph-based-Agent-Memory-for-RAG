# Query Pipeline Refactor Notes

這份文件整理本輪針對查詢管線所做的第一版重構，對應先前規劃的 6 個優先改動。

## 範圍

主要修改檔案：

- `memory/query_engine.py`
- `test_query_engine_rerank.py`

未在本輪直接修改，但會受行為影響的模組：

- `memory/answer_formatter.py`
- `memory/memory_builder.py`
- `memory/trg_memory.py`

## 1. 修正 `_rerank_and_filter()` 的 `filtered_nodes` bug

### 修改位置

- `memory/query_engine.py`
  - `_rerank_and_filter()`

### 變更內容

原本 `_rerank_and_filter()` 先做第一階段過濾：

- 人名過濾
- temporal 年份過濾

但後續真正進入打分與排序時，仍然迭代原始 `nodes`，導致第一階段過濾實際上沒有生效。

本輪修正為：

- 打分迴圈從 `for node in nodes` 改成 `for node in filtered_nodes`
- 保留原本的 fallback：
  - 如果 `filtered_nodes` 太少，仍回退到 `nodes`

### 更改前後程式碼

Before:

```python
filtered_nodes = []

for node in nodes:
    ...
    filtered_nodes.append(node)

if len(filtered_nodes) < top_k // 2:
    filtered_nodes = nodes

scored_nodes = []
for node in nodes:
    ...
```

After:

```python
filtered_nodes = []

for node in nodes:
    ...
    filtered_nodes.append(node)

if len(filtered_nodes) < top_k // 2:
    filtered_nodes = nodes

scored_nodes = []
for node in filtered_nodes:
    ...
```

### 驗證

- `test_rerank_filters_by_person_name_before_scoring`
- `test_rerank_filters_by_temporal_constraint_before_scoring`
- `test_rerank_falls_back_when_filter_is_too_strict`

## 2. 統一查詢分類入口：`QueryProfile`

### 修改位置

- `memory/query_engine.py`
  - 新增 `QueryProfile`
  - 新增 `build_query_profile()`
  - 修改 `detect_query_intent()`
  - 修改 `detect_query_type()`

### 變更內容

新增結構化分類結果 `QueryProfile`，統一輸出：

- `primary_type`
- `type_scores`
- `intent_scores`
- `entity_focus`
- `temporal_focus`
- `causal_focus`
- `aggregation_focus`
- `action_focus`
- `needs_actor_consistency`
- `needs_session_routing`
- `needs_path_reasoning`

原本兩套分類邏輯：

- `detect_query_intent()`
- `detect_query_type()`

改成共用 `build_query_profile()` 的結果。

目前舊函式保留為相容層：

- `detect_query_intent()` 從 `intent_scores` 取最大值
- `detect_query_type()` 回傳 `primary_type`

### 更改前後程式碼

Before:

```python
def detect_query_intent(self, question: str) -> str:
    q_lower = question.lower()
    if any(word in q_lower for word in ['why', 'because', 'cause', 'reason']):
        return 'WHY'
    if any(word in q_lower for word in ['when', 'time', 'date', 'before', 'after']):
        return 'WHEN'
    return 'ENTITY'

def detect_query_type(self, question: str) -> str:
    q_lower = question.lower()
    if any(multi_hop_patterns):
        return 'multi_hop'
    if any(phrase in q_lower for phrase in temporal_phrases):
        return 'temporal'
    ...
    return 'general'
```

After:

```python
@dataclass
class QueryProfile:
    primary_type: str
    type_scores: dict
    intent_scores: dict
    entity_focus: float
    temporal_focus: float
    causal_focus: float
    aggregation_focus: float
    action_focus: float
    needs_actor_consistency: bool
    needs_session_routing: bool
    needs_path_reasoning: bool

def build_query_profile(self, question: str) -> QueryProfile:
    ...
    return QueryProfile(...)

def detect_query_intent(self, question: str) -> str:
    profile = self.build_query_profile(question)
    return max(profile.intent_scores.items(), key=lambda item: item[1])[0]

def detect_query_type(self, question: str) -> str:
    return self.build_query_profile(question).primary_type
```

### 驗證

- `test_query_profile_unifies_type_and_intent`
- `test_query_profile_marks_actor_consistency_for_action_questions`

## 3. `query()` 只建立一次 profile，並下傳到核心階段

### 修改位置

- `memory/query_engine.py`
  - `query()`
  - `get_adaptive_params()`
  - `_adaptive_graph_traversal()`
  - `_probabilistic_beam_search()`
  - `_rerank_and_filter()`

### 變更內容

`query()` 現在只建立一次：

- `profile = self.build_query_profile(question)`

並將同一份 `profile` 傳給：

- `get_adaptive_params(profile)`
- `_adaptive_graph_traversal(..., profile=profile)`
- `_probabilistic_beam_search(..., profile=profile)`
- `_rerank_and_filter(..., profile=profile)`

另外 `QueryContext.metadata` 也新增：

- `query_profile`

方便後續 debug 與離線分析。

### 更改前後程式碼

Before:

```python
def query(self, question: str, top_k: int = 15):
    query_type = self.detect_query_type(question)
    adaptive_params = self.get_adaptive_params(query_type)
    ...
    traversed = self._adaptive_graph_traversal(
        anchor_nodes=initial_top,
        question=question,
        similarity_threshold=adaptive_params.get('similarity_threshold', 0.3),
        ...
    )
    ...
    top_nodes = self._rerank_and_filter(
        all_candidates,
        question,
        top_k,
        query_type=query_type,
        scoring_weights=adaptive_params.get('scoring_weights', {})
    )
```

After:

```python
def query(self, question: str, top_k: int = 15):
    profile = self.build_query_profile(question)
    query_type = profile.primary_type
    adaptive_params = self.get_adaptive_params(profile)
    ...
    traversed = self._adaptive_graph_traversal(
        anchor_nodes=initial_top,
        question=question,
        profile=profile,
        similarity_threshold=adaptive_params.get('similarity_threshold', 0.3),
        ...
    )
    ...
    top_nodes = self._rerank_and_filter(
        all_candidates,
        question,
        top_k,
        query_type=query_type,
        profile=profile,
        scoring_weights=adaptive_params.get('scoring_weights', {})
    )
```

## 4. 將 beam search 接進主流程

### 修改位置

- `memory/query_engine.py`
  - `query()`
  - `_probabilistic_beam_search()`

### 變更內容

原本 `_probabilistic_beam_search()` 已存在，但沒有實際接進 `query()` 主流程。

本輪整合方式：

1. 先保留 `_adaptive_graph_traversal()` 作為第一階段候選擴展
2. 若 `profile.needs_path_reasoning == True`
   - 對高分 anchor 跑 `_probabilistic_beam_search()`
3. 將 beam score 回灌到 `all_candidates`
   - 已存在候選：提升 `similarity_score`
   - 尚未存在候選：直接補進 `all_candidates`

### 注意

這一版是低風險整合：

- 仍以 node ranking 為主
- 尚未全面升級成「以 path 為一級排序單位」

### 更改前後程式碼

Before:

```python
if not self.ablation_config.get('basic_retrieval') and all_candidates:
    traversed = self._adaptive_graph_traversal(...)
    for node, similarity in traversed:
        if node.node_id not in existing_ids:
            node.similarity_score = similarity
            all_candidates.append(node)
            existing_ids.add(node.node_id)
```

After:

```python
if not self.ablation_config.get('basic_retrieval') and all_candidates:
    traversed = self._adaptive_graph_traversal(...)
    for node, similarity in traversed:
        if node.node_id not in existing_ids:
            node.similarity_score = similarity
            all_candidates.append(node)
            existing_ids.add(node.node_id)

    if profile.needs_path_reasoning and initial_top:
        beam_results = self._probabilistic_beam_search(
            anchor_nodes=initial_top[:beam_seed_count],
            question=question,
            profile=profile,
            ...
        )
        beam_score_map = {node.node_id: score for node, score in beam_results}
        ...
```

Before:

```python
def _probabilistic_beam_search(
    self,
    anchor_nodes: List[EventNode],
    question: str,
    query_intent: str,
    ...
):
    attention_weights = {
        'WHY': {...},
        'WHEN': {...},
        'ENTITY': {...}
    }
    w_tq = attention_weights.get(query_intent, attention_weights['ENTITY'])
```

After:

```python
def _probabilistic_beam_search(
    self,
    anchor_nodes: List[EventNode],
    question: str,
    query_intent: Optional[str] = None,
    profile: Optional[QueryProfile] = None,
    ...
):
    if profile is None:
        profile = self.build_query_profile(question)

    dominant_intent = query_intent or max(profile.intent_scores.items(), key=lambda item: item[1])[0]
    ...
    w_tq = {
        'CAUSAL': max(default_attention.get('CAUSAL', 0.0), profile.causal_focus),
        'TEMPORAL': max(default_attention.get('TEMPORAL', 0.0), profile.temporal_focus),
        'ENTITY': max(default_attention.get('ENTITY', 0.0), profile.entity_focus),
        'SEMANTIC': max(default_attention.get('SEMANTIC', 0.0), 0.2 + profile.aggregation_focus * 0.3),
    }
```

### 驗證

- `test_probabilistic_beam_search_accepts_query_profile`

## 5. Session routing 改成 session retriever

### 修改位置

- `memory/query_engine.py`
  - `_identify_target_sessions()`
  - 新增 `_retrieve_relevant_session_nodes()`

### 變更內容

原本 `_identify_target_sessions()` 內有硬編碼 session map，例如：

- `may -> [1]`
- `july -> [6, 7, 8, 10]`
- `adoption -> [2, 17]`

這段已移除。

現在改成動態 session routing，依據三種訊號：

1. `SESSION` 節點的 `summary`
2. `entity_session_map`
3. 當前候選 event 的 `session_id` 與 `similarity_score`

流程如下：

1. `_retrieve_relevant_session_nodes(question, nodes)`
2. 對 session 做加權排序
3. 回傳排序後的 target session ids

### 更改前後程式碼

Before:

```python
temporal_keywords = {
    'may': [1],
    'june': [3],
    'july': [6, 7, 8, 10],
    'august': [15],
    'september': [16],
    'october': [18],
    'charity race': [2],
    'lgbtq': [1, 7],
    'adoption': [2, 17],
}

for keyword, sessions in temporal_keywords.items():
    if keyword in question_lower:
        target_sessions.extend(sessions)
```

After:

```python
def _identify_target_sessions(self, question: str, nodes: List[EventNode]) -> List[int]:
    session_nodes = self._retrieve_relevant_session_nodes(question, nodes)
    target_sessions = []
    for session_node in session_nodes:
        session_id = getattr(session_node, 'session_id', None)
        ...
        target_sessions.append(int(session_id))
    return target_sessions

def _retrieve_relevant_session_nodes(self, question: str, nodes: List[EventNode], top_k: int = 3) -> List:
    ...
    for node in self.trg.graph_db.nodes.values():
        if hasattr(node, 'node_type') and 'SESSION' in str(node.node_type):
            ...
            if term in summary_lower:
                session_scores[session_id_int] += 3.0
    ...
    for session_id, support_score in candidate_support.items():
        session_scores[session_id] = session_scores.get(session_id, 0.0) + support_score
```

### 驗證

- `test_identify_target_sessions_uses_session_summaries`
- `test_identify_target_sessions_uses_candidate_event_support`

## 6. Full scan 降級為真正 fallback，新增 sparse retrieval

### 修改位置

- `memory/query_engine.py`
  - `query()`
  - 新增 `_sparse_retrieval()`
  - 新增 `_should_use_full_scan()`
  - 保留 `_scan_all_nodes()`

### 變更內容

原本 `query()` 每次都會跑：

- vector
- keyword
- full scan

本輪改成：

- vector
- keyword
- sparse retrieval

只有在 recall 不足時，才觸發：

- `_scan_all_nodes()`

新增 `_sparse_retrieval()`：

- 直接使用 `node_index`
- 支援 term match
- partial match
- bigram match
- 以近似 sparse scoring 排序，不掃整張圖

新增 `_should_use_full_scan()`：

- 根據 `ranked_lists`
- `candidate_count`
- `top_k`
- `profile.needs_path_reasoning`

來決定是否要 fallback 到 full scan

### 更改前後程式碼

Before:

```python
keyword_nodes = self._keyword_search(question)[:keyword_size]
if keyword_nodes:
    ranked_lists.append(keyword_nodes)

scan_nodes = self._scan_all_nodes(question)[:scan_size]
if scan_nodes:
    ranked_lists.append(scan_nodes)
```

After:

```python
keyword_nodes = self._keyword_search(question)[:keyword_size]
if keyword_nodes:
    ranked_lists.append(keyword_nodes)

sparse_nodes = self._sparse_retrieval(question, top_k=sparse_size)[:sparse_size]
if sparse_nodes:
    ranked_lists.append(sparse_nodes)

if self._should_use_full_scan(ranked_lists, len(all_candidates), top_k, profile):
    scan_nodes = self._scan_all_nodes(question)[:scan_size]
    if scan_nodes:
        ranked_lists.append(scan_nodes)
        fused_results = self._rrf_fusion(ranked_lists, k=60)
```

Before:

```python
def _scan_all_nodes(self, question: str) -> List[EventNode]:
    question_lower = question.lower()
    words = [...]
    relevant_nodes = []
    for node in self.trg.graph_db.nodes.values():
        ...
```

After:

```python
def _sparse_retrieval(self, question: str, top_k: int = 40) -> List[EventNode]:
    ...
    for term in terms:
        add_match(term, 3.0)
        if len(term) >= 4:
            for key in self.node_index.keys():
                if term in key or key in term:
                    add_match(key, 1.5, is_partial=True)
    ...

def _should_use_full_scan(...):
    if candidate_count < minimum_candidates:
        return True
    if non_empty_lists <= 1 and candidate_count < minimum_candidates * 2:
        return True
    return False
```

### 驗證

- `test_sparse_retrieval_uses_index_without_full_scan`
- `test_full_scan_only_used_when_recall_is_low`

## 7. 固定鄰居擴展改成缺口驅動 evidence expansion

### 修改位置

- `memory/query_engine.py`
  - 新增 `_identify_evidence_gaps()`
  - 新增 `_gap_driven_evidence_expansion()`
  - `query()` 在 `_rerank_and_filter()` 後接入 gap-driven expansion

### 變更內容

原本擴展策略偏固定：

- `_get_neighbors()` 會強調 `CONTEXT_NEIGHBOR`
- `_expand_qa_context()` 會對 question node 補 temporal successor

本輪新增一層缺口驅動擴展：

1. `_identify_evidence_gaps()`
   - 檢查目前 top nodes 是否缺：
     - `temporal`
     - `actor`
     - `causal`
2. `_gap_driven_evidence_expansion()`
   - 只補足對應類型的 neighbor

補證據規則：

- 缺 temporal：
  - `TEMPORAL`
  - `PRECEDES`
  - `SUCCEEDS`
  - `TEMPORALLY_CLOSE`
- 缺 actor：
  - speaker / entity 與問題主體一致的節點
- 缺 causal：
  - `CAUSAL`
  - `ANSWERED_BY`
  - `RESPONSE_TO`
  - `LEADS_TO`
  - `BECAUSE_OF`

### 額外修正

在 temporal gap 判定中，不能把每個 `EventNode` 預設的 `timestamp` 視為已具備 temporal evidence。

現在 temporal evidence 主要依賴：

- `dates_mentioned`
- 明確年份
- 日期格式
- 相對時間詞

### 更改前後程式碼

Before:

```python
if top_nodes:
    top_nodes = self._expand_qa_context(top_nodes)
```

After:

```python
if top_nodes:
    top_nodes = self._gap_driven_evidence_expansion(
        question=question,
        nodes=top_nodes,
        profile=profile,
        top_k=top_k,
    )

if top_nodes:
    top_nodes = self._expand_qa_context(top_nodes)
```

After:

```python
def _identify_evidence_gaps(self, question: str, nodes: List[EventNode], profile: QueryProfile) -> Set[str]:
    ...
    if not temporal_evidence:
        gaps.add('temporal')
    if not actor_evidence:
        gaps.add('actor')
    if not causal_evidence:
        gaps.add('causal')
    return gaps

def _gap_driven_evidence_expansion(self, question: str, nodes: List[EventNode], profile: QueryProfile, top_k: int = 15) -> List[EventNode]:
    gaps = self._identify_evidence_gaps(question, nodes, profile)
    ...
    if 'temporal' in gaps and (...):
        include_neighbor = True
    if 'actor' in gaps and ...:
        include_neighbor = True
    if 'causal' in gaps and (...):
        include_neighbor = True
```

### 驗證

- `test_gap_driven_expansion_adds_temporal_evidence_when_missing`
- `test_gap_driven_expansion_adds_actor_evidence_when_missing`

## 現在的 `query()` 高層流程

目前主流程可簡化成：

1. `build_query_profile()`
2. `vector + keyword + sparse retrieval`
3. recall 不足才 `full scan`
4. `RRF`
5. `SESSION` routing boost
6. `_adaptive_graph_traversal()`
7. 若需要 path reasoning，跑 `_probabilistic_beam_search()`
8. `_rerank_and_filter()`
9. `_gap_driven_evidence_expansion()`
10. `_expand_qa_context()`
11. `_expand_session_context()`
12. format answer context

## 測試

本輪新增/更新的測試檔：

- `test_query_engine_rerank.py`

目前涵蓋：

- rerank filter 生效
- query profile 統一分類
- actor consistency 標記
- beam search 接入 profile
- session retriever
- sparse retrieval
- full scan fallback 判定
- gap-driven evidence expansion

執行方式：

```bash
python -m unittest test_query_engine_rerank.py
```

## 尚未完成的部分

這一輪是第一版落地，仍有幾個明確未完成項：

- `_probabilistic_beam_search()` 已接入主流程，但最終排序仍以 node 為主，不是完整 path-level ranking
- actor consistency 目前仍偏強化版 rerank / evidence 補強，尚未升級成完整 subject-action-object gate
- sparse retrieval 仍是 index-based heuristic scoring，不是真正 BM25 / SPLADE
- `_get_neighbors()` 本身仍保留 `CONTEXT_NEIGHBOR` 優先邏輯，現階段是由 gap-driven expansion 在主流程中補上更細的控制

## 建議下一步

如果要再往下收斂，我建議順序：

1. 把 actor consistency 從 boost 升級成明確 evidence gate
2. 將 beam search 的輸出升級成 path-level rerank，而不是只回灌 node score
3. 將 sparse retrieval 換成真正的 BM25 或倒排索引檢索器
