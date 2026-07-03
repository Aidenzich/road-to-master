# 摘錄自官方實作 Zleap-AI/SAG-Benchmark（靜態檢視，未執行任何程式碼）。
# 來源：https://github.com/Zleap-AI/SAG-Benchmark
#   - pipeline/db/models.py                        L519-527（event_entity 多對多關聯表）
#   - pipeline/modules/search/step5_strategies.py  L91-104（query-time 擴展的 SQL join）
#   - pipeline/modules/search/config.py            L116-117, L418-419（部分預設值）
# 逐字保留原始程式碼與中文註解；此檔僅為 ledger 佐證用途。

# (1) 事項-實體關聯表：一個 event 對多個 entity，即論文所稱的 latent hyperedge。
class EventEntity(Base):
    """事項-实体关联表（多对多关系）"""

    __tablename__ = "event_entity"
    __table_args__ = (
        Index("uk_event_entity", "event_id", "entity_id", unique=True),
        Index("idx_event_id", "event_id"),
        Index("idx_entity_id", "entity_id"),
    )


# (2) query-time 擴展：以 frontier entity 反查新的 event，本質是一次關聯式 JOIN，
#     不是 PageRank / 圖推理。
async def _query_new_event_ids(self, entity_ids, exclude_ids, source_config_ids):
    stmt = select(EventEntity.event_id).where(
        EventEntity.entity_id.in_(entity_ids)
    ).distinct()
    if source_config_ids:
        stmt = stmt.join(
            SourceEvent, SourceEvent.id == EventEntity.event_id
        ).where(
            SourceEvent.source_config_id.in_(source_config_ids)
        )
    # ... 執行後回傳未見過的 event_id，作為下一跳候選


# (3) 與論文一致的兩個預設值（config.py）：直接向量召回門檻 0.4、擴展跳數 1。
#     similarity_threshold: float = Field(default=0.4, ...)
#     max_hops: int = Field(default=1, ...)   # [multi] 多跳扩展次数（0=不扩展，1=扩展1轮）
