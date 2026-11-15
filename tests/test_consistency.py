"""已发布结果一致性：查询快照与发布响应计数。

用可控的 SQLite 连接代理把并发交错固定在关键位置：
- 查询先读到数据集行（旧版），随后另一发布提交新版 -> 位点行不得混入新版；
- 发布提交后、统计计数前，另一发布提交 -> 本次响应计数仍属于本次发布。
"""
import pytest
from fastapi.testclient import TestClient

from app.service import NormVariant
from app.store import Store


class _ConnectionProxy:
    """透明代理 sqlite3.Connection，在指定 SQL/提交前后插入回调。"""

    def __init__(self, conn, on_execute=None, on_commit=None):
        self._conn = conn
        self._on_execute = on_execute
        self._on_commit = on_commit

    def execute(self, sql, *args, **kwargs):
        if self._on_execute is not None:
            self._on_execute(sql)
        return self._conn.execute(sql, *args, **kwargs)

    def commit(self):
        self._conn.commit()
        if self._on_commit is not None:
            self._on_commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _variants(prefix: int, n: int) -> list[NormVariant]:
    return [
        NormVariant(pos=prefix + i, ref="A", alt="T", lines=[i + 1])
        for i in range(n)
    ]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """启动应用，并直接构造三个就绪任务（旧版 1 / 新版 4 / 另一版 2 个位点）。"""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "consistency.db"))
    from app.main import app

    with TestClient(app) as client:
        store = client.app.state.store
        did = store.create_dataset("ds")["id"]
        old_job = store.create_job(
            did, "old.fa", "old.vcf", "chr1", 1, [], _variants(10, 1)
        )
        new_job = store.create_job(
            did, "new.fa", "new.vcf", "chr1", 4, [], _variants(100, 4)
        )
        other_job = store.create_job(
            did, "other.fa", "other.vcf", "chr1", 2, [], _variants(200, 2)
        )
        assert store.publish(did, old_job["id"], 0)["version"] == 1
        yield client, store, did, old_job["id"], new_job["id"], other_job["id"]


def test_detail_reads_are_one_snapshot(env):
    """读取到的版本/任务号/位点/数量必须来自同一已提交版本。"""
    client, store, did, old_job_id, new_job_id, _other_job_id = env
    writer = Store(store.path)  # 独立连接，模拟另一发布请求

    fired = []

    def on_execute(sql: str) -> None:
        # 数据集行（v1）已读，位点行尚未读：此时另一请求发布 v2
        if not fired and "FROM variants" in sql:
            fired.append(True)
            assert writer.publish(did, new_job_id, 1)["version"] == 2

    original_connect = store._connect
    store._connect = lambda: _ConnectionProxy(original_connect(), on_execute=on_execute)
    try:
        body = client.get(f"/datasets/{did}").json()
    finally:
        store._connect = original_connect

    assert fired, "未触发并发发布"
    # 读事务快照：全部字段仍是完整旧版 v1
    assert body["version"] == 1
    assert body["published_job_id"] == old_job_id
    assert body["chrom"] == "chr1"
    assert body["variant_count"] == 1
    assert [(v["pos"], v["ref"], v["alt"]) for v in body["variants"]] == [(10, "A", "T")]

    # 读取期间的新版本发布已成功提交（读快照未阻塞写）
    latest = writer.get_dataset_detail(did)
    assert latest["version"] == 2
    assert latest["published_job_id"] == new_job_id
    assert latest["variant_count"] == 4

    # 后续查询完整看到新版
    after = client.get(f"/datasets/{did}").json()
    assert after["version"] == 2
    assert after["published_job_id"] == new_job_id
    assert after["variant_count"] == 4


def test_publish_response_count_is_own_publish(env):
    """发布响应的 variant_count 属于本次发布，不统计随后提交的新版。"""
    client, store, did, old_job_id, new_job_id, other_job_id = env
    writer = Store(store.path)

    fired = []

    def on_commit() -> None:
        # 本次发布已提交（v2），另一请求随后发布 v3（2 个位点）
        if not fired:
            fired.append(True)
            assert writer.publish(did, other_job_id, 2)["version"] == 3

    original_connect = store._connect
    store._connect = lambda: _ConnectionProxy(original_connect(), on_commit=on_commit)
    try:
        r = client.post(
            f"/datasets/{did}/publish",
            json={"job_id": new_job_id, "expected_version": 1},
        )
    finally:
        store._connect = original_connect

    assert fired, "未触发并发发布"
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 2
    assert body["published_job_id"] == new_job_id
    assert body["variant_count"] == 4  # 不是随后提交的版本数据


def test_publish_count_isolated_from_followup_publish(env):
    """发布 v2 的响应只统计本次位点，即使紧接着发布 v3 只有 1 个位点。"""
    client, store, did, old_job_id, new_job_id, other_job_id = env
    writer = Store(store.path)

    fired = []

    def on_commit() -> None:
        # 紧随其后发布仅 1 个位点的旧任务，作为 v3
        if not fired:
            fired.append(True)
            assert writer.publish(did, old_job_id, 2)["version"] == 3

    original_connect = store._connect
    store._connect = lambda: _ConnectionProxy(original_connect(), on_commit=on_commit)
    try:
        r = client.post(
            f"/datasets/{did}/publish",
            json={"job_id": new_job_id, "expected_version": 1},
        )
    finally:
        store._connect = original_connect

    assert fired
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 2
    assert r.json()["variant_count"] == 4
    assert writer.get_dataset_detail(did)["variant_count"] == 1
