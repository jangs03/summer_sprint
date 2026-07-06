"""
팀원용: 본인 노트북에서 arXiv 논문을 수집해서 공유 서버(EC2)의 DB에 바로 반영.

collector.py / preprocess.py를 그대로 재사용하므로 무거운 임베딩 라이브러리
(chromadb, sentence-transformers, torch)는 설치할 필요 없음.
임베딩 계산은 서버가 담당함.

준비 (최초 1회):
    pip install -r requirements-client.txt   # requests + arxiv 만 설치됨
    cp local_config.example.py local_config.py
    # local_config.py에 API_BASE, API_KEY 입력

사용 예:
    python collect_and_push.py --days 7
    python collect_and_push.py --days 14 --categories cs.RO cs.CV
"""

import argparse
from datetime import datetime, timedelta, timezone

import requests

from collector import fetch_by_date_range_chunked
from preprocess import dedupe_raw_papers

try:
    from local_config import API_BASE, API_KEY
except ImportError:
    raise SystemExit(
        "local_config.py가 없습니다. local_config.example.py를 복사해서 "
        "local_config.py로 저장한 뒤 실제 값을 채워주세요."
    )

HEADERS = {"X-API-Key": API_KEY}
BATCH_SIZE = 200  # 한 번에 서버로 보내는 논문 수 (너무 크면 요청이 오래 걸림)


def _to_payload(p):
    return {
        "arxiv_id": p.arxiv_id,
        "version": p.version,
        "title": p.title,
        "authors": p.authors,
        "abstract_raw": p.abstract_raw,
        "categories": p.categories,
        "primary_category": p.primary_category,
        "comments": p.comments,
        "submitted_date": p.submitted_date.isoformat(),
        "updated_date": p.updated_date.isoformat(),
        "abs_url": p.abs_url,
        "pdf_url": p.pdf_url,
    }


def _push_batch(papers, total_changed, total_embedded, batch_label):
    for i in range(0, len(papers), BATCH_SIZE):
        chunk = papers[i : i + BATCH_SIZE]
        payload = {"papers": [_to_payload(p) for p in chunk], "auto_embed": True}

        resp = requests.post(
            f"{API_BASE}/papers/ingest", json=payload, headers=HEADERS, timeout=180
        )
        resp.raise_for_status()
        result = resp.json()
        total_changed += result["ingested_or_updated"]
        total_embedded += result["newly_embedded"]
        print(
            f"  {batch_label} 배치 {i // BATCH_SIZE + 1}: "
            f"신규/갱신 {result['ingested_or_updated']}건, 임베딩 {result['newly_embedded']}건"
        )
    return total_changed, total_embedded


def collect_and_push(days: int, categories: list, chunk_days: int = 7):
    """
    arXiv API는 한 쿼리의 페이지네이션이 약 10,000건을 넘어가면 HTTP 500을 낸다
    (arXiv 서버 쪽 제약). collector.fetch_by_date_range_chunked가 전체 기간을
    chunk_days 단위 창으로 쪼개서 순서대로 조회해주므로, 여기서는 창마다
    중복 제거 + 서버 전송만 하면 된다.
    """
    end = datetime.now(timezone.utc)
    overall_start = end - timedelta(days=days)

    total_changed = 0
    total_embedded = 0
    window_no = 0

    for window_start, window_end, raw_papers in fetch_by_date_range_chunked(
        overall_start, end, categories, chunk_days=chunk_days
    ):
        window_no += 1
        label = f"{window_start.date()}~{window_end.date()}"

        deduped = dedupe_raw_papers(raw_papers)
        print(f"[{window_no}] {label}: 조회 {len(raw_papers)}건 -> 중복 제거 후 {len(deduped)}건, 전송 시작")

        total_changed, total_embedded = _push_batch(
            deduped, total_changed, total_embedded, f"[{window_no}] {label}"
        )

    print(f"완료: 총 신규/갱신 {total_changed}건, 임베딩 {total_embedded}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="본인 노트북에서 arXiv 수집 후 공유 서버 DB에 반영"
    )
    parser.add_argument("--days", type=int, default=7, help="최근 N일치 수집 (기본 7일)")
    parser.add_argument(
        "--categories", nargs="+", default=["cs.RO", "cs.CV", "cs.CL"],
        help="수집할 카테고리 목록 (기본: cs.RO cs.CV cs.CL)",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=7,
        help="한 번의 arXiv 쿼리로 조회할 기간 단위 (기본 7일). "
             "결과가 너무 많으면(약 10,000건↑) arXiv API가 500 에러를 내므로, "
             "긴 기간을 수집할 땐 이 값을 줄이세요 (예: 3).",
    )
    args = parser.parse_args()
    collect_and_push(args.days, args.categories, args.chunk_days)
