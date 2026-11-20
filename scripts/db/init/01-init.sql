-- postgres 컨테이너 최초 기동 시(빈 볼륨일 때만) 1회 실행된다.
CREATE EXTENSION IF NOT EXISTS vector;

-- 지식 저장소 (청크·임베딩 등) — agent-api, ingest
CREATE SCHEMA IF NOT EXISTS knowledge;

-- mock 업무 시스템 (계정·티켓·권한 신청) — ops-api
CREATE SCHEMA IF NOT EXISTS ops;
