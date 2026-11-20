// 지식그래프 초기 제약·인덱스. 모든 문장은 IF NOT EXISTS로 여러 번 실행해도 안전하다.
// 엔티티 라벨은 common.schema.knowledge.EntityType 값과 정확히 일치해야 한다.

// --- 유니크 제약: 엔티티 canonical_name ---
CREATE CONSTRAINT system_canonical_name_unique IF NOT EXISTS
FOR (n:System) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT error_canonical_name_unique IF NOT EXISTS
FOR (n:Error) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT solution_canonical_name_unique IF NOT EXISTS
FOR (n:Solution) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT permission_canonical_name_unique IF NOT EXISTS
FOR (n:Permission) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT usergroup_canonical_name_unique IF NOT EXISTS
FOR (n:UserGroup) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT policy_canonical_name_unique IF NOT EXISTS
FOR (n:Policy) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT team_canonical_name_unique IF NOT EXISTS
FOR (n:Team) REQUIRE n.canonical_name IS UNIQUE;

// --- 유니크 제약: 원천 문서 ---
CREATE CONSTRAINT source_source_id_unique IF NOT EXISTS
FOR (n:Source) REQUIRE n.source_id IS UNIQUE;

// --- 조회용 인덱스 ---
CREATE INDEX error_code_idx IF NOT EXISTS
FOR (n:Error) ON (n.code);

CREATE INDEX solution_knowledge_type_idx IF NOT EXISTS
FOR (n:Solution) ON (n.knowledge_type);

CREATE INDEX policy_valid_from_idx IF NOT EXISTS
FOR (n:Policy) ON (n.valid_from);

CREATE INDEX policy_valid_to_idx IF NOT EXISTS
FOR (n:Policy) ON (n.valid_to);

CREATE INDEX source_source_type_idx IF NOT EXISTS
FOR (n:Source) ON (n.source_type);
