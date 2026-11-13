# SICA 개선 계획 (2026-11-12)

대상: `self_improving_coding_agent` (runner.py + base_agent/)
진행 순서: **Phase 3 탐색 전략 → Phase 1 기반 안정화 → Phase 2 에이전트 능력**

## 0. 현재 상태 요약

루프는 `runner.py:main()`에 있고 한 iteration은 다음과 같다.

1. `run_benchmarks_with_job_queue` — 벤치마크마다 `seed=1, subset_size=18` 문제를 문제당 컨테이너 1개로 실행
2. `run_meta_agent_benchmark` — `select_base_agent`가 부모 1개를 고르고, 그 코드를 `agent_{i+1}/agent_code`로 복사한 뒤, 최고 점수 에이전트(`archive.agent_{best_iter}.agent_code`)가 그 복사본을 수정
3. `metadata.json`의 `agent_iteration`을 +1

계획에 직접 영향을 주는 사실:

| # | 사실 | 위치 |
|---|------|------|
| F1 | 부모 선택은 "최신부터 거슬러 올라가며 best의 CI 하한 이상인 첫 에이전트" 하나. 계보는 사실상 한 줄 | `runner.py:88-147` |
| F2 | 어떤 부모에서 파생됐는지 저장하지 않음 (로그에만 출력) | `runner.py:771-773` |
| F3 | CI는 문제별 점수의 독립 표본 z-구간. 같은 18문제를 쓰는데도 대응(paired) 비교를 하지 않음 | `archive_analysis.py:780-845` |
| F4 | 문제당 1회 실행, 벤치마크당 18문제. 표본이 작아 CI가 넓음 | `runner.py:669` |
| F5 | 메타 개선 실패 시 `return`만 하고 루프는 iteration을 증가시킴. 생성된 에이전트의 테스트 통과 여부를 runner가 확인하지 않음 | `runner.py:877-880`, `1037-1044` |
| F6 | 메타 에이전트 컨테이너 포트 8080 고정, 워커 포트 `808{worker_id}` (워커 ≥10이면 잘못된 포트) | `runner.py:838`, `423` |
| F7 | API 키를 build ARG → ENV로 이미지에 저장 | `sandbox/Dockerfile:139-151`, `runner.py:936-962` |
| F8 | 타임아웃 600s, 비용 $3, subset 18이 코드 상수. 벤치마크 선택은 주석 해제 | `runner.py:367-368`, `benchmarks/__init__.py` |
| F9 | utility의 `time_limit`은 300s인데 runner 타임아웃은 600s | `archive_analysis.py:66` |
| F10 | 결과 전달이 stdout 마지막 줄을 `\|`로 분리하는 방식 | `runner.py:510`, `882` |
| F11 | 편집 도구는 `OverwriteFile` 하나 | `tools/edit_tools/` |
| F12 | 기본 모델 `SONNET_37` / `O3_MINI`, `anthropic[vertex]==0.42.0` 고정 | `config.py`, `requirements.txt` |
| F13 | CI 파이프라인 없음. 테스트 파일 14개, `make test` = `pytest base_agent` | `Makefile` |

---

## Phase 3 — 탐색 전략 (먼저 진행)

목표: 단일 계보 hill-climbing을 **계보 트리 + 확률적 부모 샘플링 + 분산을 줄인 평가**로 바꾼다. README의 미해결 과제 "reduce the variance of self-improvement runs"가 직접 대상이다.

### 3-0. 선행 최소 작업 (구현 완료)

순서를 3 → 1로 잡았지만 아래 두 가지는 3-3 이후가 동작하려면 먼저 필요하다. Phase 1에서 가져와 여기서 처리한다.

- **포트 동적 할당** (F6): `-p 8080:8080` / `808{worker_id}`를 `-p 127.0.0.1::8080`(Docker가 빈 포트 선택)으로 바꾸고, 필요하면 `docker port`로 조회해 로그에 남긴다. 자식 여러 개를 병렬로 만들려면 필수.
- **실패한 자식 표시** (F5의 일부): 메타 개선이 실패하면 `agent_{i+1}`을 `status: "failed"`로 기록하고 부모 후보에서 제외. 전체 검증 게이트는 1-1에서 완성한다.

### 3-1. 계보 기록 (구현 완료, change log 항목만 3-3으로 연기)

- `results/run_{id}/agent_{i}/lineage.json` 추가: `{id, parent_id, meta_agent_id, status, created_at, generation}`.
- `agent_{i}` 디렉터리 구조는 그대로 둔다. `ArchiveAnalyzer`와 컨테이너 안 아카이브 도구(`archive_tools.py`)가 이 구조를 읽기 때문이다.
- `ArchiveAnalyzer`에 `get_lineage()`, `get_children(i)`, `get_ancestors(i)` 추가. `analyze_run_formatted()` 표에 `parent` 열 추가.
- `agent_change_log.md` 복사 규칙은 **3-3으로 연기**: 지금은 "현재 에이전트"의 로그를 복사한다. 부모의 로그만 복사하면 계보별 이력은 맞지만, 다른 가지에서 이미 시도하고 버린 아이디어 기록이 사라져 같은 실패를 반복할 수 있다. 세대당 자식이 1개인 동안은 현재 방식이 더 안전하므로, 다중 자식을 도입하는 3-3에서 "부모 로그 + 다른 가지 시도 요약" 형태로 함께 설계한다.
- 테스트: `tests/utils/test_archive_analysis.py`에 분기 있는 가짜 아카이브 fixture 추가.

### 3-2. 부모 선택 전략을 교체 가능하게 (구현 완료)

`runner.py`의 `select_base_agent`를 `search/selection.py`로 옮기고 전략을 인터페이스로 분리한다.

```python
class ParentSelector(Protocol):
    def select(self, archive: ArchiveView, k: int, rng: Random) -> list[int]: ...
```

| 전략 | 동작 | 용도 |
|------|------|------|
| `greedy_ci` | 현재 로직 그대로 | 기준선, 기존 결과 재현 |
| `score_child_prop` | 가중치 = sigmoid(점수) × 1/(1+자식 수). 점수가 높고 덜 탐색된 노드 우선 | 기본 권장 |
| `thompson` | 문제별 점수로 Beta 사후분포를 만들고 샘플링해 최댓값 선택 | 평가 표본이 적을 때 |

- `status != "ok"`인 노드는 후보에서 제외.
- 선택 시 사용한 seed와 각 후보의 가중치를 `lineage.json`에 남겨 재현 가능하게 한다.
- 구현 메모: `select`는 부모 목록과 후보별 선택 확률을 함께 담은 `Selection`을 반환한다. `--selection`, `--seed` 옵션은 3-3이 아니라 여기서 추가했고 기본값은 `greedy_ci`다. `score_child_prop`의 sigmoid는 `1/(1+exp(-10·(점수-0.5)))`이다.
- 테스트: 고정 seed에서 선택 결과가 결정적인지, 실패 노드가 제외되는지, `greedy_ci`가 현재 함수와 같은 결과를 내는지.

### 3-3. 세대당 자식 여러 개 (구현 완료)

구현 메모:
- 한 세대 = "미평가 에이전트 평가 → 자식 K개 계획 → K개 병렬 생성". 평가는 에이전트별로 순차 실행한다 (문제 단위 병렬은 `--workers` 그대로).
- 상태는 `metadata.json`의 `generation`, `pending_agents`, `planned_children`. 계획을 세대당 한 번만 세워 저장하므로 재개해도 같은 부모가 쓰인다. 예전 `agent_iteration`만 있는 실험은 자동 변환된다.
- 중단된 생성물은 삭제하지 않고 `_interrupted/`로 옮긴다.
- 이미지: 기존 재빌드는 build context가 `./base_agent`(seed)라서 새 에이전트의 `requirements.txt`를 반영하지 못했다. 이제 에이전트별 태그를 그 에이전트의 코드로 빌드하고, 빌드 실패는 `failed`로 기록한다.
- change log: 부모의 로그를 그대로 물려받고, 다른 가지의 시도는 runner가 생성하는 표시 구간("Attempts on other branches")에 매번 새로 쓴다.
- 메타 개선 로그는 `agent_{child}/meta_improvement_logs/`에 둔다 (이전: 직전 에이전트 디렉터리).
- `--best-iter`에는 부모 id, `--current-iter`에는 자식 id를 넘긴다.

- CLI: `--children K` (기본 1, 기존 동작 유지), `--selection {greedy_ci,score_child_prop,thompson}`.
- `main()` 루프를 "세대" 단위로 변경: 부모 K개 샘플링 → 메타 개선 K개 병렬 → 자식 K개 평가.
- `metadata.json`의 `agent_iteration`(다음 번호 하나)만으로는 재개가 안 되므로 `pending_agents: [id...]` 목록을 추가해 미평가 자식부터 재개한다.
- 메타 개선 뒤의 이미지 재빌드(`runner.py:929-966`)는 전역 `sica_sandbox` 태그를 덮어쓴다. 자식이 여러 개면 서로의 `requirements.txt`를 덮어쓰므로 **에이전트별 태그** `sica_sandbox:run{id}_agent{i}`로 빌드하고 해당 에이전트 평가에 그 태그를 쓴다.

### 3-4. 분산을 줄인 평가

- **대응 비교** (F3, 구현 완료 — 모듈 함수 `paired_delta(scores_df, i, j)`와 `ArchiveAnalyzer.paired_delta(i, j)`, 개요 표의 `vs Parent` 열): 같은 seed의 같은 문제를 쓰므로, 자식 vs 부모를 문제별 점수 차이의 paired bootstrap으로 비교한다. `compute_statistics`에 `paired_delta(i, j) -> (mean, ci_low, ci_high)` 추가. 독립 z-구간보다 구간이 좁아진다.
- **반복 실행** (F4, 구현 완료 — 첫 실행은 기존처럼 `problem_id`, 이후는 `problem_id#r`과 `repeat` 필드): `--repeats R`로 문제당 R회 실행. `results.jsonl` 키를 `problem_id` → `problem_id#r`로 확장하고 집계 시 문제 단위로 먼저 평균. `BenchmarkTracker`와 `_parse_results_file`이 영향받는다.
- **단계적 평가** (구현 완료 — 승격 규칙은 아래 메모 참고): 1단계는 작은 부분집합(예: 벤치마크당 8문제)으로 모든 자식을 평가하고, 부모 대비 paired 하한이 임계값 이상인 자식만 2단계 전체 부분집합으로 승격. 비용을 탐색 폭으로 돌린다.
- **검증용 분리 세트** (구현 완료 — 저장 위치는 `results/holdout/run_{id}/`): 메타 에이전트가 아카이브에서 볼 수 있는 "dev" 문제와, 볼 수 없는 "holdout" 문제를 seed로 분리. holdout 결과는 `/home/agent/archive` 마운트 밖(`results/run_{id}/_holdout/`)에 저장한다. dev 점수만 오르고 holdout이 안 오르면 과적합이다.
- F9 정리: utility의 `time_limit`/`cost_limit`을 runner 설정값에서 받도록 한다. **1-3으로 연기**: runner의 한도가 아직 코드 상수라서 설정 파일화(1-3)와 함께 해야 한다. 또한 벤치마크 단위 utility(`BenchmarkScore.utility`)가 벤치마크 **전체 합계** 비용·시간을 문제당 한도와 비교하고 있어 문제가 여러 개면 거의 항상 0이 된다. 정의 자체를 문제별 utility의 평균으로 바꿀지 함께 정한다.

구현 메모:
- 승격 규칙: 계획의 "paired 하한 ≥ 임계값" 대신 "paired **상한** ≥ 임계값"(기본 0)으로 구현했다. 1단계는 문제 수가 적어 구간이 넓으므로, 하한 기준이면 실제로 개선된 자식도 대부분 탈락한다. 상한 기준은 "부모만큼 좋을 가능성이 데이터로 배제된" 자식만 떨어뜨린다. `--promote-threshold`를 올리면 더 공격적으로 걸러낸다.
- 탈락한 에이전트는 `lineage.json`의 `evaluation.promoted: false`로 기록되고, 부모·메타 에이전트 후보에서 제외된다 (일부 문제의 점수만 있어 전체 점수를 다른 에이전트와 비교할 수 없다).
- holdout 위치: 계획의 `results/run_{id}/_holdout/`은 실험 디렉터리 안이라 메타 에이전트 컨테이너에 그대로 마운트된다. 실험 디렉터리 밖인 `results/holdout/run_{id}/`로 바꿨다.
- 분할은 seed와 문제 id의 해시로 정하고 `metadata.json`의 `evaluation.split_seed`에 고정한다. 평가 결과가 생긴 뒤에는 holdout 비율을 바꿀 수 없다.

### 3-5. 메타 에이전트 분리 (구현 완료)

구현 메모: `--meta-model`은 `AGENT_MODEL`만 덮어쓴다 (`AGENT_REASONING_MODEL`, `AGENT_OVERSIGHT_MODEL`은 그대로). 모델 이름은 seed의 `Model` enum으로 검증한다. 메타 에이전트와 모델은 세대 계획과 각 에이전트의 `lineage.json`에 기록된다.

- 지금은 "최고 점수 에이전트가 메타 에이전트" 고정(`runner.py:866`). `--meta-agent {best,parent,seed,<id>}` 옵션으로 분리.
- `--meta-model`로 메타 개선에만 다른 모델을 쓰게 한다 (`AGENT_MODEL` 환경변수를 `docker exec -e`로 전달). README의 "stronger LLM builds scaffold for weaker LLM" 실험이 가능해진다.

### 3-6. 실험 리포트 (구현 완료)

구현 메모: 곡선 그림은 그리지 않는다 (matplotlib가 의존성에 없다). 대신 여러 run의 세대별 최고 점수를 표와 `--csv`로 내보내고 최종 점수의 표준편차를 출력한다. 계보 트리는 `--dot`으로 Graphviz 형식을 쓴다. 비용 집계를 위해 메타 개선 단계의 사용량을 `lineage.json`의 `usage`에 기록하도록 runner를 고쳤다 (이전 에이전트는 메타 비용이 0으로 나온다).

- `python -m search.report results/run_{id}`: 계보 트리(텍스트 + Graphviz), 세대별 최고 점수, 누적 비용, 전략별 선택 로그.
- 여러 run의 곡선을 겹쳐 그려 run 간 분산을 직접 측정한다. 이 수치가 Phase 3의 성공 지표다.

### Phase 3 완료 기준

- `--children 1 --selection greedy_ci`가 현재 동작과 같은 부모를 고른다 (회귀 테스트).
- `--children 3 --selection score_child_prop`로 2세대 이상이 포트 충돌 없이 끝나고, 중간에 죽여도 재개된다.
- 동일 설정 3회 run의 최종 점수 표준편차가 기준선(`greedy_ci`, repeats 1)보다 작다.

---

## Phase 1 — 기반 안정화

### 1-1. 검증 게이트 (F5) (구현 완료)

구현 메모:
- 게이트는 새 에이전트 자신의 이미지로 띄운 별도 컨테이너에서 돈다. 코드는 읽기 전용으로 마운트한 뒤 컨테이너 안으로 복사해, 테스트가 남기는 캐시가 에이전트 코드에 섞이지 않게 한다.
- 테스트 검사는 "부모도 실패하던 테스트"는 문제 삼지 않는다. seed의 테스트가 샌드박스에서 원래 실패하면 모든 자식이 탈락하기 때문이다. 각 에이전트의 테스트 결과는 `lineage.json`의 `tests_passed`에 남는다.
- 실험 시작 시 초기 에이전트에 게이트를 한 번 돌린다. import나 스모크 실행이 실패하면 환경 문제이므로 아무 비용도 쓰기 전에 멈춘다.
- 스모크 실행은 기본적으로 에이전트 자신의 모델을 쓴다 (계획의 "저가 모델"은 `--gate-model`로 지정). 다른 제공사의 저가 모델을 기본값으로 하면 그 API 키가 없는 환경에서 게이트가 전부 실패한다.
- `--gate-retries`의 재시도는 메타 에이전트를 일반 모드(`python -m ….agent --prompt`)로 다시 불러 실패 로그를 주고 고치게 한다. 이 호출의 비용은 사용량 줄을 출력하지 않아 `usage`에 잡히지 않는다.
- 이미지 빌드 실패는 재시도 없이 `failed`로 남는다.

메타 개선 직후, 평가 전에 컨테이너 안에서 순서대로 실행한다.

1. `python -c "import agent_code.agent"` — import 가능 여부
2. `pytest agent_code -x -q` — 단위 테스트
3. 저가 모델로 스모크 실행: `python -m agent_code.agent benchmark --prompt-file <2+2 문제> --timeout 120`

하나라도 실패하면 `status: "invalid"`로 기록하고 평가를 건너뛴다. `--gate-retries N`이면 실패 로그를 붙여 메타 개선을 다시 시도한다.

### 1-2. 시크릿 런타임 주입 (F7)

- Dockerfile의 `ARG`/`ENV` 키 블록 삭제. `docker run`에 `--env-file`(runner가 0600 권한 임시 파일로 생성) 전달.
- `runner.py:936-962`와 `Makefile`의 `--build-arg` 키 전달 삭제.
- `sandbox/GOOGLE_APPLICATION_CREDENTIALS.json`은 이미지에 COPY하지 않고 읽기 전용 마운트.
- 기존 이미지에는 키가 레이어에 남아 있으므로 재빌드 후 이전 이미지를 삭제하도록 README에 안내.

### 1-3. 설정 파일화 (F8)

- `runner_config.py`(pydantic-settings, `base_agent/src/config.py`와 같은 방식): `timeout_s`, `cost_limit_usd`, `subset_size`, `seed`, `repeats`, `benchmarks: list[str]`, `workers`.
- `--config run.yaml` + CLI 오버라이드. 사용한 설정 전체를 `metadata.json`에 저장.
- `benchmark_registry`는 전부 등록하고, 실행할 것만 설정의 이름 목록으로 고른다 (주석 해제 방식 제거).

### 1-4. 결과 프로토콜과 상태 저장 (F10)

- 에이전트가 `logdir/usage.json`(`{tokens, cached, cost, wall_time}`)을 쓰고 runner가 읽는다. 이전 세대 에이전트와의 호환을 위해 stdout 파싱은 fallback으로 남긴다.
- `update_metadata`를 임시 파일 + `os.replace`로 원자적 쓰기.
- `process_job` 실패 경로(`return False`)에서 실패 사유를 `results.jsonl`에 기록. 지금은 로그에만 남아 재개 시 원인을 알 수 없다.

### 1-5. 모델·SDK 갱신 (F12)

- `Model` enum에 현재 모델 추가, `config.py` 기본값 교체, `FALLBACK` 표 정리.
- `anthropic` 버전 고정 해제 후 `providers/anthropic.py`(897줄) 회귀 확인. 토큰 단가는 추가 전 공식 가격표에서 확인한다.
- 구형 모델 항목은 과거 run 재분석을 위해 남긴다.

### 1-6. CI와 정리 (F13)

- GitHub Actions: `ruff check` + `pytest base_agent` (API 호출 없는 테스트만).
- `runner.py`의 주석 처리된 옛 코드 블록 삭제 (`786-796`, `810-817`, `841-846`, `915-927`).
- `runner.py`(1051줄)를 `runner/` 패키지로 분리: `docker.py`, `jobs.py`, `meta.py`, `state.py`, `cli.py`. Phase 3에서 만든 `search/`와 나란히 둔다.
- `.gitignore`에 `.DS_Store`, `.omc/`, `.bkit/` 추가.

### Phase 1 완료 기준

- `docker history sica_sandbox`에 API 키가 보이지 않는다.
- 일부러 문법 오류를 넣은 자식이 `invalid`로 기록되고 루프가 계속된다.
- `python runner.py --config run.yaml` 하나로 벤치마크·한도·반복 수가 정해진다.

---

## Phase 2 — 에이전트 능력

`base_agent/`(seed 에이전트)에 직접 넣는 기능이다. **주의**: 이 기능들은 원 논문에서 에이전트가 스스로 발견하는 대상이다. seed에 넣으면 출발점이 달라져 기존 결과와 직접 비교할 수 없으므로, Phase 3의 리포트로 "기존 seed vs 강화 seed"를 따로 비교한다.

### 2-1. 부분 편집 도구 (F11)

- `tools/edit_tools/search_replace.py`: `old_string`/`new_string` 정확 일치, 일치가 0개거나 2개 이상이면 실패와 함께 주변 문맥 반환.
- `tools/edit_tools/apply_patch.py`: unified diff 적용 (`diff-match-patch`가 이미 requirements에 있음).
- 편집 후 `.py`는 `ast.parse`로 문법 확인, 실패 시 되돌리고 오류 반환.
- `toolkits["coding"]`에 등록, `CodingAgent.AVAILABLE_TOOLS`에 추가.
- 측정: `file_editing` 벤치마크 활성화 후 토큰·비용·점수 비교.

### 2-2. 코드 탐색 도구

- `tools/symbol_tools.py`: tree-sitter 기반 `find_definition`, `find_references`, `outline_file`.
- 측정: `symbol_location` 벤치마크.

### 2-3. 테스트 검증 루프

- `reasoning_structures/test_driven.py`: 재현 테스트 작성 → 실패 확인 → 수정 → 통과 확인 → 전체 테스트 실행의 단계 구조. 기존 `ToolBasedReasoningStructure`(`sequential.py`)를 상속.
- `ExecuteCommand` 결과에서 pytest 실패 요약만 추려 컨텍스트 사용량을 줄이는 후처리.
- 측정: `swebench_verified`, `livecodebench`.

### 2-4. 진단 도구

- `tools/diagnostics.py`: `ruff` + `pyright`를 파일 단위로 실행해 구조화된 진단 반환. 2-1 편집 도구가 편집 후 자동 호출할 수 있게 옵션 제공.

### 2-5. 컨텍스트 관리

- `OpenFile`로 연 파일이 계속 컨텍스트에 남는 구조(`utils/file_views.py`)에 토큰 예산 추가: 예산 초과 시 가장 오래 안 쓴 파일부터 요약본으로 교체.
- 하위 에이전트 반환값 길이 상한과 요약.

### Phase 2 완료 기준

- 각 도구에 `generate_examples()`와 단위 테스트가 있다 (기존 `tests/tools/` 형식).
- 강화 seed가 `file_editing`, `symbol_location`에서 기존 seed 대비 paired 하한 > 0.

---

## 결정이 필요한 항목

| 항목 | 기본 선택 | 대안 |
|------|-----------|------|
| 기본 부모 선택 전략 | `score_child_prop` | `thompson` |
| 세대당 자식 수 기본값 | 1 (기존 동작) | 3 |
| Phase 2 기능을 seed에 넣을지 | 넣되 별도 seed 디렉터리(`base_agent_plus/`)로 분리 | `base_agent/`에 직접 |
| holdout 비율 | 문제의 30% | 벤치마크 단위 분리 |

## 작업 순서 요약

1. 3-0 포트·실패 표시 → 3-1 계보 → 3-2 선택 전략 → 3-4 대응 비교 → 3-3 다중 자식 → 3-4 나머지 → 3-5 → 3-6
2. 1-1 게이트 → 1-2 시크릿 → 1-4 프로토콜 → 1-3 설정 → 1-5 모델 → 1-6 CI·분리
3. 2-1 편집 → 2-4 진단 → 2-2 탐색 → 2-3 테스트 루프 → 2-5 컨텍스트
