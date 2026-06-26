# Codebuilder — Visão Geral de Arquitetura e Implementação

> Documento de referência para a Vivo. Descreve, em português, **o que o Codebuilder faz**, **como ele foi construído sobre o CrewAI** e **passo a passo de implementação** que pode ser usado como template para novos agentes/flows.

---

## 1. O que é o Codebuilder

O Codebuilder é um **Flow do CrewAI** que recebe um *brief* de projeto (descrição em linguagem natural + objetivos + stack + anexos) e devolve **código pronto** — seja como um projeto novo zipado, seja como um *patch* aplicado sobre um repositório existente acompanhado do projeto completo corrigido em `.zip`.

O ciclo de vida de um job é:

```
brief  →  ingest  →  PLAN (crew)  →  HITL approval  →  BUILD loop (writer↔reviewer)  →  FINALIZE (QA + arquitetura)  →  artefatos
                                         ↑                                                       │
                                         └────── amend (HITL volta com mudanças) ────────────────┘
```

Características principais:

- **Plan-then-build** com aprovação humana obrigatória antes de qualquer código ser gerado.
- **Workspace isolado por job** (filesystem sandbox) — agentes nunca enxergam disco fora do diretório do job.
- **QA determinístico** (`ruff` + `pytest`) como gate, antes de qualquer revisão por LLM.
- **Gate de arquitetura por domínio** — projetos novos passam por uma checagem estrutural específica do domínio (ex.: RPA exige Clean Architecture + producer/consumer/orchestrator).
- **HITL assíncrono via webhook** — o flow pausa, dispara webhook para a UI, e é retomado por uma chamada HTTP separada.

---

## 2. Por que **Flow** e não Crew

O CrewAI oferece duas abstrações:

- **Crew** = um time de agentes resolvendo uma tarefa multi-step. Bom para uma unidade de trabalho.
- **Flow** = uma máquina de estados com múltiplas crews, lógica condicional, pausas para humano (HITL) e ramificações.

Codebuilder é Flow porque precisa:

1. Orquestrar **três crews diferentes** (Planner, Writer, Reviewer).
2. **Pausar para aprovação humana** entre planejar e executar.
3. Ramificar conforme o resultado (`approved` / `amend` / `rejected`).
4. **Iterar** internamente no loop writer↔reviewer.

No `pyproject.toml`:

```toml
[tool.crewai]
type = "flow"
```

---

## 3. Estrutura de pastas

```
codebuilder/
├── pyproject.toml              # deps + entrypoints (kickoff/plot/run_crew)
├── CLAUDE.md / AGENTS.md       # docs internas
├── src/codebuilder/
│   ├── main.py                 # CodebuilderFlow + entrypoints
│   ├── schemas.py              # Pydantic: Plan, SubTask, CodeArtifact, ReviewResult, QAReport, CodebuilderState
│   ├── runtime_qa.py           # QA determinístico + gates de arquitetura
│   ├── history.py              # SQLite project_history (contexto entre runs)
│   ├── feedback_provider.py    # WebhookFeedbackProvider (HITL)
│   ├── crews/
│   │   ├── planner_crew/       # @CrewBase + agents.yaml + tasks.yaml
│   │   ├── writer_crew/
│   │   └── reviewer_crew/
│   ├── tools/
│   │   ├── workspace_tool.py   # Read/Write/List sandboxed no workspace
│   │   ├── lint_runner_tool.py # ruff
│   │   ├── (test_runner)       # pytest
│   │   ├── git_tool.py         # clone/init/diff
│   │   ├── attachment_tool.py  # materializa anexos
│   │   └── s3_artifacts.py     # upload do build final
│   └── skills/
│       ├── rpa/                # guideline canônico RPA (PT-BR)
│       └── code-review-gate/   # critérios de aceitação de projetos gerados
└── workspaces/<session_id>/    # sandbox de cada job
    ├── inputs/                 # anexos materializados (repo clonado, PDFs, zips…)
    └── output/                 # projeto novo gerado (modo new_project)
```

---

## 4. Conceitos do CrewAI usados

| Conceito | Onde aparece | Para que serve |
|---|---|---|
| `Flow[State]` | `CodebuilderFlow` | Máquina de estados com `@start`, `@listen` |
| `@start()` | `ingest` | Ponto de entrada do flow |
| `@listen(...)` | `plan`, `build`, `finalize` | Encadeia métodos por evento ou outro método |
| `@human_feedback(...)` | `plan`, `revise_plan` | Pausa o flow e roteia `approved/amend/rejected` via LLM classifier |
| `FlowState` (pydantic) | `CodebuilderState` | Estado tipado, persistido |
| `SQLiteFlowPersistence` | implícito | Persistência automática que viabiliza `from_pending(...).resume(...)` |
| `@CrewBase` | `PlannerCrew`, `WriterCrew`, `ReviewerCrew` | Carrega `agents.yaml` + `tasks.yaml` |
| `@agent`, `@task`, `@crew` | dentro de cada crew | Decoradores que registram os componentes |
| `Agent(config=cfg, llm=LLM(...), tools=[...], skills=[...])` | `planner`, `writer`, `reviewer`, `qa_agent` | Construção do agente |
| `Task(config=..., output_pydantic=Schema, guardrail=...)` | `plan_task`, `write_task`, etc. | Saída estruturada validada |
| `Process.sequential` | todas as crews | Tarefas em sequência |
| `discover_skills(...)` + `activate_skill(...)` | topo de cada crew | Carrega skills (markdown guidelines) e ativa eventos |

---

## 5. Estado do flow (`CodebuilderState`)

Tudo que persiste entre passos vive no `CodebuilderState` (pydantic, herda de `FlowState`):

```python
session_id: str               # identidade do USUÁRIO/UI (estável, vem do caller)
# state.id (= flow_id)        # identidade da EXECUÇÃO (UUID auto-gerado pelo Flow)
brief, project_name, goals, tech_stack, attachments
attachment_records: list[dict] # resumo compacto dos anexos materializados
project_key: str              # chave estável p/ histórico (canônica do git URL OU slug do project_name)
workspace_dir: str
plan: Plan | None
amendments: str
amend_cycles: int
artifacts: list[CodeArtifact]
review_results: list[ReviewResult]
qa_report: QAReport | None
preflight_qa_report: QAReport | None  # QA completo antes do plano em patch_existing
final_qa_repair_attempts: int
patch: str                    # modo patch_existing
zip_path, zip_url              # só aparecem no payload público quando QA passa
project_archive               # zip completo do projeto; QA define se é verificado ou "salvage"
status: "pending" | "planning" | "awaiting_approval" | "executing" | "done" | "failed"
```

**Regra de ouro**: `session_id` ≠ `state.id`. Nunca passe `id` no `kickoff(inputs=...)` — sobrescrever `state.id` desalinha os traces OTel do AMP (eles ficam indexados no `flow_id` original no Wharf, mas o AMP busca pelo `state.id` sobrescrito). Use `session_id` para a UI e deixe o `state.id` ser o UUID do flow.

---

## 6. O Flow passo a passo (`src/codebuilder/main.py`)

### 6.1 `ingest` — `@start()`

```python
@start()
def ingest(self):
    # CrewAI auto-merge: as keys de inputs={...} entram em self.state ANTES desta função rodar.
    # session_id, brief, project_name, goals, tech_stack, attachments já estão populados.
    self.state.attachments = [Attachment(**a) if isinstance(a, dict) else a
                              for a in self.state.attachments]

    session_key = self.state.session_id or self.state.id
    workspace_dir = WORKSPACE_ROOT / session_key
    (workspace_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "output").mkdir(parents=True, exist_ok=True)
    self.state.workspace_dir = str(workspace_dir)

    if self.state.attachments:
        self.state.attachment_records = attachment_tool.materialize(
            [a.model_dump() for a in self.state.attachments],
            self.state.workspace_dir,
        )

    self.state.project_key = history.project_key_from(self.state) or session_key
    self.state.status = "planning"
```

**O que esta etapa faz:**
- Cria o sandbox `workspaces/<session_id>/`.
- Materializa anexos: clones git vão para `inputs/repo`, zips são extraídos, PDFs viram texto, imagens são salvas como blob. O retorno fica em `attachment_records`, um resumo compacto usado pelo planner.
- Deriva uma **chave estável de projeto** (`project_key`) usando o URL git canonicalizado ou um slug do `project_name` — isso permite que múltiplos runs do mesmo projeto compartilhem histórico.

### 6.2 `plan` — `@listen(ingest)` + `@human_feedback(...)`

```python
@listen(ingest)
@human_feedback(
    message="Review the generated plan. Reply 'approve' to start coding, "
            "describe changes to amend, or 'reject' to cancel.",
    emit=["approved", "amend", "rejected"],
    llm=GUARDRAIL_LLM,
    default_outcome="amend",
)
def plan(self) -> dict:
    result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
    plan_obj = validate_plan(result.pydantic)
    self.state.plan = plan_obj
    self.state.status = "awaiting_approval"
    return plan_obj.model_dump()
```

**O que acontece:**
1. `PlannerCrew` recebe um dict com `brief`, `goals`, `tech_stack`, listagem de anexos, plan anterior (se houver amend), histórico, amendments do usuário.
2. O planner devolve um `Plan` (Pydantic). `validate_plan()` impõe 1–15 subtasks com `file_path` e `test_criteria` não-vazios.
3. `@human_feedback` **pausa o flow** e dispara o `WebhookFeedbackProvider`, que faz POST para `$CODEBUILDER_APPROVAL_WEBHOOK` com o plano serializado e levanta `HumanFeedbackPending`.
4. CrewAI auto-instancia um `SQLiteFlowPersistence` e grava a linha de pending. **O processo do worker pode morrer aqui** — quando a UI chamar `CodebuilderFlow.from_pending(job_id).resume(feedback)`, o flow é reconstruído do SQLite.
5. Quando o feedback chega, o LLM classificador (`GUARDRAIL_LLM`) traduz o texto livre do humano para um dos labels: `approved | amend | rejected`. O flow continua pela ramificação correspondente.

### 6.3 `revise_plan` — `@listen("amend")`

```python
@listen("amend")
@human_feedback(...)
def revise_plan(self, prior) -> dict:
    self.state.amendments = getattr(prior, "feedback", "") or ""
    self.state.amend_cycles += 1
    result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
    plan_obj = validate_plan(result.pydantic)
    self.state.plan = plan_obj
    self.state.status = "awaiting_approval"
    return plan_obj.model_dump()
```

Roda o planner de novo com `prior_plan` + `amendments` no contexto e regate em outro `@human_feedback`. Loop até `approved` ou `rejected`.

### 6.4 `on_rejected` — `@listen("rejected")`

Marca `status = "failed"`, grava histórico, retorna payload de cancelamento.

### 6.5 `build` — `@listen("approved")`

```python
@listen("approved")
def build(self, prior):
    self.state.status = "executing"
    plan = self.state.plan

    if plan.mode == "patch_existing":
        build_dir = <inputs/repo do clone>        # escreve sobre o repo clonado
    else:
        build_dir = <workspace/output>            # projeto novo
        git_tool.init_and_commit(build_dir)       # baseline para diff posterior

    for index, subtask in enumerate(plan.subtasks, start=1):
        self._build_subtask(subtask, build_dir, index=index, total=len(plan.subtasks))
```

E `_build_subtask` é o coração do gerador:

```python
for attempt in range(_max_subtask_retries() + 1):
    artifact = writer.crew().kickoff(inputs={subtask, workspace_listing,
                                              amendments, prior_review_issues}).pydantic
    persist_artifact(artifact, build_dir)          # escreve no disco

    deterministic = run_deterministic_review(subtask, artifact, build_dir)
    # determinístico = path match + content match + ruff + pytest (se for arquivo de teste)
    review = deterministic.result
    if deterministic.needs_fallback:
        review = reviewer.crew().kickoff(...).pydantic   # fallback LLM apenas em casos ambíguos

    if review.passed: break
    prior_issues = "\n".join(review.issues)
    # short-circuit: se as mesmas issues voltam, para de queimar retries
```

**Pontos-chave:**
- O writer escreve **um bundle por subtask** (1–6 arquivos planejados), com paths validados via `resolve_within()` (impossível escapar do sandbox).
- A revisão **determinística** (lint+test) é a primeira linha de defesa. O reviewer LLM só roda quando o determinístico não consegue decidir — economia de tokens.
- Issues idênticas em retries consecutivos → **circuit breaker** que entrega o erro pro QA final em vez de queimar mais retries.
- A cada subtask, um webhook de progresso (`CODEBUILDER_PROGRESS_WEBHOOK`) é disparado com `subtask_started / completed / failed` para a UI mostrar progresso fino.

### 6.6 `finalize` — `@listen(build)`

```python
@listen(build)
def finalize(self, _prior=None):
    build_dir = getattr(self, "_build_dir", self.state.workspace_dir)

    # 1. QA determinístico
    self.state.qa_report = run_final_qa(build_dir)        # ruff + pytest

    # 2. Repair pass — 1 tentativa do writer pra corrigir QA quebrado
    self._repair_final_qa_if_needed(build_dir)

    # 3. Gate de arquitetura por domínio (só new_project)
    if self.state.plan.mode == "new_project":
        review = run_full_architecture_gate(build_dir, self.state.plan)
        # ↑ despacha por plan.domain (ex.: "rpa" → _rpa_full_gate)

    # 4. Patch (modo patch_existing) + zip completo (somente se QA passou)
    if mode == "patch_existing": self.state.patch = git_tool.diff(build_dir)
    self.state.zip_path = _zip_build(build_dir, ...)
    self.state.project_archive = ProjectArchiveRef(...)

    # 5. Upload S3 dos artefatos
    # O zip completo é o artefato primário; arquivos individuais são auditoria.
    self.state.qa_report.artifact_urls = [project_archive_ref, *file_refs]

    # 6. Histórico (best-effort, nunca quebra o flow)
    history.record(self.state)

    self.state.status = "done" if qa_report.passed else "failed"
```

Em `patch_existing`, lint/type rodam nos arquivos alterados e pytest roda a suíte inteira quando existem testes no projeto. Se o projeto realmente não tem arquivos de teste, `pytest` sem coleta vira aviso não-bloqueante; em `new_project`, ausência de testes continua sendo falha de QA. Mesmo quando lint/type falham, pytest roda para alimentar o repair com as falhas reais.

### 6.7 Gate de arquitetura por domínio

Em `runtime_qa.py`:

```python
_ARCHITECTURE_GATES: dict[str, Any] = {
    "rpa": _rpa_full_gate,
    # adicionar aqui novos domínios: "flask-api", "python-package", ...
}

def run_full_architecture_gate(build_dir, plan):
    domain = plan.domain if plan else ""
    gate = _ARCHITECTURE_GATES.get(domain)
    if gate is None:
        return ReviewResult(subtask_id="architecture_gate", passed=True,
                            suggestions=["No domain gate registered."])
    return gate(build_dir, plan)
```

O **slug `plan.domain`** é emitido pelo planner e bate com o nome de uma **skill** ativa (ex.: `rpa`). Cada gate combina:
1. Um **check determinístico** de estrutura (ex.: `pyproject.toml` declara `ruff/pytest/mypy`, existe `src/<package>/{domain,application,infrastructure}/`, existem componentes `producer/consumer/orchestrator`, há `tests/`).
2. Um **review LLM** com a skill do domínio ativada (`ReviewerCrew.architecture_gate_crew()`).

Projetos sem `domain` ou com domínio não-registrado finalizam só pelo lint+test.

---

## 7. As três crews

Todas seguem o mesmo padrão do CrewAI:

```python
@CrewBase
class XxxCrew:
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def some_agent(self) -> Agent:
        cfg = self.agents_config["some_agent"]
        return Agent(config=cfg,
                     tools=[...],
                     skills=[_SKILLS["rpa"]],
                     llm=LLM(model=cfg["llm"], max_tokens=32768))

    @task
    def some_task(self) -> Task:
        return Task(config=self.tasks_config["some_task"],
                    output_pydantic=SomeSchema)

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks,
                    process=Process.sequential, verbose=True)
```

| Crew | Agentes | Tarefas | Ferramentas | Schema de saída |
|---|---|---|---|---|
| **PlannerCrew** | `planner` | `plan_task` | `FileReadTool`, `DirectoryReadTool` (**não escreve**) | `Plan` (1–15 SubTasks) |
| **WriterCrew(workspace_dir)** | `writer` | `write_task`, `repair_task` | `WorkspaceRead/Write/ListTool` (sandboxed) | `CodeArtifact` |
| **ReviewerCrew(workspace_dir)** | `reviewer`, `qa_agent` | `review_task`, `qa_task`, `architecture_gate_task` | `WorkspaceRead/ListTool`, `LintRunnerTool`, `TestRunnerTool` | `ReviewResult` / `QAReport` |

**`max_tokens` explícito** em todos os agentes — o default da `AnthropicCompletion` é 4096, que trunca `Plan.subtasks` e `CodeArtifact.content` para projetos não-triviais. Use 16k–32k.

### Saída estruturada (Pydantic)

Toda task crítica usa `output_pydantic=SomeSchema`. `Plan` e `CodeArtifact` herdam de `StrictOutputModel` (`extra="forbid"`) — o agente NÃO pode inventar campos. `validate_plan()` adiciona um guardrail de domínio (1–15 subtasks, file_path/test_criteria não vazios) que **levanta** se violado, e a YAML do task descreve as `Rules:` e o `expected_output:` casando com o schema.

---

## 8. Skills do CrewAI

Skills são **documentos markdown com frontmatter** que descrevem metodologia/checklist. O `Agent` carrega skills via `skills=[...]`. Quando ativadas, o engine do CrewAI emite eventos próprios e injeta o conteúdo no contexto.

```
src/codebuilder/skills/
├── rpa/                 # guideline canônico RPA (Clean Arch + producer/consumer/orchestrator + TDD)
└── code-review-gate/    # critérios de aceitação genéricos de projeto
```

Carregamento (no topo de cada crew):

```python
_SKILLS = {s.name: activate_skill(s)
           for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")}

# uso:
Agent(..., skills=[_SKILLS["rpa"], _SKILLS["code-review-gate"]])
```

**Para adicionar um novo domínio** (ex.: `flask-api`):
1. Criar `src/codebuilder/skills/flask-api/SKILL.md` com o guideline.
2. Anexar `_SKILLS["flask-api"]` aos agentes que precisam.
3. (Opcional) Registrar um gate de arquitetura: adicionar `"flask-api": _flask_full_gate` em `_ARCHITECTURE_GATES`.
4. O planner pode emitir `plan.domain = "flask-api"` e o gate é despachado automaticamente.

---

## 9. Tools — o sandbox

`workspace_tool.py` expõe três tools com `resolve_within()` que rejeita qualquer path que escape do `workspace_dir`:

```python
WorkspaceReadTool(workspace_dir=...)
WorkspaceWriteTool(workspace_dir=...)
WorkspaceListTool(workspace_dir=...)
```

**Regra absoluta**: agentes **nunca** recebem `FileReadTool` apontado pro disco real. Toda I/O passa por essas tools.

Demais tools:
- `LintRunnerTool` → `ruff check` (timeout 120s)
- `TestRunnerTool` → `pytest -q` (timeout 300s)
- `git_tool` → `clone`, `init_and_commit`, `diff` (gitpython)
- `attachment_tool.materialize()` → decodifica `Attachment[]` em arquivos no `inputs/`
- `s3_artifacts` → upload do workspace pro S3 com prefixo `<project_key>/<flow_id>/...`

---

## 10. HITL (Human-In-The-Loop) ponta a ponta

```
[Flow]                              [Webhook UI]                [Operador]
  │                                       │                         │
  │── @human_feedback dispara ──────────► │                         │
  │   POST $APPROVAL_WEBHOOK              │                         │
  │   payload: plan + flow_id + session   │                         │
  │                                       │── exibe plano ────────► │
  │   raise HumanFeedbackPending          │                         │
  │   ↓ SQLitePersistence salva pending   │                         │
  │   worker pode morrer aqui             │                         │
  │                                       │ ◄─── feedback texto ────│
  │                                       │                         │
  │                                       │── POST /resume(job_id, feedback)
  │                                       ▼
  │   CodebuilderFlow.from_pending(job_id).resume(feedback)
  │   LLM classifier → "approved"/"amend"/"rejected"
  │── continua pela ramificação correspondente
```

**Cuidados não-óbvios:**
- **NÃO mutar `CREWAI_STORAGE_DIR` em runtime.** O CrewAI usa esse env var como `app_name` no `appdirs.user_data_dir(...)`, que define onde o `SQLiteFlowPersistence()` grava. `from_pending(flow_id)` SEMPRE constrói um persistence default — qualquer drift entre save-time e resume-time quebra com "No pending feedback found for flow_id".
- **NÃO passar `id` em `kickoff(inputs=...)`.** Sobrescrever `state.id` antes do primeiro span OTel desalinha os traces no AMP/Wharf. Use `session_id`.

---

## 11. Histórico por projeto (`history.py`)

Tabela SQLite **separada** da flow persistence:

```
data/codebuilder_history.db
└── project_history
    └── (project_key, job_id) → {date, mode, status, plan_json, qa_json,
                                  files_touched[], top_issues[], git_diff}
```

API pública:
- `canonicalize_git_url(url)` — reduz git URLs a `host/org/repo` (HTTPS/SSH/scp-style/.git).
- `project_key_from(state)` — URL canônica do 1º git attachment, OU slug de `project_name`, OU `""`.
- `record(state)` — chamado em `finalize`, `on_rejected` e build-failure. Best-effort: `try/except` envolve todas as chamadas.
- `summarize_for_planner(project_key, limit=3)` — markdown compacto dos últimos N runs, injetado em `{prior_history}` no tasks.yaml do planner.

**É observabilidade, não correção.** Falhas no DB nunca quebram o flow.

---

## 12. Variáveis de ambiente relevantes

| Var | Default | Função |
|---|---|---|
| `CODEBUILDER_WORKSPACE_ROOT` | `./workspaces` | Onde nascem os sandboxes |
| `CODEBUILDER_APPROVAL_WEBHOOK` | (vazio → console) | URL pra HITL |
| `CODEBUILDER_APPROVAL_WEBHOOK_SECRET` | — | Header `X-Codebuilder-...` |
| `CODEBUILDER_PROGRESS_WEBHOOK` | — | Eventos de subtask started/completed/failed |
| `CODEBUILDER_PROGRESS_WEBHOOK_SECRET` | — | idem |
| `CODEBUILDER_MAX_SUBTASK_RETRIES` | `3` | Retries por subtask no loop writer↔reviewer; tentativas totais = escrita inicial + retries |
| `CODEBUILDER_MAX_FINAL_QA_REPAIRS` | `1` em `patch_existing`, `2` em `new_project` | Tentativas de reparo após QA final falhar |
| `CODEBUILDER_PATCH_TEST_SCOPE` | full se houver testes | Compatibilidade: `full`/`all`/`whole` também força suíte inteira; sem arquivos de teste, falta de coleta é aviso |
| `CODEBUILDER_HISTORY_DB` | `./data/codebuilder_history.db` | SQLite de histórico |
| `CODEBUILDER_GUARDRAIL_LLM` | `openai/gpt-5.4-mini` | LLM que classifica feedback do humano |

Modelos dos agentes vivem nos `agents.yaml`, não em env vars.

---

## 13. Entrypoints

```bash
uv sync                # instala deps via uv
uv run kickoff         # roda CodebuilderFlow().kickoff(inputs={...hardcoded em main.py})
uv run plot            # gera codebuilder_flow.html (gráfico do flow)
crewai run             # equivalente a `uv run kickoff`

# Lint/test direto:
uv run ruff check src
uv run pytest -q
```

`main.py::kickoff()` traz um exemplo mínimo:

```python
CodebuilderFlow().kickoff(inputs={
    "session_id": "local-dev-session",
    "project_name": "criador-de-piada",
    "brief": "Um projeto python extremamente simples que cria piadas usando OpenAI",
    "goals": ["Criar piadas"],
    "tech_stack": ["python", "openai"],
    "attachments": [],
})
```

E `resume(job_id, feedback)`:

```python
CodebuilderFlow.from_pending(job_id).resume(feedback)
```

---

## 14. Passo a passo para construir algo parecido com CrewAI

Este é o template mental que você pode aplicar a um novo problema (Vivo poderia adaptar este mesmo arcabouço para outros domínios):

### Passo 1 — Modelar o estado
Defina um `FlowState` (pydantic) com **todos os campos** que precisam atravessar etapas. Inclua identificadores duais (session vs execution).

### Passo 2 — Esquematizar as saídas estruturadas
Para cada decisão importante (plano, artefato, revisão, QA), defina um `BaseModel` (idealmente com `extra="forbid"`). Isso vira o `output_pydantic` das tasks.

### Passo 3 — Quebrar em crews
Identifique **papéis distintos** (planner, executor, reviewer). Cada papel = uma crew com seu `agents.yaml` + `tasks.yaml`. Mantenha YAMLs como single-source-of-truth do prompt; o Python só configura LLM, tools, skills.

### Passo 4 — Criar tools sandboxed
Se os agentes precisam tocar disco/rede/recursos sensíveis, **sempre** crie tools que validem o input (path containment, allow-list, timeouts). NUNCA dê acesso a tools nativas sobre paths reais.

### Passo 5 — Adicionar skills por domínio
Skills carregam **guidelines longos** (checklists, padrões arquiteturais) sem inflar o system prompt. Use `discover_skills()` + `activate_skill()` e anexe por nome.

### Passo 6 — Montar o Flow
- `@start()` faz setup (workspace, ingestion).
- `@listen(<método anterior>)` ou `@listen("<evento string>")` encadeia.
- `@human_feedback(emit=[...], llm=GUARDRAIL_LLM)` para pausas + roteamento.
- Cada `@listen(...)` chama `XxxCrew().crew().kickoff(inputs={...})` e atualiza `self.state`.

### Passo 7 — Verificação determinística PRIMEIRO
Antes de pedir um reviewer LLM, faça **checks determinísticos** (lint, test, schema match, path match, regex de placeholder). LLM review é fallback caro — use só quando o determinístico não decide.

### Passo 8 — Repair loop limitado
Permita um retry com o erro como input. Mas adicione **circuit breaker** ("se a issue se repete, sai").

### Passo 9 — Persistência + retomada HITL
Use o `SQLiteFlowPersistence` default. Exponha `from_pending(id).resume(feedback)` num endpoint HTTP. Garanta que `CREWAI_STORAGE_DIR` é estável em runtime.

### Passo 10 — Histórico cross-run
Mantenha uma tabela leve (independente da persistence do CrewAI) com o resumo de runs anteriores. Injete-o como `{prior_history}` no prompt do planner.

### Passo 11 — Gates de domínio plugáveis
Use um registry (`dict[str, callable]`) indexado pelo slug do domínio. O planner emite o slug, o finalize despacha. Adicionar um novo domínio = 1 skill + 1 entrada no registry.

### Passo 12 — Observabilidade
Webhooks de progresso por evento (`subtask_started/completed/failed/final_qa_*`). Logs estruturados. Histórico em SQLite. Upload de artefatos pra storage durável (S3).

---

## 15. Coisas que ESPECIFICAMENTE NÃO fazer (lições aprendidas no Codebuilder)

1. **Não passe `id` no `kickoff(inputs=...)`.** Desalinha OTel traces no AMP/Wharf.
2. **Não mute `CREWAI_STORAGE_DIR` em runtime.** Quebra `from_pending().resume()` por drift de path do SQLite.
3. **Não aplique `@persist()` no Flow se você não quer per-step recovery.** O SQLiteFlowPersistence auto-criado pelo HITL é suficiente.
4. **Não dê `FileReadTool` nativo pros agentes** sobre paths reais. Use sempre Workspace*Tool com `resolve_within()`.
5. **Não confie no default `max_tokens=4096` da Anthropic.** Plans grandes e arquivos médios são truncados. Configure 16k–32k explicitamente.
6. **Não pule a validação Pydantic.** Use `output_pydantic` + `guardrail` em toda task estruturada. Sem isso, agentes inventam campos e o flow quebra a jusante.
7. **Não use `state.project_name` como chave de histórico.** Use `project_key` (canônico de git URL ou slug) — dois usuários no mesmo repo precisam compartilhar contexto.
8. **Não confie em "skipped" como passa.** Lint/test skipped = falha (`SKIP:` no output das tools).
9. **Não burn retries em falhas idênticas.** Detecte issues repetidas e short-circuit.
10. **História é observabilidade, não correção.** Sempre envolva `history.record(...)` em `try/except`.

---

## 16. Como evoluir para um novo caso de uso na Vivo

Use o Codebuilder como template:

1. **Identifique o domínio** (ex.: geração de bots Genesys, scripts de provisionamento, jobs Spark…).
2. **Crie a skill** em `src/codebuilder/skills/<dominio>/SKILL.md` com o padrão arquitetural da Vivo para aquele domínio (Clean Arch? camadas específicas? frameworks obrigatórios? logging/observabilidade interna?).
3. **Decida se precisa de gate determinístico** — se existe um padrão de pastas/arquivos obrigatórios, escreva uma função `run_<dominio>_deterministic_gate(build_dir) -> ReviewResult` em `runtime_qa.py`.
4. **Registre no `_ARCHITECTURE_GATES`**.
5. **Ajuste o tasks.yaml do planner** para que ele saiba quando emitir `domain="<dominio>"`.
6. **Anexe a skill** aos agentes relevantes (planner, writer, reviewer).
7. Para mudanças mais profundas (ex.: novo papel — "security-reviewer"), crie uma nova `Crew` no mesmo padrão e adicione um `@listen(...)` no flow.

O *flow shape* (`ingest → plan(HITL) → build loop → finalize`) é **estável** e funciona para qualquer geração de código com aprovação humana. O que muda entre domínios é o conteúdo das skills, dos gates e dos templates de prompt nos YAMLs.

---

## Apêndice — Diagrama resumido

```
        ┌───────────┐
inputs ─► ingest    │  cria workspace, materializa anexos, deriva project_key
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ plan      │  PlannerCrew (Plan validado) → @human_feedback
        └─────┬─────┘
       approved│   amend │   rejected
              ▼         ▼         ▼
        ┌───────────┐ ┌───────────────┐ ┌─────────────┐
        │ build     │ │ revise_plan   │ │ on_rejected │
        │ for each  │ │ + HITL again  │ │ status=fail │
        │  subtask: │ └───────┬───────┘ └─────────────┘
        │  Writer   │         │
        │   ↓ (write)         │
        │  determ. review     │
        │  (lint+test)        │
        │   ↓ falha?          │
        │  Reviewer LLM       │
        │  retry≤N            │
        └─────┬─────┘         │
              ▼               │
        ┌───────────┐         │
        │ finalize  │ ◄───────┘ (se aprovado depois)
        │  - run_final_qa (ruff + pytest)
        │  - repair pass (writer.repair_crew)
        │  - architecture gate por domain
        │  - patch / zip
        │  - upload S3
        │  - history.record
        └───────────┘
```

---

## Changelog

### 2026-05-27 — `patch_existing` no-op + grafos de import quebrados

Iteração disparada pelo feedback da Vivo (POC). Dois bugs reais corrigidos
de forma coordenada, mantendo o fluxo Planner → HITL → Writer↔Reviewer →
finalize/QA inalterado em forma — cada etapa só passou a entender uma coisa
nova.

**Bug A — modo `patch_existing` virava no-op.** A Vivo enviava um repo
existente + um markdown pedindo refactor (extrair método X de Y para um
novo arquivo Z). O writer devolvia Y byte-idêntico ao original; QA final
falhava. Causa raiz: o writer não distinguia "create" de "modify", o
orquestrador nunca lia o arquivo existente antes de chamar o writer, e o
prompt não dizia que a saída precisava diferir do snapshot.

**Bug B — modo `new_project` gerava grafos de import quebrados.** O zip da
falha mostrava `application/use_cases.py` importando
`terra_faturamento.domain.entities` — módulo nunca gerado. Também tinha 9
Protocols num único `repositories.py`. Três causas se compunham: planner
limitado a 15 subtasks (forçava bundling), skill RPA não dizia "uma classe
por arquivo" de forma explícita, e nada validava completude de imports
antes do QA final.

**O que mudou**

1. **`SubTask.change_type`** (`"create" | "modify"`). Para subtasks de
   modify, o orquestrador lê o arquivo existente *antes* do writer e passa
   o conteúdo via **side-channel** (não no schema do SubTask) — chaves
   `change_type` e `existing_contents` no kickoff input do writer. Snapshot
   truncado a 12k chars. Ver `main.py::_build_subtask`.

2. **Writer YAML branca em `change_type`**. Quando `modify`, mostra o
   bloco `EXISTING FILE` e exige que o conteúdo devolvido difira
   materialmente. Ver `crews/writer_crew/config/tasks.yaml`.

3. **Guard determinístico de no-op**. `run_deterministic_review` recebe
   `existing_snapshot=` e falha a review se o artifact persistido for
   igual (após `strip()`) ao snapshot — entra no loop de retry existente
   com a issue visível em `prior_review_issues`. Ver `runtime_qa.py`.

4. **Planner vira chain de 3 tasks** (skeleton → review → expand):
   - `skeleton_task` emite `PlanSkeleton` — enumera os arquivos com o
     **mínimo** necessário que respeite as regras do skill ativo,
     marcando `change_type` por arquivo. Também declara
     `external_packages` (libs internas que a Vivo anexa em `inputs/` e
     que o writer só vai *consumir*, não recriar).
   - `review_skeleton_task` faz uma crítica rápida do skeleton contra a
     checklist do skill (para RPA: 1 entidade/Protocol/use case por
     arquivo, `__init__.py` presente, scaffolding, etc.). Pode revisar
     ou passar inalterado. **Sem viés para editar.**
   - `expand_task` expande o skeleton (revisado) em `Plan` com
     `SubTask`s — uma por arquivo. Propaga `change_type` e
     `external_packages` para o Plan.
   - Cap de subtasks subiu de **15 → 60** (`runtime_qa._validate_plan`).
   - Encadeamento via `Task(context=[prior_task])`. Crew sequencial; o
     resultado pydantic final continua sendo o `Plan`.

5. **Gate de completude de imports** (`run_import_completeness_gate` em
   `runtime_qa.py`). Roda **depois do build loop, antes do QA final**.
   Anda em todos os `.py` com `ast`, resolve cada
   `from pkg.x import …` contra o workspace materializado, e respeita
   duas whitelists:
   - apenas o(s) top-level package(s) do próprio projeto (debaixo de
     `src/`) são candidatos a "missing" — qualquer coisa fora é tratada
     como externa (stdlib, terceiros, SDK interno);
   - qualquer package em `plan.external_packages` é ignorado pelo gate
     (a resolução em runtime — vendor / pyproject / sys.path — é uma
     conversa separada com a Vivo).

   Para cada módulo faltante, sintetiza **1 stub `SubTask`**
   (`change_type="create"`, descrição lista os símbolos que os
   importadores esperam). Cap de 8 stubs por execução; overflow vira
   nota no `qa_report.integration_notes` e marca o QA como `passed=False`.

6. **`rpa` SKILL.md** ganhou a seção *"Regra: um módulo = uma
   responsabilidade"* com exemplos concretos (`domain/entities/job.py` →
   `class Job`, `domain/repositories/job_repository.py` →
   `class JobRepository(Protocol)` etc.) e a regra de que helpers/config/
   logging/CLI glue continuam coesos no arquivo que os usa.

7. **Schemas novos**: `FileSkeleton(path, purpose, change_type)` e
   `PlanSkeleton(project_name, mode, domain, tech_stack, files,
   external_packages, open_questions, assumptions)`. `Plan` ganhou
   `external_packages: list[str]`.

**Cobertura de testes**

`tests/test_planning_and_imports.py` — 9 testes cobrindo: default do
`change_type`, no-op guard do deterministic review, gate de imports
(próprio pacote / whitelist `external_packages` / top-package
desconhecido / cap de 8 stubs), cap de 60 subtasks, roundtrip do
`PlanSkeleton`. Suite completa: 35 passed.

**Fora de escopo (deferido propositalmente)**

- Lista de `imports` por `SubTask` + validação de grafo pré-kickoff
  (o gate em runtime cobre o caso real e é mais barato de embarcar).
- Multi-pass do gate de completude (stubs podem ter, por sua vez,
  imports não resolvidos — fica para a próxima rodada se a Vivo
  reportar).
- Vendoring automático de bibliotecas de referência declaradas em
  `external_packages` (depende de como a Vivo estrutura a lib; decidir
  na próxima sessão).
- Arquitetura gate para `patch_existing` (continua só em `new_project`).

### 2026-05-28 — Performance: derrubar pre-passes de reasoning desnecessários

Após o fix de 2026-05-27, o primeiro teste end-to-end mostrou ~6 min só
na fase de planning, com o trace AMP exibindo um "Create Reasoning Plan"
antes da execução de cada uma das 3 tasks do planner — tudo em Opus. O
writer tinha o mesmo padrão (`reasoning: true`) e pagava pre-pass em
Sonnet para cada subtask, mesmo recebendo um `SubTask` já totalmente
especificado pelo planner.

**O que mudou**

1. **Writer perde `reasoning`.** `writer/config/agents.yaml` removeu
   `reasoning: true`. O writer já tem `description`/`tech_notes`/
   `test_criteria` por subtask e, em modo `modify`, o `existing_contents`
   via side-channel. Correção fica por conta do loop determinístico
   de review + retry — re-planning era puro desperdício de chamada LLM.
   Economiza ~1 chamada Sonnet por subtask (×30 subtasks no caso RPA).

2. **Planner: `reasoning: true` → `planning: true`.** No CrewAI
   instalado o `reasoning: true` está deprecated e auto-converte
   para `PlanningConfig(max_attempts=None)`. Trocamos para o toggle
   legacy não-deprecated `planning: true` — mesmo comportamento
   on/off, sem o warning de deprecation em cada instanciação de agent.

3. **Novo agent `planner_lite` em Sonnet 4.6.** As tasks
   `review_skeleton_task` (cheque mecânico contra o skill) e
   `expand_task` (transcrição mecânica do skeleton revisado em
   `SubTask`s) não precisam da criatividade arquitetural do Opus.
   Skeleton continua no `planner` (Opus 4.7) com `planning: true`;
   review e expand foram roteadas para `planner_lite` sem
   `planning`/`reasoning`. Reduz custo (2 das 3 chamadas saem do
   Opus) e latência (Sonnet TTFT/throughput muito maiores).

4. **`prior_history` removido do prompt do `expand_task`.** A
   skeleton já consome esse contexto e o expand recebe o skeleton
   revisado via `context=[…]` — re-injetar o histórico no expand era
   pagar tokens à toa em todas as runs subsequentes do mesmo projeto.

5. **Linguagem de "memória" varrida da documentação.** O codebuilder
   intencionalmente não usa `crew.memory=True`; o `project_history`
   é o único mecanismo de contexto entre runs. `CLAUDE.md`, `README.md`
   e `ARQUITETURA.md` foram limpos para refletir isso sem se referir
   a "memória"/"memory" em contextos onde a palavra induzia erro.
   Comentários inline obsoletos em `main.py` e `reviewer_crew.py`
   também caíram.

**Resultado esperado**

- Planning chain meta: ~3 min típico (1 chamada Opus + 2 chamadas
  Sonnet sequenciais), contra ~6 min no pior caso anterior.
- Custo da fase de planning cai ~⅔ (2 das 3 chamadas saem do Opus).
- Build loop: 1 chamada Sonnet por subtask em vez de 2 (sem
  reasoning pre-pass). Para um plano de 30 subtasks isso é ~30
  chamadas Sonnet a menos.
- Warning `DeprecationWarning: The 'reasoning' parameter is
  deprecated` some no startup da suíte de testes (verificado).
- Suíte de testes existente (35 testes) continua verde.

**Fora de escopo (deferido)**

- Migrar `planning: true` → `planning_config: PlanningConfig(...)`
  programaticamente. Esperar o CrewAI deprecar o toggle legacy.
- Remover a construção dead-code de `ReviewerCrew.crew()` por
  subtask em `_build_subtask` (objeto alocado mas nunca usado,
  porque `needs_fallback` é sempre False).
- Cachear a listagem do workspace entre subtasks (`workspace_listing`
  hoje é recomputada por iteração).

### 2026-05-28 (b) — Reverter `planner_lite` Sonnet

A primeira corrida em produção depois da introdução do `planner_lite`
(Sonnet 4.6 cobrindo `review_skeleton_task` + `expand_task`) falhou em
`runtime_qa.validate_plan` com `ValueError: Invalid plan: plan must
contain between 1 and 60 subtasks`. O `result.pydantic` da crew (output
da última task — o `expand_task`) chegou com `subtasks` vazio.

Hipótese mais provável: Sonnet 4.6 sob `output_pydantic=Plan` com
contexto via `Task(context=[…])` colapsou a lista de subtasks. Não
consegui reproduzir localmente (sem chave AMP) e a próxima
demonstração com a Vivo é amanhã, então o pragmático foi reverter o
roteamento Sonnet e manter os outros ganhos.

**O que reverteu**

- `planner_lite` removido de `planner_crew/config/agents.yaml` e de
  `planner_crew.py`.
- `review_skeleton_task` e `expand_task` voltaram a `agent: planner`
  (Opus 4.7).

**O que ficou (e por quê)**

- Writer ainda sem `reasoning`/`planning` — é o ganho mais limpo
  (correção é forçada pelo loop determinístico, não há motivo para
  re-planning).
- Planner em `planning: true` (em vez de `reasoning: true`) — sem
  mudança de comportamento, só silencia o warning de deprecation.
- `prior_history` continua fora do `expand_task` — a skeleton já
  consome esse contexto e o reviewed skeleton chega via
  `Task.context=[…]`.

**Próxima tentativa de Sonnet no planner (deferida)**

Antes de re-introduzir Sonnet em qualquer task que produza o `Plan`
final, adicionar um `guardrail=` deterministic no Task que assegure
`len(subtasks) >= 1` e cubra os campos obrigatórios. Sem isso,
qualquer drift de structured-output do Sonnet derruba o flow inteiro
em vez de só retentar a task.
