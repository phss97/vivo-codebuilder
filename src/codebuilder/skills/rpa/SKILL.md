---
name: rpa
description: Apply the required Python RPA project architecture, packaging, configuration, and testing standards (orchestrator/producer/consumer, Clean Architecture, CCM-backed config, pytest coverage).
---

# Guia de Onboarding — Projetos RPA em Python

> Material de apoio para novos membros do time. Reúne as premissas técnicas e arquiteturais que todo projeto novo deve seguir desde o primeiro commit.

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura de Processos — Orquestrador, Produtor e Consumidor](#2-arquitetura-de-processos--orquestrador-produtor-e-consumidor)
3. [Clean Architecture — Organização do Código](#3-clean-architecture--organização-do-código)
4. [Modelagem do Banco de Dados](#4-modelagem-do-banco-de-dados)
5. [Princípios SOLID](#5-princípios-solid)
6. [TDD com Padrão Given-When-Then](#6-tdd-com-padrão-given-when-then)
7. [Stack de Qualidade e Testes](#7-stack-de-qualidade-e-testes)
8. [Estrutura de Pastas Recomendada](#8-estrutura-de-pastas-recomendada)
9. [Empacotamento — Executável Windows (.exe)](#9-empacotamento--executável-windows-exe)
10. [Checklist para Iniciar um Projeto Novo](#10-checklist-para-iniciar-um-projeto-novo)

---

## 1. Visão Geral

**Stack padrão:**

- Python 3.13
- `uv` para gerenciamento de dependências e ambiente virtual
- `hatchling` como build backend
- `pytest` + `pytest-cov` para testes (mínimo 80% de cobertura)
- `ruff` para lint e formatação
- `mypy` para análise estática de tipos
- Layout `src/` (código em `src/<package_name>/`)

---

## 2. Arquitetura de Processos — Orquestrador, Produtor e Consumidor

A automação é dividida em três processos com responsabilidades claras.

### 2.1. Orquestrador

Ponto de entrada da aplicação. Não executa regra de negócio — coordena.

**Responsabilidades:**

- Inicializar a aplicação (logger, configurações, container de dependências).
- Estabelecer e gerenciar a conexão com o banco de dados.
- Buscar credenciais e segredos no CCM
- Registrar o job em execução (`tbl_Job`) e adquirir o lock (`tbl_Lock`).
- Disparar o produtor e o consumidor.
- Garantir liberação do lock e fechamento limpo dos recursos ao final (sucesso ou falha).

### 2.2. Produtor

Alimenta a fila de trabalho.

**Responsabilidades:**

- Acessar o diretório de rede onde os arquivos de input são depositados.
- Validar nomenclatura dos arquivos (regex, máscara, prefixos esperados).
- Validar regras de negócio sobre o conteúdo dos arquivos (colunas obrigatórias, tipos, integridade referencial básica).
- Inserir registros válidos em `tbl_WorkQueueItem`.

### 2.3. Consumidor

Executa o trabalho propriamente dito.

**Responsabilidades:**

- Buscar itens de `tbl_WorkQueueItem`.
- Executar a regra de negócio: integração com SAP, leitura/escrita de Excel, chamadas a APIs, etc.
- Atualizar o item ao final, registrando duração e mensagens.
- Aplicar política de retry/attempts quando aplicável

---

## 3. Clean Architecture — Organização do Código

O código é organizado em três camadas, com a regra de dependência apontando sempre para dentro: **infrastructure → application → domain**. O domínio nunca conhece a infraestrutura.

### 3.1. `domain/`

Coração do projeto. Independente de frameworks, bibliotecas e detalhes técnicos.

- **Entidades**: classes que representam conceitos de negócio.
- **Exceções de negócio**: `InvalidInvoiceError`, `DuplicatedReferenceError`, `BusinessRuleViolationError`.
- **Interfaces (Protocols)**: contratos que serão implementados na camada de infraestrutura.

### 3.2. `application/`

Orquestra as entidades do domínio para executar casos de uso. Não conhece detalhes de banco, SAP, Excel

- **Use Cases / Services**: `ProcessWorkItemUseCase`, `EnqueueFilesFromDirectoryUseCase`, `AcquireJobLockService`.
- **DTOs**: estruturas para entrada e saída dos casos de uso.
- **Mapeadores**: conversão entre entidades do domínio e DTOs.

### 3.3. `infrastructure/`

Implementações concretas.

- **Configurações**: leitura de `.env`, conexão com CCM.
- **Repositórios**: implementações concretas dos contratos do domínio.
- **Integrações**: clientes SAP, leitores/escritores de Excel, acesso a diretórios de rede.
- **Logging**: configuração de handlers, formatters.

---

## 4. Modelagem do Banco de Dados

Cinco tabelas dão suporte ao ciclo completo: execução do job → travamento → fila lógica → itens de trabalho → log.

### 4.1. `tbl_Job`

Histórico de cada execução da automação.

### 4.2. `tbl_Lock`

Garante execução única. Impede que duas instâncias do mesmo job rodem simultaneamente.

### 4.3. `tbl_WorkQueue`

Fila lógica. Cada automação pode ter uma ou mais filas (ex.: `FaturamentoTerra_01Produtor_WQ`, `FaturamentoTerra_02Consumidor_WQ`).

### 4.4. `tbl_WorkQueueItem`

Os itens propriamente ditos. É o coração operacional.

### 4.5. `tbl_Log`

Log centralizado da automação. Complementa (não substitui) os logs em arquivo.

---

## 5. Princípios SOLID

Aplicar sempre que fizer sentido.

- **S — Single Responsibility**: cada classe tem uma única razão para mudar. Um repositório acessa banco; um serviço orquestra regra; um cliente SAP fala com SAP.
- **O — Open/Closed**: o código é aberto para extensão e fechado para modificação. Novos tipos de validação, por exemplo, entram como novas implementações de um `Validator` — não como `if/elif` crescente.
- **L — Liskov Substitution**: qualquer implementação de uma interface deve ser substituível sem quebrar o comportamento esperado pelos consumidores.
- **I — Interface Segregation**: prefira várias interfaces pequenas e específicas a uma interface gorda. Um caso de uso que só lê dados não deveria depender de uma interface que também escreve.
- **D — Dependency Inversion**: módulos de alto nível (use cases) dependem de abstrações, não de implementações concretas. Quem injeta as implementações é a camada de composição (orquestrador / container de DI).

---

## 6. TDD com Padrão Given-When-Then

Todo código novo nasce de um teste. O teste descreve o comportamento esperado antes da implementação existir.

### 6.1. Ciclo

1. **Red**: escreva um teste que falha.
2. **Green**: escreva o código mínimo para o teste passar.
3. **Refactor**: melhore a implementação mantendo o teste verde.

### 6.2. Estrutura GWT

Cada teste tem três blocos claros:

- **Given (Arrange)**: contexto e pré-condições.
- **When (Act)**: a ação sendo testada.
- **Then (Assert)**: o resultado esperado.

### 6.3. Exemplo

```python
def test_should_mark_item_as_failed_when_sap_integration_raises():
    # Given
    item = WorkQueueItemFactory.build(status=Status.IN_PROGRESS)
    sap_client = FakeSapClient(raises=SapConnectionError("timeout"))
    repository = InMemoryWorkQueueItemRepository(items=[item])
    use_case = ProcessWorkItemUseCase(sap_client=sap_client, repository=repository)

    # When
    use_case.execute(item_id=item.id)

    # Then
    updated = repository.get_by_id(item.id)
    assert updated.status == Status.FAILED
    assert "timeout" in updated.error_message
```

### 6.4. Boas práticas

- Nome do teste descreve o comportamento, não a implementação: `test_should_<resultado>_when_<condição>`.
- Um assert por conceito (não necessariamente um assert por teste).
- Testes são isolados: não compartilham estado, não dependem de ordem.
- Mocks/fakes ficam restritos aos testes — código de produção não conhece test doubles.
- Dependências externas (banco, SAP, sistema de arquivos) são abstraídas via interface no domínio e mockadas nos testes unitários.

---

## 7. Stack de Qualidade e Testes

### 7.1. `pytest`

Framework de testes. Roda testes unitários, de integração e end-to-end.

```bash
uv run pytest
```

### 7.2. `pytest-cov`

Cobertura de testes. Mínimo de **80%** para qualquer projeto novo.

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

### 7.3. `ruff`

Lint e formatação.

```bash
uv run ruff check .
uv run ruff format .
```

### 7.4. `mypy`

Verificação estática de tipos. Toda função pública tem type hints.

```bash
uv run mypy src/
```

---

## 8. Estrutura de Pastas Recomendada

```
app-meu-projeto/
├── src/
│   └── meu_projeto/
│       ├── __init__.py
│       ├── __main__.py             # entry-point p/ PyInstaller: chama o main() do orquestrador
│       ├── domain/
│       │   ├── entities/
│       │   ├── exceptions/
│       │   └── repositories/        # Protocols/interfaces
│       ├── application/
│       │   ├── use_cases/
│       │   ├── services/
│       │   └── dtos/
│       └── infrastructure/
│           ├── config/
│           ├── persistence/         # implementações de repositórios
│           ├── integrations/
│           │   ├── sap/
│           │   ├── excel/
│           │   └── network_drive/
│           └── logging/
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── e2e/
├── .env.example
├── build.spec                      # spec do PyInstaller (Analysis aponta p/ __main__.py)
├── build.ps1                       # script de build no Windows (gera dist/<nome>.exe)
├── pyproject.toml
├── README.md
└── uv.lock
```

### Regra: um módulo = uma responsabilidade

Cada arquivo `.py` deve conter **exatamente uma** classe pública principal
(entidade, Protocol, caso de uso, service ou adapter). Exemplos concretos:

- `domain/entities/job.py` → `class Job`
- `domain/entities/work_queue_item.py` → `class WorkQueueItem`
- `domain/exceptions/job_locked_error.py` → `class JobLockedError(Exception)`
- `domain/repositories/job_repository.py` → `class JobRepository(Protocol)`
- `domain/repositories/sap_client.py` → `class SapClient(Protocol)`
- `domain/repositories/excel_reader.py` → `class ExcelReader(Protocol)`
- `application/use_cases/process_job.py` → `class ProcessJobUseCase`
- `application/use_cases/acquire_job_lock.py` → `class AcquireJobLockUseCase`
- `infrastructure/integrations/excel/excel_invoice_reader.py` → `class ExcelInvoiceReader`
- `infrastructure/integrations/sap/sap_client_impl.py` → `class SapClientImpl`

Nunca agrupe múltiplas entidades, múltiplos Protocols, múltiplas exceções, nem
múltiplos casos de uso em um único arquivo. O planner deve manter um arquivo
por responsabilidade pública, mas pode agrupar vários arquivos relacionados em
um mesmo subtask de execução. O writer deve gerar exatamente um artefato por
arquivo planejado dentro do subtask. Helpers privados (funções `_foo`) ficam no
mesmo arquivo da classe que os usa — não fragmente por fragmentar.
Configuração, logging e glue de CLI podem viver em um arquivo coeso por
concern; a regra de um-por-arquivo vale apenas para entidades, Protocols,
exceções, casos de uso, services e adapters.

---

## 9. Empacotamento — Executável Windows (.exe)

Todo projeto RPA novo (`new_project`) deve ser entregue com um **kit de
build** que gera um executável Windows (`.exe`) via PyInstaller. A execução
por terminal (`uv run`, `[project.scripts]`) **permanece inalterada** — o
`.exe` apenas embrulha o mesmo `main()` do orquestrador.

> PyInstaller **não faz cross-build**: o `.exe` precisa ser gerado em uma
> máquina Windows. O kit é entregue pronto; o build em si roda no ambiente
> Windows do cliente.

### 9.1. Entry-script real (obrigatório)

O `Analysis` do PyInstaller recebe o **caminho de um arquivo `.py`**, não o
nome de um console_script. Por isso o projeto precisa de um módulo de entrada
explícito que o spec possa apontar — `src/<pacote>/__main__.py` (ou um
`launcher.py` na raiz). Ele apenas importa o orquestrador e chama `main()`:

```python
# src/meu_projeto/__main__.py
"""Entry-point para empacotamento (PyInstaller) e `python -m meu_projeto`."""
from meu_projeto.infrastructure.orchestrator import main  # ajuste ao seu orquestrador

if __name__ == "__main__":
    main()
```

O `[project.scripts]` (terminal) e este `__main__.py` (`.exe`) chamam o
**mesmo** `main()` — comportamento idêntico nos dois caminhos.

### 9.2. `build.spec` (PyInstaller, onefile + console)

```python
# build.spec — rode com: pyinstaller build.spec
block_cipher = None

a = Analysis(
    ['src/meu_projeto/__main__.py'],   # caminho REAL do entry-script
    pathex=['src'],
    binaries=[],
    datas=[('.env.example', '.')],     # inclua aqui config/templates lidos em runtime
    hiddenimports=[],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='meu_projeto',
    console=True,          # RPA roda em terminal; mantenha True
    strip=False,
    upx=True,
    runtime_tmpdir=None,
)
```

### 9.3. Script de build no Windows (`build.ps1`)

```powershell
# build.ps1 — gera dist\meu_projeto.exe (rode em uma máquina Windows)
$ErrorActionPreference = "Stop"

uv sync                       # instala dependências, incluindo pyinstaller (dev)
uv run pyinstaller build.spec --noconfirm

Write-Host "Executável gerado em dist\meu_projeto.exe"
```

> Se `uv` não estiver disponível, use o fallback com venv:
> `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install . pyinstaller; pyinstaller build.spec --noconfirm`.

### 9.4. Dependência de desenvolvimento

`pyinstaller` deve constar no `pyproject.toml` como dependência de dev:

```toml
[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
    "mypy>=1.11",
    "pyinstaller>=6",
]
```

### 9.5. Seção no README

O README deve trazer uma seção **"Gerar o executável (Windows)"** com os
comandos exatos:

````markdown
## Gerar o executável (Windows)

Pré-requisitos: Windows, Python 3.13 e uv.

```powershell
.\build.ps1
```

Gera `dist\meu_projeto.exe` (mesmo `main()` do `uv run`). Para executar:

```powershell
.\dist\meu_projeto.exe
```
````

### 9.6. Checklist de empacotamento

- [ ] Entry-script `__main__.py` (ou `launcher.py`) chamando `main()`.
- [ ] `build.spec` com `Analysis` apontando para o caminho do entry-script.
- [ ] `build.ps1` (e/ou `build.bat`) na raiz.
- [ ] `pyinstaller` nas dependências de dev do `pyproject.toml`.
- [ ] Seção "Gerar o executável (Windows)" no README.
- [ ] `datas` do spec inclui todo arquivo de config/template lido em runtime.
