# Domain Docs

This repository uses a **single-context** domain documentation layout.

## Before domain-oriented exploration

Read the following when they exist and are relevant to the work:

- `CONTEXT.md` at the repository root
- Relevant decision records under `docs/adr/`

If these files do not exist, proceed silently. Create them only when terminology or an architectural decision needs to be recorded.

## Consumer rules

- Use terms defined in `CONTEXT.md` consistently in code, issues, tests, and documentation.
- Surface conflicts with existing ADRs rather than silently overriding them.
- Store system-level architecture decisions under `docs/adr/`.
