# Local RLGym documentation

Offline copies of public RLGym docs for this project.

| Folder | Contents | Upstream |
|---|---|---|
| `rlgym-site/` | Tutorials & guides (Getting Started, Rocket League, Learn, Tools, Cheatsheets) | [rlgym.org](https://rlgym.org/) |
| `rlgym-api/` | Sphinx AutoAPI class/module reference | [API docs](https://captainglac1er.github.io/rocket-league-gym/) |
| `rlgym/` | Upstream Sphinx scaffold from the RLGym GitHub `docs/` tree | [RLGym/rlgym](https://github.com/RLGym/rlgym/tree/main/docs) |

Refresh mirrors:

```bash
.\.venv\Scripts\python.exe scripts\scrape_rlgym_docs.py
.\.venv\Scripts\python.exe scripts\scrape_rlgym_api_docs.py
```
