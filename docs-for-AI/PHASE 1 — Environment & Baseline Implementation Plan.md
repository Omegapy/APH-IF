# PHASE 1 — Environment & Baseline (Implementation Plan)

Got it. Here’s a **clean Phase 1 plan** with your constraints baked in:

* **No Git references** anywhere.
* **All non-essential `.py` files** are **comment-only** using your template (no code bodies yet).
* Still delivers: **Dockerized microservices + health checks**, **Streamlit status pill**, **pytest/ruff/mypy** runnable **without** pre-commit.

---
## comment-only template

```python

# -------------------------------------------------------------------------
# File: [your_module_name.py] (e.g., models.py, data_validators.py, ui_handlers.py)
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-07-2025
# [File Path] (e.g., backend/tools/your_module_name.py)
# ------------------------------------------------------------------------

# --- Module Objective ---
#   [A concise, high-level summary of what this *specific file/module* does.]
#   [Explain its purpose, the main functionalities it provides, and its role
#   within the larger project or package.]
#   Example: "This module defines the `HomeInventory` class, managing data
#   persistence and core CRUD operations for homes via file I/O."
#   Example: "Contains utility functions for validating various types of user
#   input, ensuring data integrity across the application."
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# [A brief, bulleted list of the primary classes, functions, or logical
#  components defined within this file. This helps at a glance.]
# - Class: [MyClassName]
# - Function: [my_utility_function]
# - Function: [another_related_function]
# - Constants: [MY_GLOBAL_CONSTANT]
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# [List any crucial external (third-party) or internal (local) modules
#  this specific file relies on. This helps understand module coupling.]
# - Standard Library: os (for file operations), sys (for path manipulation)
# - Third-Party: pandas (if using dataframes), requests (if making HTTP calls)
# - Local Project Modules:
#   - from .config import settings (if 'config' is a sibling module)
#   - from my_package.database import DBManager (if part of a larger package)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# [Explain how this file/module is intended to be used or integrated
#  by other parts of the project. Who should import it, and why?]
# Example: "The `HomeInventory` class from this module should be instantiated
#   by `main.py` or `app.py` to manage home data."
# Example: "Functions from this module (e.g., `validate_email`) are imported
#   into `user_management.py` and `forms.py` for input validation."

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

Description of the module functionality

example:
Circuit Breaker Implementation for External Service Resilience

Provides circuit breaker pattern to protect against cascading failures
when external services (Neo4j, OpenAI, Gemini) become unavailable.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
# import os
# import sys

# Third-party library imports
# import pandas as pd

# Local application/library specific imports
# from .utils import some_helper_function
# from ..config import APP_SETTINGS # Example for relative import in packages


# =========================================================================
# Global Constants / Variables
# =========================================================================
# Use ALL_CAPS for constants relevant to this module.
# MAX_RETRIES = 5 # Maximum attempts for an operation in this module
# DEFAULT_CHUNK_SIZE = 1024 # Buffer size for file operations


# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- class MyClassName
class [MyClassName]:
    """[A concise one-line summary of the class's purpose within this module.]

    [Provide a more detailed explanation of the class's responsibilities.]
    [Describe its key attributes, how it interacts with other components, and its role
    within the overall program. Discuss any invariants or pre/post conditions.]

    Class Attributes:
        [__class_attr_name (type)]: [Description of the class-level attribute.]
                                  [e.g., __instance_count (int): Tracks objects of this type.]

    Instance Attributes:
        [__instance_attr_name (type)]: [Description of the private instance attribute.]
        [public_attr_name (type)]: [Description of the public instance attribute.]

    Methods:
        [method_name()]: [Brief summary of each public method.]
        [_private_method()]: [Brief summary of each internal/private method.]
    """
    # ----------------------
    # --- Class Variable ---
    # ----------------------
    # [Define class-level variables here, if any.]
    _instance_count = 0
    
    # ---------------------------------------------------------------------------------

    # -------------------
    # --- Constructor ---
    # -------------------
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, [param1]: [type], [param2]: [type] = [default_value]) -> None:
        """Initializes the [MyClassName] object.

        [Provide a detailed explanation of what the constructor does.]
        [Describe any setup, initial state, or validations performed during object creation.]

        Args:
            [param1] ([type]): [Description of param1.]
            [param2] ([type], optional): [Description of param2.] Defaults to [default_value].
        """
        # [Initialize instance attributes here]
        self.__private_data = []
        self.public_property = [param1]
        # self.__class__._instance_count += 1  
    # --------------------------------------------------------------------------------- __init__()

    # ---------------------------------------------------------------------------------
    # --- Destructor (Use only if absolutely necessary for external resource cleanup) -
    # ---------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------- end __del__()
    def __del__(self) -> None:
        """Performs cleanup operations when the object is destroyed.
    
        [Explain what resources are being released or actions performed.]
        [e.g., Closes file handles, disconnects from a database, cleans up temp files.]
        """
        # [Cleanup code here]
        # print(f"Object {self.public_property} is being destroyed.")
    # --------------------------------------------------------------------------------- end __del__()

    # -----------------------
    # -- Embedded Classes --
    # ----------------------

    # --------------------------------------------------------------------------------- MyEmbeddedClass
    class [MyEmbeddedClass]
        # embedded class body logic
    # --------------------------------------------------------------------------------- end MyEmbeddedClass

    # -----------------------------------------------------------------------------
    # --- Getters (Property decorators are often preferred for simple getters) ---
    # -----------------------------------------------------------------------------

    # --------------------------------------------------------------------------------- some_property()
    @property
    def some_property(self) -> [type]:
        """[Retrieves the value of some_property.]"""
        return self.__private_data
    # --------------------------------------------------------------------------------- end some_property()

    # --------------------------------------------------------------------------------- get_data_by_id()
    def get_data_by_id(self, item_id: int) -> [type]:
        """Retrieves data based on a unique identifier.
    
        Args:
            item_id (int): The ID of the item to retrieve.
    
        Returns:
            [type]: The retrieved data, or None if not found.
        """
        # [Logic to retrieve data]
        pass
    # --------------------------------------------------------------------------------- end get_data_by_id()

    # ---------------------------
    # --- Setters / Mutators ---
    # ---------------------------
    
    # --------------------------------------------------------------------------------- set_value()
    def set_value(self, new_value: [type]) -> None:
        """Sets a new value for a specific attribute.
    
        Args:
            new_value ([type]): The value to set.
        """
        # [Logic to set/update data]
        pass
    # --------------------------------------------------------------------------------- end set_value()

    # --------------------------------------------------------------------------------- add_item()
    def add_item(self, item_data: dict) -> bool:
        """Adds a new item to the collection.
    
        Args:
            item_data (dict): Dictionary containing the item's details.
    
        Returns:
            bool: True if item was added successfully, False otherwise.
        """
        # [Logic to add item]
        pass
    # --------------------------------------------------------------------------------- end add_item()

    # -----------------------------------------------------------------------
    # --- Internal/Private Methods (Single leading underscore convention) ---
    # -----------------------------------------------------------------------

    # --------------------------------------------------------------------------------- _process_internal_data()
    def _process_internal_data(self, raw_data: list) -> list:
        """An internal helper method to process raw data.
    
        This method is not intended for direct external use.
    
        Args:
            raw_data (list): The raw data to be processed.
    
        Returns:
            list: The processed data.
        """
        # [Internal processing logic]
        return [processed_data]
    # --------------------------------------------------------------------------------- end _process_internal_data()

    # ---------------------------------------------------------------------
    # --- Class Information Methods (Optional, but highly recommended) ---
    # ---------------------------------------------------------------------

    # --------------------------------------------------------------------------------- __str__()
    def __str__(self) -> str:
        """Returns a user-friendly string representation of the [MyClassName] object.

        This method is primarily for end-user display (e.g., `print(object)`).
        """
        return "[A human-readable description of the object's state or summary.]"
    # --------------------------------------------------------------------------------- end __str__()

    # --------------------------------------------------------------------------------- __repr__()
    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the [MyClassName] object.

        This method is primarily for developers and debugging. The goal is that
        `eval(repr(obj))` should ideally recreate the object.
        """
        return f"[MyClassName]('{self.public_property}')" # Example, adapt to your constructor
    # --------------------------------------------------------------------------------- end __repr__()

# ------------------------------------------------------------------------- end class MyClassName   


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- my_utility_function()
def [my_utility_function]([param1]: [type]) -> [return_type]:
    """[A concise one-line summary of the function's purpose.]

    [Provide a more detailed explanation of what the function does.]
    [Describe its algorithm, any side effects, or important considerations.]

    Args:
        [param1] ([type]): [Description of param1.]
        [param2] ([type], optional): [Description of param2.] Defaults to [default_value].

    Returns:
        [return_type]: [Description of the value returned by the function.]

    Raises:
        [ExceptionType]: [Condition under which this exception is raised.]

    Examples:
        >>> [my_utility_function]([example_input])
        [expected_output]
    """
    # [Function implementation logic goes here]
    # Use inline comments for complex or non-obvious lines of code.
    # e.g., # This calculation ensures data normalization.
    pass
# --------------------------------------------------------------------------------- end my_utility_function()

# ------------------------
# --- Helper Functions ---
# ------------------------

# --------------------------------------------------------------------------------- get_user_confirmation()
def get_user_confirmation(prompt: str) -> bool:
    """Prompts the user for a yes/no confirmation.

    Args:
        prompt (str): The question to ask the user.

    Returns:
        bool: True if the user confirms (Y/y), False otherwise (N/n).
    """
    while True:
        choice = input(f"{prompt} [Y/N]: ").lower()
        if choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
# --------------------------------------------------------------------------------- end get_user_confirmation()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- process_module_data()
def process_module_data(data: list) -> dict:
    """Processes a list of data relevant to this module's scope.

    This function demonstrates a typical standalone function that might be
    called from another module's main logic.

    Args:
        data (list): The list of data items to process.

    Returns:
        dict: A dictionary of aggregated results.
    """
    # [Module-specific data processing logic]

    # -----------------------
    # -- Embedded Classes --
    # ----------------------

    # --------------------------------------------------------------------------------- my_embedded_function()
    def [my_embedded_function]()
        # embedded my_embedded_function body logic
        return ...
    # --------------------------------------------------------------------------------- end my_embedded_function()
    
    return {"processed_count": len(data)}
# --------------------------------------------------------------------------------- end process_module_data()

# =========================================================================
# Module Initialization / Main Execution Guard (if applicable)
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# For a file part of a larger program, it typically contains
# module-specific tests or example usage. It should *not* contain the main
# application logic, which belongs in the project's primary entry point (e.g., main.py).

if __name__ == '__main__':
    # --- Example Usage / Module-specific Tests ---
    print(f"Running tests for {__file__}...")

    # Example: Test a class defined in this module
    # try:
    #     my_test_object = [MyClassName]("test_value")
    #     print(f"Test object created: {my_test_object!r}") # Use !r for __repr__
    #     # Add specific test calls here
    #     # result = my_test_object.add_item({"id": 1, "name": "Test Item"})
    #     # print(f"Add item test: {result}")
    # except Exception as e:
    #     print(f"Error during module test: {e}")

    # Example: Test a standalone function
    # test_result = [my_utility_function](10)
    # print(f"Utility function test result: {test_result}")

    print(f"Finished tests for {__file__}.")   
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================

```

---

.env.example:

```
# APH-IF Main Project Configuration
# Store all environment variables in this file
# DO NOT commit this file to version control

# OpenAI API Configuration 
OPENAI_API_KEY = "sk-proj-......"
OPENAI_MODEL = "gpt-5"

# Google Gemini API Configuration (for graph building)
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.5-pro"

# Neo4j Database Configuration
NEO4J_URI = "neo4j+s://........databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = ""

# Embedding Configuration
EMBEDDING_MODEL = ""
EMBEDDING_DIMENSIONS = 

```

---

# PHASE 1 — Environment & Baseline (Revised)

## 0) Create the project skeleton

```bash
mkdir -p aph-if && cd aph-if

# root
mkdir -p documents doc_for_ai pdf_data test common

# backend
mkdir -p backend/app backend/tests

# data_processing
mkdir -p data_processing/app data_processing/tests

# frontend
mkdir -p frontend/app
```

### Root layout (what we’ll have by end of Phase 1)

```
aph-if/
├─ README.md
├─ .gitignore
├─ docker-compose.yml
├─ Makefile
├─ pyproject.toml
├─ test/
│  └─ e2e_test_query.py
├─ common/
│  ├─ __init__.py                     # comment-only template
│  ├─ config.py                       # comment-only template
│  ├─ logging.py                      # comment-only template
│  └─ models.py                       # comment-only template
├─ backend/
│  ├─ __init__.py                     # comment-only template
│  ├─ .env.example
│  ├─ requirements.txt
│  ├─ Dockerfile
│  ├─ tests/test_healthz.py
│  └─ app/
│     ├─ __init__.py                  # comment-only template
│     └─ main.py                      # minimal FastAPI with /healthz + stub /query
├─ data_processing/
│  ├─ __init__.py                     # comment-only template
│  ├─ .env.example
│  ├─ requirements.txt
│  ├─ Dockerfile
│  ├─ tests/test_healthz.py
│  └─ app/
│     ├─ __init__.py                  # comment-only template
│     └─ main.py                      # minimal FastAPI with /healthz
└─ frontend/
   ├─ __init__.py                     # comment-only template
   ├─ .env.example
   ├─ requirements.txt
   ├─ Dockerfile
   └─ app/bot.py                      # Streamlit with status pill
```

---

## 1) Add comment-only **placeholders** using your template

> For every file marked “comment-only template,” paste your template and fill the header fields.
> **Do not add executable code yet.** Keep them as documentation place-holders.

Examples (you’ll paste the full template and fill these header lines accordingly):

* `common/models.py`

  * **File:** `models.py`
  * **\[File Path]:** `common/models.py`
  * **Module Objective:** “Schemas planned for QueryRequest, RetrievedChunk, GraphHit, FusionInput, FusionOutput, Citation”
* `common/config.py` → config settings (env load later)
* `common/logging.py` → logging/tracing later
* `backend/__init__.py`, `backend/app/__init__.py`, `data_processing/__init__.py`, `data_processing/app/__init__.py`, `frontend/__init__.py` → brief purpose notes only

(Repeat for each placeholder file.)

---

## 2) Minimal, working services (health checks + stub)

### 2.1 Backend (`backend/app/main.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="APH-IF Backend")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "backend"}

# --- Stub interface for Phase 1 smoke test only ---
class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    top_k: int = 5
    min_score: float = 0.7

@app.post("/query")
def query(req: QueryRequest):
    return {
        "answer": f"Stub answer to: {req.query}",
        "citations": [],
        "retrieval": {
            "vector": {"hits": 0, "latency_ms": 0},
            "graph": {"hits": 0, "latency_ms": 0}
        },
        "meta": {"orchestrator_latency_ms": 1}
    }
```

**Tests** (`backend/tests/test_healthz.py`)

```python
from fastapi.testclient import TestClient
from app.main import app

def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["service"] == "backend"
```

**Reqs** (`backend/requirements.txt`)

```
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2
```

**Dockerfile** (`backend/Dockerfile`)

```dockerfile
FROM python:3.12.6
WORKDIR /app
COPY backend/requirements.txt /tmp/req.txt
RUN pip install -U pip && pip install -r /tmp/req.txt
COPY backend/app /app/app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

**Env example** (`backend/.env.example`)

```
# Reserved for later; Phase 1 doesn’t require secrets here.
```

---

### 2.2 Data Processing (`data_processing/app/main.py`)

```python
from fastapi import FastAPI

app = FastAPI(title="APH-IF Data Processing")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "data_processing"}
```

**Tests** (`data_processing/tests/test_healthz.py`)

```python
from fastapi.testclient import TestClient
from app.main import app

def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["service"] == "data_processing"
```

**Reqs** (`data_processing/requirements.txt`)

```
fastapi==0.115.0
uvicorn==0.30.6
```

**Dockerfile** (`data_processing/Dockerfile`)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY data_processing/requirements.txt /tmp/req.txt
RUN pip install -U pip && pip install -r /tmp/req.txt
COPY data_processing/app /app/app
EXPOSE 8010
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8010"]
```

**Env example** (`data_processing/.env.example`)

```
# Reserved for later; Phase 1 doesn’t require secrets here.
```

---

### 2.3 Frontend (`frontend/app/bot.py`)

```python
import os, requests, streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="APH-IF", layout="wide")
st.title("APH-IF • Frontend")

def backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/healthz", timeout=2)
        return r.ok
    except Exception:
        return False

ok = backend_health()
pill = "🟢 Backend OK" if ok else "🔴 Backend Down"
st.sidebar.markdown(f"**Status:** {pill}")

q = st.text_input("Your question")
if st.button("Ask") and q:
    with st.spinner("Querying backend…"):
        r = requests.post(f"{BACKEND_URL}/query", json={"query": q})
        if r.ok:
            st.write(r.json().get("answer", "No answer"))
        else:
            st.error(f"Backend error: {r.status_code}")
```

**Reqs** (`frontend/requirements.txt`)

```
streamlit==1.37.1
requests==2.32.3
```

**Dockerfile** (`frontend/Dockerfile`)

```dockerfile
FROM python:Python 3.13.6
WORKDIR /app
COPY frontend/requirements.txt /tmp/req.txt
RUN pip install -U pip && pip install -r /tmp/req.txt
COPY frontend/app /app/app
ENV BACKEND_URL=http://aph_if_backend:8000
EXPOSE 8501
CMD ["streamlit","run","app/bot.py","--server.address=0.0.0.0","--server.port","8501"]
```

**Env example** (`frontend/.env.example`)

```
BACKEND_URL=http://localhost:8000
```

---

## 3) Compose + tooling

**Compose** (`docker-compose.yml`)

```yaml
version: "3.9"
services:
  aph_if_backend:
    build: ./backend
    env_file: ./backend/.env
    container_name: aph_if_backend
    ports: ["8000:8000"]
    restart: unless-stopped

  aph_if_data_processing:
    build: ./data_processing
    env_file: ./data_processing/.env
    container_name: aph_if_data_processing
    ports: ["8010:8010"]
    restart: unless-stopped

  aph_if_frontend:
    build: ./frontend
    env_file: ./frontend/.env
    container_name: aph_if_frontend
    ports: ["8501:8501"]
    depends_on: ["aph_if_backend"]
    restart: unless-stopped
```

**Makefile** (`Makefile`)

```makefile
.PHONY: dev-up dev-down logs test lint type

dev-up:
\tdocker-compose up -d --build

dev-down:
\tdocker-compose down -v

logs:
\tdocker-compose logs -f

test:
\tpytest -q

lint:
\truff check .

type:
\tmypy .
```

**Pyproject** (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
extend-select = ["I"]
exclude = ["venv", ".venv", "dist"]

[tool.mypy]
python_version = "3.12"
warn_unused_ignores = true
warn_redundant_casts = true
warn_unreachable = true
strict_optional = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-q"
pythonpath = ["."]
```

**.gitignore** (`.gitignore`)
*(Not Git-binding; it’s harmless to have this file. If you prefer, skip it.)*

```
# Ignore all directories starting with 'i-'
i-*/

# Ignore all directories starting with 'ex-'
ex-*/

# Ignore files starting with 'i-' at any directory depth
**/i-*
i-*

# Ignore all directories starting with 'docs for AI'
docs for AI/

# Ignore the gitignore file itself
.gitignore

# API Keys and Secrets
.env
.streamlit/secrets.toml
secrets/
*.key
*.pem

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock
#poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#   pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
#   https://pdm-project.org/en/latest/usage/project/#working-with-version-control
#pdm.lock
#pdm.toml
.pdm-python
.pdm-build/

# pixi
#   Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
#pixi.lock
#   Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
#   in the .venv directory. It is recommended not to include this directory in version control.
.pixi

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.vscode/
.cursorrules
.cursorignore
.cursorindexingignore
.cursorrules
.augment/
.streamlit/.secrets.toml

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#  Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore 
#  that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#  and can be added to the global gitignore or merged into this file. However, if you prefer, 
#  you could uncomment the following to ignore the entire vscode folder
# .vscode/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/

# APH-IF Security - Environment Variables and Secrets
# Added for secure Docker deployment
.env
.env.local
.env.*.local
*.env
.env.backup

# Secrets and API Keys
*.key
*.pem
*.p12
api_keys.txt
credentials.json

# Streamlit Secrets (backup files)
.streamlit/secrets.toml.backup
.streamlit/secrets.toml.orig
secrets.toml.backup

# Docker and deployment
.dockerignore.backup
docker-compose.override.yml

# Migration and temporary files
*.backup
*.bak
*.orig
.tmp/
temp/
```

---

## 4) E2E smoke test (no external deps)

**Test file** (`test/e2e_test_query.py`)

```python
import time, requests

def test_backend_stub():
    # wait a bit for container to come up (if running via compose)
    deadline = time.time() + 10
    ok = False
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:8000/healthz", timeout=1.0)
            ok = r.ok
            if ok:
                break
        except Exception:
            time.sleep(0.5)
    assert ok, "backend healthz failed"

    resp = requests.post("http://localhost:8000/query", json={"query": "What is APH-IF?"}, timeout=2.0)
    assert resp.ok
    data = resp.json()
    assert isinstance(data.get("answer"), str) and "Stub answer" in data["answer"]
```

---

## 5) Bring it up & verify

```bash
# Start services
make dev-up

# Health checks
curl -s http://localhost:8000/healthz
curl -s http://localhost:8010/healthz

# Open UI
# http://localhost:8501

# Run tests and static checks (from host)
pip install pytest ruff mypy
make test
make lint
make type
```

**Expected:**

* `/healthz` returns `{"status":"ok","service":"backend"}` and `{"status":"ok","service":"data_processing"}`
* Streamlit sidebar shows **🟢 Backend OK**
* `pytest -q` passes; ruff/mypy clean (or only trivial warnings).

---

## Deliverables (Phase 1)

* Running containers via `make dev-up` / `make dev-down`.
* Backend & data\_processing health checks are green; Streamlit **status pill** reflects backend.
* Basic E2E smoke test passes (backend stub `/query` responds).

## Definition of Done

* ✅ All three services **build** and **run**.
* ✅ `pytest -q` passes (unit + E2E smoke).
* ✅ `docker-compose up` yields healthy endpoints and working frontend.

---

### Notes on your template usage

* Every non-essential Python file in **Phase 1** is **documentation-only** using your provided comment template (no code).
* We introduced **minimal executable code only** where absolutely required to meet Phase 1 goals:

  * `backend/app/main.py`, `data_processing/app/main.py`, `frontend/app/bot.py`
  * The two tiny `tests/test_healthz.py`
  * The E2E smoke test `test/e2e_test_query.py`
* In **Phase 2+**, we’ll replace those comment-only placeholders with actual implementations (schemas, logging, etc.) without changing the folder structure.
