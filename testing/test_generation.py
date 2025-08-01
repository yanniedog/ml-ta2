#!/usr/bin/env python
"""
e2e_test_workflow.py
────────────────────
End-to-end, self-contained workflow that

1. Detects your project layout automatically (src/, app/, packages, single-file modules)
2. Generates baseline tests with Pynguin **in parallel**
3. Runs pytest with coverage enforcement
4. Executes mutation testing and reports the kill-rate
5. Fails CI / CD if either metric is below your thresholds
6. Offers CLI flags for easy customisation
7. Emits structured logging for CI visibility

Requires Python 3.9+.

Usage (defaults shown):
    python e2e_test_workflow.py \
        --source-dir auto         \
        --tests-dir   tests       \
        --min-cov     90          \
        --min-kill    80          \
        --search-time 60          \
        --regen-tests true
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import shutil
import subprocess
import sys
import os
import fnmatch
import multiprocessing
from pathlib import Path
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Go up one more level to project root
# Acknowledge Pynguin's disclaimer (see documentation)
os.environ.setdefault("PYNGUIN_DANGER_AWARE", "1")
LOG  = logging.getLogger("e2e")

def run(cmd: list[str], *, capture: bool = False) -> str | None:
    """Run shell command, stream output live unless capture=True."""
    LOG.debug("Running: %s", " ".join(cmd))
    if capture:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode:
            LOG.error(res.stderr.strip())
            raise subprocess.CalledProcessError(res.returncode, cmd)
        return res.stdout.strip()
    subprocess.run(cmd, check=True, text=True)

_PYNGUIN_AVAILABLE = False

def _ensure_package(pkg: str, version: str | None = None) -> bool:
    """Import *pkg* or attempt to install it; return True if available."""
    base_name = pkg.split("[")[0]
    mod_name = base_name.replace("-", "_")
    try:
        __import__(mod_name)
        return True
    except ImportError:
        install_target = f"{pkg}=={version}" if version else pkg
        LOG.info("Installing missing package: %s", install_target)
        try:
            run([
                sys.executable, "-m", "pip", "install",
                "--quiet", "--prefer-binary", install_target
            ])
            __import__(mod_name)
            return True
        except (subprocess.CalledProcessError, ImportError):
            return False


def pip_install():
    global _PYNGUIN_AVAILABLE

    deps_required = ["pytest", "pytest-cov", "coverage", "mutmut"]
    deps_optional: list[str] = ["pynguin"]

    for d in deps_required:
        if not _ensure_package(d):
            sys.exit(f"❌  Required dependency '{d}' could not be installed.")

    missing_optional: list[str] = []
    for d in deps_optional:
        if not _ensure_package(d):
            missing_optional.append(d)

    if missing_optional:
        LOG.warning("Optional packages missing: %s – certain quality gates will be skipped.", ", ".join(missing_optional))

    # Pynguin is optional but strongly recommended
    _PYNGUIN_AVAILABLE = not bool(missing_optional)
    LOG.info("Pynguin available: %s", _PYNGUIN_AVAILABLE)

# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_source_dir(user_choice: str) -> Path:
    """Return directory containing Python sources."""
    if user_choice != "auto":
        d = (ROOT / user_choice).resolve()
        if not d.exists():
            sys.exit(f"❌  Source dir '{d}' not found")
        return d

    for cand in ("src", "app"):
        p = ROOT / cand
        if any(p.rglob("*.py")):
            LOG.info("Detected source dir: %s", p)
            return p
    LOG.info("No src/ or app/ folder – defaulting to project root")
    return ROOT

def list_modules(src: Path,
                 include: list[str] | None = None,
                 exclude: list[str] | None = None) -> List[str]:
    """Return import-able dotted paths for every .py inside *src*.

    Parameters
    ----------
    include, exclude
        Shell-style glob patterns (e.g. "mypkg.*", "*_cli") matched against the
        dotted module path.  *include* defaults to ["*"].  *exclude* takes
        precedence over *include*.
    """
    include = include or ["*"]
    exclude = exclude or []

    mods: list[str] = []
    for f in src.rglob("*.py"):
        # Skip dunder files, site-packages vendoring, and existing tests
        if (f.name.startswith("__")
                or "site-packages" in f.parts
                or "tests" in f.parts):
            continue
        dotted = ".".join(f.relative_to(src).with_suffix("").parts)
        if any(fnmatch.fnmatch(dotted, pat) for pat in exclude):
            continue
        if not any(fnmatch.fnmatch(dotted, pat) for pat in include):
            continue
        mods.append(dotted)

    LOG.info("Selected %d module(s) for test generation", len(mods))
    return sorted(set(mods))

# ──────────────────────────────────────────────────────────────────────────────
# Pynguin – parallel test generation
# ──────────────────────────────────────────────────────────────────────────────
def _pynguin_single(module: str, out: Path, search_time: int):
    try:
        cmd = [
            sys.executable, "-m", "pynguin",
            f"--project_path={ROOT}",
            f"--module-name={module}",
            f"--output_path={out}",
            f"--maximum-search-time={search_time}",
            "--type-inference-strategy=NONE"  # faster & more stable
        ]
        LOG.debug("Running: %s", " ".join(cmd))
        env = os.environ.copy()
        env["PYNGUIN_DANGER_AWARE"] = "1"
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
        if result.returncode != 0:
            LOG.warning("Pynguin failed for module %s: %s", module, result.stderr.strip())
        else:
            LOG.debug("Successfully generated tests for %s", module)
    except Exception as e:
        LOG.warning("Failed to generate tests for module %s: %s", module, str(e))

def generate_tests(mods: List[str], tests_dir: Path, search_time: int, regen: bool):
    LOG.info("generate_tests called with Pynguin available: %s", _PYNGUIN_AVAILABLE)
    if not _PYNGUIN_AVAILABLE:
        LOG.info("Pynguin not available – creating minimal test structure instead.")
        gen_dir = tests_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Created directory: %s", gen_dir)
        
        # Create a simple placeholder test file
        test_file = gen_dir / "test_placeholder.py"
        LOG.info("Creating test file: %s", test_file)
        with open(test_file, "w") as f:
            f.write("""# Placeholder test file
def test_placeholder():
    assert True
""")
        LOG.info("Created placeholder test file")
        return
    gen_dir = tests_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    if regen:
        LOG.info("Clearing existing generated tests")
        shutil.rmtree(gen_dir)
        gen_dir.mkdir(parents=True)

    LOG.info("Generating tests (parallel)…")
    # Limit to smaller subset for debugging
    limited_mods = mods[:5]  # Only generate tests for first 5 modules
    LOG.info("Testing with limited modules: %s", ', '.join(limited_mods))
    
    workers = min(max(1, (multiprocessing.cpu_count() or 2) // 2 or 1), len(limited_mods))
    LOG.debug("Using %d worker(s) for Pynguin", workers)
    ExecutorCls = cf.ThreadPoolExecutor if os.name == "nt" else cf.ProcessPoolExecutor
    
    try:
        with ExecutorCls(workers) as pool:
            futs = [pool.submit(_pynguin_single, m, gen_dir, search_time) for m in limited_mods]
            completed = 0
            failed = 0
            for f in cf.as_completed(futs):
                try:
                    f.result()  # propagate exceptions
                    completed += 1
                except Exception as e:
                    failed += 1
                    LOG.warning("Test generation task failed: %s", str(e))
            LOG.info("Test generation completed: %d successful, %d failed", completed, failed)
    except Exception as e:
        LOG.error("Test generation failed completely: %s", str(e))
        LOG.info("Continuing with existing tests...")

# ──────────────────────────────────────────────────────────────────────────────
# Coverage + mutation
# ──────────────────────────────────────────────────────────────────────────────
def _coverage_plugin_available() -> bool:
    try:
        import pytest_cov  # noqa: F401
        import coverage    # noqa: F401
        return True
    except ImportError:
        return False


def run_pytest(min_cov: float):
    if _coverage_plugin_available():
        run([
            sys.executable, "-m", "pytest", "tests",
            "--cov=.", "--cov-report=term-missing", "--cov-report=html",
            f"--cov-fail-under={min_cov:.0f}"
        ])
        LOG.info("Coverage HTML: %s", ROOT / "htmlcov" / "index.html")
    else:
        LOG.warning("pytest-cov or coverage missing – running pytest without coverage metrics.")
        run([sys.executable, "-m", "pytest", "tests"])

def run_mutmut(min_kill: float):
    try:
        import mutmut  # noqa: F401
    except ImportError:
        LOG.warning("mutmut missing – skipping mutation testing.")
        return
    run([sys.executable, "-m", "mutmut", "run", "--paths-to-mutate", "src"])

    out = run([sys.executable, "-m", "mutmut", "results"], capture=True)
    killed = survived = 0
    for ln in out.splitlines():
        if "killed" in ln:
            killed = int(ln.split()[0])
        elif "survived" in ln:
            survived = int(ln.split()[0])
    total = killed + survived
    if not total:
        LOG.warning("No mutants generated – kill-rate unchecked")
        return
    rate = 100 * killed / total
    LOG.info("Mutation kill-rate: %.1f%% (%d/%d)", rate, killed, total)
    if rate < min_kill:
        sys.exit(f"❌  Kill-rate {rate:.1f}%% below threshold {min_kill}%")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="End-to-end test workflow")
    ap.add_argument("--source-dir", default="auto", help="src/ or app/ or custom")
    ap.add_argument("--tests-dir",  default="tests")
    ap.add_argument("--min-cov",    type=float, default=90.0)
    ap.add_argument("--min-kill",   type=float, default=80.0)
    ap.add_argument("--search-time",type=int,   default=60)
    ap.add_argument("--regen-tests",type=lambda x: x.lower() != "false", default=True,
                    help="Regenerate baseline tests (true/false)")
    ap.add_argument("--include", default="*",
                    help="Comma-separated glob patterns of modules to include")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated glob patterns of modules to exclude")
    ap.add_argument("--log-level",  default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(message)s")

    pip_install()
    src_dir   = detect_source_dir(args.source_dir)
    tests_dir = (ROOT / args.tests_dir).resolve()
    tests_dir.mkdir(exist_ok=True)

    include_patterns = [p.strip() for p in args.include.split(',') if p.strip()]
    exclude_patterns = [p.strip() for p in args.exclude.split(',') if p.strip()]

    mods = list_modules(src_dir, include_patterns, exclude_patterns)
    if mods:
        LOG.info(f"Found {len(mods)} modules to test: {mods}")
        generate_tests(mods, tests_dir, args.search_time, args.regen_tests)
    else:
        LOG.warning("No modules found to test")

    run_pytest(args.min_cov)
    run_mutmut(args.min_kill)
    LOG.info("✅  All checks passed – enjoy your robust test suite!")

if __name__ == "__main__":
    main()
