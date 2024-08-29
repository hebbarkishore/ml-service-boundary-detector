#!/usr/bin/env python3
import subprocess
import sys
import platform
import argparse

def run(cmd, description):
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  WARNING: '{description}' failed (returncode={result.returncode})")
    return result.returncode == 0

def pip(*packages, upgrade=False):
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    return run(cmd, f"Installing: {', '.join(packages)}")

def detect_platform():
    os_name  = platform.system()         
    machine  = platform.machine()        
    py_ver   = sys.version_info[:2]      

    print(f"\n{'═'*60}")
    print(f"  Service Boundary Detector — Cross-Platform Installer")
    print(f"{'═'*60}")
    print(f"  OS       : {os_name} ({machine})")
    print(f"  Python   : {py_ver[0]}.{py_ver[1]}")
    print(f"  Platform : ", end="")

    if os_name == "Darwin" and machine == "arm64":
        platform_id = "mac_arm"
        print("Apple Silicon Mac")
    elif os_name == "Darwin":
        platform_id = "mac_intel"
        print("Intel Mac")
    elif os_name == "Windows":
        platform_id = "windows"
        print("Windows")
    else:
        platform_id = "linux"
        print("Linux")

    if py_ver < (3, 10):
        print(f"\n  Python {py_ver[0]}.{py_ver[1]} is too old. Please use Python 3.10+.")
        sys.exit(1)
    if py_ver > (3, 12):
        print(f"  Python {py_ver[0]}.{py_ver[1]} is newer than tested. "
              "Some packages may have minor issues.")

    return platform_id, py_ver

def install_core(platform_id):
    """Install packages that work on all platforms without C compilation issues."""
    pip("pip", "setuptools", "wheel", upgrade=True)

    pip("numpy>=1.24,<2.0")
    pip("scipy>=1.14")
    pip("scikit-learn>=1.5")
    pip("networkx>=3.3")

    pip(
        "click>=8.0",
        "flask>=3.0",
        "flask-restful>=0.3.10",
        "werkzeug>=3.0",
        "tqdm>=4.62",
        "PyYAML>=6.0",
        "loguru>=0.5",
        "joblib>=1.1",
        "gitpython>=3.1",
    )

def install_pandas(platform_id, py_ver):
    pip("pandas>=2.2")

def install_nlp(platform_id):
    """spaCy + gensim for semantic similarity."""
    pip("spacy>=3.7,<4.0")
    pip("gensim>=4.3,<5.0")

    run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_md"],
        "Downloading en_core_web_md"
    )

def install_hdbscan(platform_id):
    """hdbscan needs special handling — C extension, no universal wheels."""

    if platform_id == "mac_arm":
        success = pip("hdbscan")
        if not success:
            print("  ℹ  hdbscan pip install failed on Apple Silicon.")
            print("     If you use conda/miniforge, run:")
            print("     conda install -c conda-forge hdbscan")
            print("     Clustering suggestions will be disabled otherwise.")
    elif platform_id == "windows":
        run(
            [sys.executable, "-m", "pip", "install", "hdbscan", "--only-binary=:all:"],
            "Installing hdbscan (binary only, no compiler needed)"
        )
    else:
        pip("hdbscan")

def install_imbalanced(platform_id):
    pip("imbalanced-learn>=0.12")

def install_java_parsing():
    pip("javalang>=0.13.0")
    pip("lizard>=1.17")

def install_doc_ingestion():
    pip("pdfminer.six>=20231228")
    pip("python-docx>=1.1")
    pip("Markdown>=3.6")
    pip("beautifulsoup4>=4.12")

def verify_install():
    print(f"\n{'═'*60}")
    print("  Verifying installation …")
    print(f"{'═'*60}")

    checks = [
        ("numpy",         "import numpy; print(numpy.__version__)"),
        ("scipy",         "import scipy; print(scipy.__version__)"),
        ("scikit-learn",  "import sklearn; print(sklearn.__version__)"),
        ("networkx",      "import networkx; print(networkx.__version__)"),
        ("flask",         "import flask; print(flask.__version__)"),
        ("gitpython",     "import git; print(git.__version__)"),
        ("click",         "import click; print(click.__version__)"),
        ("pandas",        "import pandas; print(pandas.__version__)"),
        ("spacy",         "import spacy; print(spacy.__version__)"),
        ("gensim",        "import gensim; print(gensim.__version__)"),
        ("hdbscan",       "import hdbscan; print(hdbscan.__version__)"),
        ("javalang",      "import javalang; print('ok')"),
        ("pdfminer",      "import pdfminer; print('ok')"),
        ("docx",          "import docx; print('ok')"),
    ]

    results = []
    for name, code in checks:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True
        )
        status  = "Success" if result.returncode == 0 else "Failed"
        version = result.stdout.strip() if result.returncode == 0 else "not installed"
        print(f"  {status}  {name:<20} {version}")
        results.append(result.returncode == 0)

    core_ok = all(results[:7])  
    print(f"\n  Core dependencies : {'ALL OK' if core_ok else 'SOME FAILED'}")
    print(f"  Optional extras   : {sum(results[7:])} / {len(results)-7} installed")
    return core_ok

def main():
    parser = argparse.ArgumentParser(description="SBD cross-platform installer")
    parser.add_argument("--minimal", action="store_true",
                        help="Core only – no NLP/docs/Java extras")
    parser.add_argument("--all", action="store_true",
                        help="Install all optional extras")
    args = parser.parse_args()

    platform_id, py_ver = detect_platform()

    install_core(platform_id)
    install_pandas(platform_id, py_ver)

    if not args.minimal:
        install_nlp(platform_id)
        install_hdbscan(platform_id)
        install_imbalanced(platform_id)

    if args.all:
        install_java_parsing()
        install_doc_ingestion()

    ok = verify_install()

    print(f"\n{'═'*60}")
    if ok:
        print("  Next steps:")
        print("    1. python cli.py analyze --code /path/to/your/src")
        print("    2. python cli.py serve          # start the REST API")
    else:
        print("   Some CORE dependencies failed. Check errors above.")
        print("     Try running:  pip install --upgrade pip setuptools wheel")
        print("     Then re-run:  python install.py")
    print(f"{'═'*60}\n")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
