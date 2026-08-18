"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import sys
import subprocess


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest module-by-module for responsive feedback and robustness."""
    import re
    test_files = [
        ("M1 Chunking", "tests/test_m1.py"),
        ("M2 Search", "tests/test_m2.py"),
        ("M3 Rerank", "tests/test_m3.py"),
        ("M4 Eval", "tests/test_m4.py"),
        ("M5 Enrichment", "tests/test_m5.py"),
    ]
    total_passed = 0
    total_count = 0
    for name, test_path in test_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-q", "--tb=no"],
                capture_output=True, text=True, timeout=180,
            )
            summary = result.stdout.strip()
            m_pass = re.search(r"(\d+)\s+passed", summary)
            m_fail = re.search(r"(\d+)\s+failed", summary)
            passed = int(m_pass.group(1)) if m_pass else 0
            failed = int(m_fail.group(1)) if m_fail else 0
            file_total = passed + failed
            total_passed += passed
            total_count += file_total
            if failed == 0 and passed > 0:
                print(f"  ✅ {name:<15}: {passed}/{file_total} passed")
            else:
                print(f"  ⚠️  {name:<15}: {passed}/{file_total} passed ({failed} failed)")
        except Exception as e:
            print(f"  ⚠️  {name:<15}: error {e}")
    return total_passed, total_count


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    check_file("analysis/failure_analysis.md")
    check_file("analysis/group_report.md")

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections = [f for f in os.listdir(ref_dir) if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"]
    if reflections:
        for r in reflections:
            print(f"  ✅ {ref_dir}/{r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân trong {ref_dir}/")

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct >= 80 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
    else:
        print("  ⚠️  Không chạy được tests")

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)


if __name__ == "__main__":
    validate()
