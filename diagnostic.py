import sys
import os

print("=" * 70)
print("DIAGNOSTIC INFORMATION")
print("=" * 70)

print("\n1. Current Working Directory:")
print(f"   {os.getcwd()}")

print("\n2. Project Root Check:")
if os.path.exists("src") and os.path.exists("tests"):
    print("   ✓ Correct: tests/ is at root level")
elif os.path.exists("src/tests"):
    print("   ✗ WRONG: tests/ is inside src/")
    print("   Fix: mv src/tests ./tests")
else:
    print("   ? Tests folder not found")

print("\n3. __init__.py Files:")
print(f"   src/__init__.py exists: {os.path.exists('src/__init__.py')}")
print(f"   src/modules/__init__.py exists: {os.path.exists('src/modules/__init__.py')}")

print("\n4. Resume Parser File:")
print(f"   src/modules/resume_parser.py exists: {os.path.exists('src/modules/resume_parser.py')}")

print("\n5. Python Path:")
for i, path in enumerate(sys.path[:5]):
    print(f"   {i}: {path}")

print("\n6. Import Test:")
try:
    from src.modules.resume_parser import ResumeProfile
    print("   ✓ Import successful!")
except ModuleNotFoundError as e:
    print(f"   ✗ Import failed: {e}")

print("\n" + "=" * 70)
