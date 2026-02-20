# tests/test_pdf_extractor.py
"""
Test PDF Extractor
"""

import sys
import os
import re

# Fix imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.pdf_extractor import PDFExtractor
from pathlib import Path

print("=" * 70)
print("TESTING PDF EXTRACTOR")
print("=" * 70)

# Test 1: Initialize extractor
print("\n1. Initializing PDF extractor...")
extractor = PDFExtractor()
print("✓ Extractor initialized")

# Test 2: Check if resume PDFs exist
print("\n2. Checking for resume PDFs...")
resumes_dir = Path("data/resumes")
pdf_files = list(resumes_dir.glob("*.pdf"))

if pdf_files:
    print(f"✓ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
else:
    print("⚠️  No PDF files found in data/resumes/")
    print("   Please add resume PDFs to continue testing")
    exit(0)

# Test 3: Extract text from first PDF
print("\n3. Extracting text from first PDF...")
first_pdf = pdf_files[0]
try:
    text = extractor.extract_text(str(first_pdf))
    print(f"✓ Extracted text from {first_pdf.name}")
    print(f"   Text length: {len(text)} characters")
    
    # Show first 300 characters
    print(f"\n   First 300 characters:")
    print("   " + "-" * 60)
    preview = text[:300].replace('\n', '\n   ')
    print(f"   {preview}")
    print("   " + "-" * 60)
    
except Exception as e:
    print(f"✗ Extraction failed: {e}")
    exit(1)

# Test 4: Validate extraction
print("\n4. Validating extraction...")
is_valid, message = extractor.validate_extraction(text)
if is_valid:
    print(f"✓ {message}")
else:
    print(f"✗ {message}")

# Test 5: Get statistics
print("\n5. Text statistics:")
stats = extractor.get_text_statistics(text)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"   {key:25s}: {value:.2f}")
    else:
        print(f"   {key:25s}: {value}")

# Test 6: Extract from all PDFs
print("\n6. Extracting from all PDFs in directory...")
results = extractor.extract_from_directory("data/resumes")
print(f"\n   Successfully extracted {len(results)} files")

# Test 7: Check for common issues
print("\n7. Checking for common extraction issues...")
issues_found = []

# Check for excessive whitespace
if '    ' in text:  # 4+ spaces
    issues_found.append("Excessive whitespace found")

# Check for missing newlines (wall of text)
lines = text.split('\n')
avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
if avg_line_length > 200:
    issues_found.append(f"Very long average line length ({avg_line_length:.0f} chars)")

# Check for page numbers
if re.search(r'Page \d+', text):
    issues_found.append("Page numbers detected (should be cleaned)")

if issues_found:
    print("   ⚠️  Potential issues:")
    for issue in issues_found:
        print(f"      - {issue}")
else:
    print("   ✓ No common issues detected")

# Test 8: Test error handling
print("\n8. Testing error handling...")
try:
    extractor.extract_text("nonexistent_file.pdf")
    print("   ✗ Should have raised FileNotFoundError")
except FileNotFoundError:
    print("   ✓ FileNotFoundError handled correctly")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)

# Save extracted text for inspection
print("\n9. Saving extracted text for inspection...")
output_dir = Path("data/resumes/parsed")
output_dir.mkdir(parents=True, exist_ok=True)

for filename, text_content in results.items():
    if text_content:
        output_file = output_dir / f"{Path(filename).stem}_extracted.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"   ✓ Saved: {output_file.name}")
