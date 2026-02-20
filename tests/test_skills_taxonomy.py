# tests/test_skills_taxonomy.py
"""
Test Skills Taxonomy
"""

import sys
import os

# Fix imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.skills_taxonomy import SkillsTaxonomy

print("=" * 70)
print("TESTING SKILLS TAXONOMY")
print("=" * 70)

# Test 1: Initialize taxonomy
print("\n1. Initializing taxonomy...")
taxonomy = SkillsTaxonomy()
print("✓ Taxonomy initialized")

# Test 2: Check statistics
print("\n2. Taxonomy statistics:")
stats = taxonomy.get_statistics()
for category, count in stats.items():
    print(f"   {category:25s}: {count:3d}")

# Test 3: Test normalization
print("\n3. Testing normalization:")
test_cases = [
    ("Python", "python"),
    ("JAVA", "java"),
    ("K8S", "kubernetes"),
    ("TensorFlow", "tensorflow"),
    ("  AWS  ", "aws"),
    ("Python 3.11", "python")
]
for input_skill, expected in test_cases:
    result = taxonomy.normalize_skill(input_skill)
    status = "✓" if result == expected else "✗"
    print(f"   {status} '{input_skill}' → '{result}' (expected: '{expected}')")

# Test 4: Extract skills from sample text
print("\n4. Extracting skills from sample resume:")
sample_text = """
Senior Data Scientist with 5 years of experience in Python, TensorFlow,
and AWS. Proficient in machine learning, deep learning, and NLP.
Strong leadership and communication skills. Experienced with Docker,
Kubernetes, PostgreSQL, and Git. Built REST APIs using Flask and FastAPI.
"""
hard_skills, soft_skills = taxonomy.extract_skills_from_text(sample_text)

print(f"\n   Found {len(hard_skills)} hard skills:")
for skill in hard_skills:
    category = taxonomy.get_skill_category(skill)
    print(f"      - {skill:20s} [{category}]")

print(f"\n   Found {len(soft_skills)} soft skills:")
for skill in soft_skills:
    print(f"      - {skill}")

# Test 5: Category lookup
print("\n5. Testing category lookup:")
test_skills = ["python", "tensorflow", "aws", "postgresql", "leadership"]
for skill in test_skills:
    category = taxonomy.get_skill_category(skill)
    print(f"   '{skill}' → {category}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
