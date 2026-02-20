# tests/test_education_extractor.py
"""
Test Education Extractor
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.education_extractor import EducationExtractor


def main():
    print("=" * 70)
    print("TESTING EDUCATION EXTRACTOR")
    print("=" * 70)
    
    # Test 1: Initialize
    print("\n1. Initializing education extractor...")
    extractor = EducationExtractor(use_nlp=True)
    print("✓ Extractor initialized")
    
    # Test 2: Test with various education formats
    print("\n2. Testing education extraction patterns...")
    
    test_cases = [
        {
            "text": "Bachelor of Science in Computer Science from Stanford University",
            "expected_degree": "Bachelor",
            "expected_field": "Computer Science"
        },
        {
            "text": "MS Computer Science, MIT, 2020",
            "expected_degree": "Master",
            "expected_field": "Computer Science"
        },
        {
            "text": "PhD in Machine Learning at Carnegie Mellon University",
            "expected_degree": "PhD",
            "expected_field": "Machine Learning"
        },
        {
            "text": "Master of Science (M.S.) in Electrical Engineering",
            "expected_degree": "Master",
            "expected_field": "Electrical Engineering"
        },
        {
            "text": "B.Tech in Information Technology from IIT Delhi",
            "expected_degree": "Bachelor",
            "expected_field": "Information Technology"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        result = extractor.extract_education(test["text"])
        print(f"\n   Test {i}:")
        print(f"   Input:    {test['text'][:60]}...")
        print(f"   Result:   {result}")
        
        # Check if degree and field are present
        if result:
            if test["expected_degree"] in result and test["expected_field"] in result:
                print(f"   ✓ Contains degree '{test['expected_degree']}' and field '{test['expected_field']}'")
            else:
                print(f"   ⚠ Missing expected degree or field")
                print(f"      Expected: {test['expected_degree']} in {test['expected_field']}")
        else:
            print(f"   ⚠ No education found")
    
    # Test 3: Test with full resume text
    print("\n3. Testing with sample resume text...")
    
    resume_text = """
    JOHN DOE
    Senior Data Scientist
    
    EDUCATION
    Master of Science in Computer Science
    Stanford University, 2018
    
    Bachelor of Science in Mathematics
    MIT, 2016
    
    EXPERIENCE
    Senior Data Scientist at Google (2020-present)
    """
    
    education = extractor.extract_education(resume_text)
    print(f"   Extracted: {education}")
    
    if "Master" in education and "Computer Science" in education:
        print(f"   ✓ Correctly extracted highest degree")
    else:
        print(f"   ⚠ Unexpected result")
    
    # Test 4: Extract all degrees
    print("\n4. Extracting all degrees (not just highest)...")
    all_degrees = extractor.extract_all_degrees(resume_text)
    print(f"   Found {len(all_degrees)} degrees:")
    for degree in all_degrees:
        print(f"     - {degree}")
    
    if len(all_degrees) >= 2:
        print(f"   ✓ Found multiple degrees")
    else:
        print(f"   ⚠ Expected to find 2 degrees")
    
    # Test 5: Test edge cases
    print("\n5. Testing edge cases...")
    
    edge_cases = [
        {
            "text": "No education mentioned",
            "expected_empty": True
        },
        {
            "text": "I studied at university",
            "expected_empty": True
        },
        {
            "text": "BS CS MIT",
            "should_contain": "Computer Science"
        },
        {
            "text": "Ph.D. in Artificial Intelligence",
            "should_contain": "PhD"
        }
    ]
    
    for i, test in enumerate(edge_cases, 1):
        result = extractor.extract_education(test["text"])
        print(f"\n   Edge case {i}:")
        print(f"   Input:  '{test['text']}'")
        print(f"   Result: '{result}'")
        
        if test.get("expected_empty"):
            if not result:
                print(f"   ✓ Correctly empty")
            else:
                print(f"   ⚠ Expected empty, got: {result}")
        elif test.get("should_contain"):
            if test["should_contain"] in result:
                print(f"   ✓ Contains expected: '{test['should_contain']}'")
            else:
                print(f"   ⚠ Missing expected: '{test['should_contain']}'")
    
    # Test 6: Test with your actual resumes
    print("\n6. Testing with actual resume files...")
    from pathlib import Path
    
    parsed_dir = Path("data/resumes/parsed")
    if parsed_dir.exists():
        parsed_files = list(parsed_dir.glob("*_extracted.txt"))
        
        if parsed_files:
            print(f"   Found {len(parsed_files)} parsed resume files")
            
            for parsed_file in parsed_files[:3]:  # Test first 3
                try:
                    with open(parsed_file, 'r', encoding='utf-8') as f:
                        resume_content = f.read()
                    
                    education = extractor.extract_education(resume_content)
                    print(f"\n   File: {parsed_file.name}")
                    print(f"   Education: {education if education else '(none found)'}")
                    
                except Exception as e:
                    print(f"   ✗ Error reading {parsed_file.name}: {e}")
        else:
            print(f"   ⚠ No parsed resume files found")
    else:
        print(f"   ⚠ Parsed directory not found: {parsed_dir}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    
    # Summary
    print("\nSUMMARY:")
    print("- If most tests pass: ✓ Ready to proceed")
    print("- If patterns fail: Check regex patterns in _build_degree_patterns()")
    print("- If institutions missing: Check spaCy NER and known_institutions list")


if __name__ == "__main__":
    main()
