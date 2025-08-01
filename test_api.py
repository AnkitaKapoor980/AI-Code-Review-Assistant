import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Sample bad code for testing
SAMPLE_CODE = """
def process_user_data(user_input):
    # Security issue: using eval
    result = eval(user_input)
    
    # Performance issue: inefficient loop
    processed_data = []
    for i in range(1000):
        processed_data.append(i * 2)
    
    # Style issue: line too long
    very_long_variable_name_that_exceeds_the_recommended_line_length_limit = "This line is way too long"
    
    # Best practice issue: bare except
    try:
        risky_operation = 10 / 0
    except:
        pass
    
    return result, processed_data
"""

def test_basic_analysis():
    print("🧪 Testing basic analysis...")
    response = requests.post(f"{BASE_URL}/analyze", 
                           json={"text": SAMPLE_CODE})
    print("✅ Analysis Result:", json.dumps(response.json(), indent=2))

def test_review_endpoint():
    print("🧪 Testing code review...")
    response = requests.post(f"{BASE_URL}/review", 
                           json={
                               "code": SAMPLE_CODE,
                               "language": "python"
                           })
    result = response.json()
    print(f"✅ Review completed! Score: {result['overall_score']}/100")
    print(f"✅ Issues found: {len(result['issues'])}")

def test_quick_scan():
    print("🧪 Testing quick scan...")
    response = requests.post(f"{BASE_URL}/quick-scan", 
                           json={
                               "code": SAMPLE_CODE,
                               "language": "python"
                           })
    result = response.json()
    print(f"✅ Quick scan: {result['status']} (Score: {result['quality_score']})")

if __name__ == "__main__":
    print("🚀 Starting API tests...")
    print("⏳ Make sure the API is running on http://localhost:8000")
    time.sleep(2)
    
    try:
        test_review_endpoint()
        test_quick_scan()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to run 'python main.py' first!")