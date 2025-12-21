import requests
import json

BASE_URL = "http://localhost:8000"

def test_query(name, query, context=None):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    payload = {"query": query}
    if context:
        payload["context"] = context
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: SUCCESS")
            print(f"Intent: {result['intent']}")
            print(f"Model: {result['used_model']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Answer: {result['answer'][:200]}...")
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

# Run tests
if __name__ == "__main__":
    # Test 1: Pesticide (Known to work)
    test_query(
        "Pesticide Recommendation",
        "which pesticide for aphids"
    )
    
    # Test 2: Crop Production
    test_query(
        "Crop Production",
        "rice production in Punjab"
    )
    
    # Test 3: Crop Calendar
    test_query(
        "Crop Calendar",
        "when to sow paddy in Bihar"
    )
    
    # Test 4: Pest Management
    test_query(
        "Pest Management",
        "how to control stem borer in rice"
    )
    
    # Test 5: Fertilizer with features
    test_query(
        "Fertilizer Recommendation",
        "recommend fertilizer",
        context={
            "crop": "wheat",
            "features": {
                "N": 20, "P": 15, "K": 25,
                "Temperature": 28, "Humidity": 65, "Moisture": 45
            }
        }
    )
    
    # Test 6: Yield with features
    test_query(
        "Yield Prediction",
        "predict crop yield",
        context={
            "crop": "Rice",
            "features": {
                "N": 25, "P": 20, "K": 30,
                "Temperature": 28.5, "Humidity": 70, "Moisture": 45,
                "Crop": "Rice"
            }
        }
    )
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)