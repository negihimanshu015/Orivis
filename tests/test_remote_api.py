import requests
import time
import sys

# Default to your Hugging Face Space URL
API_BASE = "https://phenominal-orivis.hf.space"
TEST_VIDEO = "Test/Fake.mp4"

def test_api(url=API_BASE, video_path=TEST_VIDEO):
    print(f"🚀 Testing Orivis API at: {url}")
    print(f"📁 Video: {video_path}")
    
    try:
        # 1. Submit Job
        with open(video_path, 'rb') as f:
            files = {'video': f}
            res = requests.post(f"{url}/detect", files=files)
            
        if res.status_code != 200:
            print(f"❌ Error submitting job: {res.text}")
            return
            
        job_id = res.json().get("job_id")
        print(f"✅ Job submitted! ID: {job_id}")
        
        # 2. Poll Status
        status = "queued"
        while status in ["queued", "processing"]:
            print(f"⌛ Status: {status}...")
            time.sleep(2)
            res = requests.get(f"{url}/job/{job_id}")
            data = res.json()
            status = data.get("status")
            
        if status == "completed":
            results = data.get("results", {})
            label = results.get("label", "Unknown")
            prob = results.get("final_synthetic_probability", 0.0)
            print("-" * 30)
            print(f"🏆 ANALYSIS COMPLETE")
            print(f"LABEL: {label.upper()}")
            print(f"CONFIDENCE: {prob*100:.1f}%")
            print("-" * 30)
        else:
            print(f"❌ Job failed: {data.get('error')}")
            
    except Exception as e:
        print(f"❌ Request error: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else API_BASE
    test_api(url)
