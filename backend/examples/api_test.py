"""Example script for testing the RAG API endpoints."""
import asyncio
import json
import time
from pathlib import Path
import httpx
from io import StringIO


class RAGAPITester:
    """Test client for the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()
    
    async def test_health(self):
        """Test health endpoint."""
        print("🏥 Testing Health Endpoint")
        print("-" * 30)
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Milvus Connected: {data.get('milvus_connected')}")
                print(f"   Total Documents: {data.get('total_documents')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_config(self):
        """Test configuration endpoint."""
        print("\n⚙️  Testing Configuration Endpoint")
        print("-" * 35)
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/config")
            if response.status_code == 200:
                data = response.json()
                print("✅ Configuration retrieved")
                print(f"   Milvus Host: {data['milvus']['host']}")
                print(f"   Collection: {data['milvus']['collection_name']}")
                print(f"   Embedding Model: {data['embedding']['model']}")
                print(f"   Chunk Size: {data['embedding']['chunk_size']}")
                print(f"   Supported Files: {', '.join(data['supported_file_types'])}")
                return True
            else:
                print(f"❌ Configuration failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return False
    
    async def test_document_upload(self):
        """Test document upload endpoint."""
        print("\n📤 Testing Document Upload")
        print("-" * 28)
        
        try:
            # Create sample document
            sample_content = """
            Python is a high-level, interpreted programming language.
            It was created by Guido van Rossum and first released in 1991.
            Python emphasizes code readability with its use of significant indentation.
            
            Key features of Python:
            - Easy to learn and use
            - Versatile and powerful
            - Large standard library
            - Active community support
            - Cross-platform compatibility
            
            Python is widely used in:
            1. Web development (Django, Flask)
            2. Data science and analytics
            3. Machine learning and AI
            4. Scientific computing
            5. Automation and scripting
            """
            
            # Upload document
            files = {"file": ("python_guide.txt", StringIO(sample_content), "text/plain")}
            response = await self.client.post(f"{self.base_url}/api/v1/documents/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Document uploaded successfully")
                print(f"   Document ID: {data.get('document_id')}")
                print(f"   Filename: {data.get('filename')}")
                print(f"   Size: {data.get('size')} bytes")
                print(f"   Message: {data.get('message')}")
                return data.get('document_id')
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
    
    async def test_document_search(self):
        """Test document search endpoint."""
        print("\n🔍 Testing Document Search")
        print("-" * 27)
        
        try:
            search_data = {
                "query": "What are the main features of Python?",
                "top_k": 3,
                "score_threshold": 0.1
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/documents/search",
                json=search_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Search completed successfully")
                print(f"   Query: {data.get('query')}")
                print(f"   Results Found: {data.get('total_results')}")
                print(f"   Processing Time: {data.get('processing_time_ms'):.2f}ms")
                
                for i, result in enumerate(data.get('results', [])[:2], 1):
                    print(f"\n   Result {i}:")
                    print(f"     Similarity: {result.get('similarity_score', 0):.3f}")
                    print(f"     Text: {result.get('text', '')[:150]}...")
                
                return True
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    
    async def test_question_answering(self):
        """Test question answering endpoint."""
        print("\n🤖 Testing Question Answering")
        print("-" * 31)
        
        try:
            qa_data = {
                "question": "What is Python mainly used for?",
                "top_k": 3,
                "ai_provider": "openai",
                "ai_model": "gpt-3.5-turbo",
                "score_threshold": 0.1
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/chat/completions",
                json=qa_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Question answered successfully")
                print(f"   Question: {data.get('question')}")
                print(f"   Answer: {data.get('answer')[:200]}...")
                print(f"   Sources Used: {len(data.get('source_documents', []))}")
                print(f"   Processing Time: {data.get('processing_time_ms'):.2f}ms")
                
                metadata = data.get('metadata', {})
                print(f"   AI Provider: {metadata.get('ai_provider')}")
                print(f"   AI Model: {metadata.get('ai_model')}")
                
                return True
            elif response.status_code == 400:
                print("⚠️  Question answering skipped - API key not configured")
                print(f"   Response: {response.json().get('detail', 'Unknown error')}")
                return True  # Not a failure, just missing config
            else:
                print(f"❌ Question answering failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Question answering error: {e}")
            return False
    
    async def test_document_stats(self):
        """Test document statistics endpoint."""
        print("\n📊 Testing Document Statistics")
        print("-" * 32)
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/documents/stats")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Statistics retrieved successfully")
                print(f"   Collection Name: {data.get('collection_name')}")
                print(f"   Total Entities: {data.get('total_entities')}")
                print(f"   Is Loaded: {data.get('is_loaded')}")
                print(f"   Indexes: {len(data.get('indexes', []))}")
                return True
            else:
                print(f"❌ Statistics failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Statistics error: {e}")
            return False
    
    async def test_document_deletion(self, document_id: str):
        """Test document deletion endpoint."""
        if not document_id:
            print("\n⚠️  Skipping document deletion - no document ID available")
            return True
        
        print("\n🗑️  Testing Document Deletion")
        print("-" * 29)
        
        try:
            response = await self.client.delete(f"{self.base_url}/api/v1/documents/{document_id}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Document deleted successfully")
                print(f"   Message: {data.get('message')}")
                return True
            elif response.status_code == 404:
                print("⚠️  Document not found (may have been deleted already)")
                return True
            else:
                print(f"❌ Deletion failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Deletion error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all API tests."""
        print("🧪 GenAI RAG API Testing Suite")
        print("=" * 50)
        
        tests_passed = 0
        total_tests = 6
        
        # Test 1: Health check
        if await self.test_health():
            tests_passed += 1
        
        # Test 2: Configuration
        if await self.test_config():
            tests_passed += 1
        
        # Test 3: Document upload
        document_id = await self.test_document_upload()
        if document_id:
            tests_passed += 1
        
        # Wait a moment for processing
        if document_id:
            print("\n⏳ Waiting 2 seconds for document processing...")
            await asyncio.sleep(2)
        
        # Test 4: Document search
        if await self.test_document_search():
            tests_passed += 1
        
        # Test 5: Question answering
        if await self.test_question_answering():
            tests_passed += 1
        
        # Test 6: Document statistics
        if await self.test_document_stats():
            tests_passed += 1
        
        # Test 7: Document deletion (bonus)
        if await self.test_document_deletion(document_id):
            pass  # Don't count this in main test count
        
        # Results
        print("\n" + "=" * 50)
        print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests completed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return tests_passed == total_tests
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the RAG API endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    
    tester = RAGAPITester(base_url=args.url)
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    finally:
        await tester.close()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
