"""Example script demonstrating RAG functionality with LangChain and Milvus."""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.core.milvus_client import milvus_client
from src.services.rag_service import rag_service


async def main():
    """Main example function."""
    print("🚀 GenAI RAG Example with LangChain and Milvus")
    print("=" * 50)
    
    try:
        # Step 1: Connect to Milvus
        print("\n1. Connecting to Milvus...")
        await milvus_client.connect()
        print("✅ Connected to Milvus successfully!")
        
        # Step 2: Check collection stats
        print("\n2. Checking collection statistics...")
        stats = await milvus_client.get_collection_stats()
        print(f"📊 Collection: {stats.get('collection_name', 'unknown')}")
        print(f"📄 Total documents: {stats.get('total_entities', 0)}")
        
        # Step 3: Create sample document
        print("\n3. Creating sample document...")
        sample_text = """
        Artificial Intelligence (AI) is transforming industries worldwide.
        Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.
        Natural Language Processing (NLP) enables computers to understand and generate human language.
        Deep learning uses neural networks with multiple layers to solve complex problems.
        Computer vision allows machines to interpret and analyze visual information from images and videos.
        
        The applications of AI are diverse:
        - Healthcare: Diagnostic imaging, drug discovery, personalized medicine
        - Finance: Fraud detection, algorithmic trading, risk assessment
        - Transportation: Autonomous vehicles, route optimization
        - Education: Personalized learning, intelligent tutoring systems
        - Entertainment: Recommendation systems, content generation
        
        Challenges in AI include:
        1. Data privacy and security
        2. Algorithmic bias and fairness
        3. Explainability and transparency
        4. Job displacement concerns
        5. Regulatory and ethical considerations
        """
        
        # Save sample text to a temporary file
        sample_file = Path("sample_ai_document.txt")
        with open(sample_file, "w") as f:
            f.write(sample_text)
        
        print(f"📝 Created sample document: {sample_file}")
        
        # Step 4: Process and store document
        print("\n4. Processing and storing document...")
        document_id = await rag_service.process_and_store_document(
            file_path=str(sample_file),
            metadata={
                "title": "AI Overview",
                "author": "GenAI Boilerplate",
                "category": "technology"
            }
        )
        print(f"✅ Document processed and stored with ID: {document_id}")
        
        # Step 5: Search for similar content
        print("\n5. Searching for similar content...")
        search_query = "What are the applications of artificial intelligence?"
        search_results = await rag_service.search_documents(
            query=search_query,
            top_k=3,
            score_threshold=0.1
        )
        
        print(f"🔍 Search query: '{search_query}'")
        print(f"📋 Found {len(search_results)} relevant chunks:")
        
        for i, result in enumerate(search_results, 1):
            print(f"\n  Result {i}:")
            print(f"    Similarity: {result['similarity_score']:.3f}")
            print(f"    Text: {result['text'][:200]}...")
        
        # Step 6: Answer questions using RAG
        if settings.openai_api_key:
            print("\n6. Answering questions using RAG...")
            questions = [
                "What are the main applications of AI in healthcare?",
                "What challenges does AI face?",
                "How does machine learning work?"
            ]
            
            for question in questions:
                print(f"\n❓ Question: {question}")
                
                try:
                    response = await rag_service.answer_question(
                        question=question,
                        top_k=3,
                        ai_provider="openai",
                        ai_model="gpt-3.5-turbo",
                        score_threshold=0.1
                    )
                    
                    print(f"🤖 Answer: {response['answer']}")
                    print(f"📚 Sources used: {len(response['source_documents'])}")
                    
                except Exception as e:
                    print(f"❌ Error answering question: {e}")
                    
        else:
            print("\n⚠️  OpenAI API key not configured - skipping Q&A example")
            print("   Set OPENAI_API_KEY environment variable to try this feature")
        
        # Step 7: Collection statistics after processing
        print("\n7. Updated collection statistics...")
        updated_stats = await milvus_client.get_collection_stats()
        print(f"📊 Total documents after processing: {updated_stats.get('total_entities', 0)}")
        
        # Step 8: Demonstrate custom LangChain usage
        print("\n8. Custom LangChain example...")
        try:
            # Get custom retriever
            retriever = rag_service._get_retriever(top_k=5)
            
            # Use retriever directly
            relevant_docs = await retriever.aget_relevant_documents("machine learning algorithms")
            print(f"📖 Retrieved {len(relevant_docs)} documents using custom retriever")
            
            for i, doc in enumerate(relevant_docs[:2], 1):
                print(f"  Doc {i}: {doc.page_content[:150]}...")
                print(f"    Metadata: {doc.metadata}")
                
        except Exception as e:
            print(f"❌ Custom retriever example failed: {e}")
        
        print("\n✨ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during example execution: {e}")
        
    finally:
        # Cleanup
        try:
            if sample_file.exists():
                sample_file.unlink()
                print(f"🧹 Cleaned up sample file: {sample_file}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup sample file: {e}")
        
        # Disconnect from Milvus
        try:
            await milvus_client.disconnect()
            print("👋 Disconnected from Milvus")
        except Exception as e:
            print(f"⚠️  Error disconnecting: {e}")


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path(project_root) / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("📋 Please copy .env.example to .env and configure your settings")
        print("\nExample configuration:")
        print("  cp .env.example .env")
        print("  # Edit .env with your API keys and settings")
        sys.exit(1)
    
    # Run the example
    asyncio.run(main())
