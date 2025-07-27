#!/usr/bin/env python3
"""
Setup script for Self-RAG system
Checks prerequisites and initializes the system
"""
import subprocess
import sys
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

def check_ollama():
    """Check if Ollama is running and has the required model"""
    print("🔍 Checking Ollama...")
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            print(f"✅ Ollama is running")
            print(f"📦 Available models: {model_names}")
            
            if "llama3.2:1b" in model_names:
                print("✅ llama3.2:1b model is available")
                return True
            else:
                print("❌ llama3.2:1b model not found")
                print("📥 Please run: ollama pull llama3.2:1b")
                return False
        else:
            print("❌ Ollama is not responding properly")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("🚀 Please start Ollama: ollama serve")
        return False

def check_docker():
    """Check if Docker is running"""
    print("🔍 Checking Docker...")
    
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Docker is installed")
            
            # Check if Docker daemon is running
            result = subprocess.run(["docker", "ps"], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("❌ Docker daemon is not running")
                print("🚀 Please start Docker: sudo systemctl start docker")
                return False
        else:
            print("❌ Docker is not installed")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker is not available")
        return False

def check_pdfs():
    """Check if PDF files are available"""
    print("🔍 Checking PDF files...")
    
    pdf_dir = Path("../pdf_files")  # pdf_files directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"   📄 {pdf.name}")
        return True
    else:
        print("❌ No PDF files found in pdf_files directory")
        print("📁 Please ensure MEB PDF files are in the pdf_files/ directory")
        return False

def check_embedding_model():
    """Check if embedding model is available and download if needed"""
    print("🔍 Checking embedding model...")
    
    try:
        # Check if model is already cached
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        # HuggingFace cache location (used by SentenceTransformer)
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        model_cache_path = cache_dir / 'models--sentence-transformers--all-MiniLM-L6-v2'
        
        if model_cache_path.exists():
            print("✅ Embedding model already cached")
            # Quick verification that it loads
            try:
                model = SentenceTransformer(model_name)
                print("✅ Embedding model verified successfully")
                return True
            except Exception as e:
                print(f"⚠️ Cached model corrupted, re-downloading: {str(e)}")
        else:
            print("❌ Embedding model not found in cache")
        
        print("📥 Downloading sentence-transformers/all-MiniLM-L6-v2...")
        model = SentenceTransformer(model_name)
        print("✅ Embedding model downloaded and cached successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load/download embedding model: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 Self-RAG System Setup")
    print("=" * 50)
    
    checks = [
        ("Ollama & Model", check_ollama),
        ("Docker", check_docker),
        ("PDF Files", check_pdfs),
        ("Embedding Model", check_embedding_model),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)
        
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during {check_name}: {str(e)}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! System is ready.")
        print("\n🚀 Next steps:")
        print("1. Run the pipeline: python main.py")
        print("2. Or start web UI: chainlit run app_chainlit.py")
    else:
        print("⚠️ Some checks failed. Please fix the issues above.")
        print("\n📋 Prerequisites:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Pull model: ollama pull llama3.2:1b")
        print("3. Start Ollama: ollama serve")
        print("4. Install Docker and start the daemon")
        print("5. Place MEB PDF files in the pdf_files/ directory")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1) 