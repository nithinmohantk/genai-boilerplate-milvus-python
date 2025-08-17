#!/bin/bash

# GenAI RAG Boilerplate Setup Script
# This script helps you quickly set up the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_status "Please install the missing dependencies and run this script again."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.10"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python $required_version or higher is required (found $python_version)"
        exit 1
    fi
    
    print_success "All system requirements met!"
}

# Function to setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f "backend/.env" ]; then
        cp backend/.env.example backend/.env
        print_success "Created backend/.env from template"
        
        print_warning "Please edit backend/.env with your configuration:"
        echo "  - Set SECRET_KEY to a secure random string"
        echo "  - Add your AI provider API keys (OpenAI, Anthropic, Google)"
        echo "  - Adjust Milvus and embedding settings if needed"
        
        read -p "Would you like to open the .env file for editing now? [y/N]: " edit_env
        if [[ $edit_env =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} backend/.env
        fi
    else
        print_warning "backend/.env already exists, skipping creation"
    fi
}

# Function to create virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    cd backend
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python dependencies installed successfully"
    
    cd ..
}

# Function to start services with Docker
start_services() {
    print_status "Starting services with Docker Compose..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Start services
    docker-compose up --build -d
    
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be healthy
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/api/v1/health >/dev/null 2>&1; then
            print_success "All services are ready!"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Services failed to start within expected time"
        print_status "Check service status with: docker-compose ps"
        print_status "Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to run example
run_example() {
    print_status "Running example to test the setup..."
    
    cd backend
    source venv/bin/activate
    
    # Run API test
    if python examples/api_test.py --url http://localhost:8000; then
        print_success "API test completed successfully!"
    else
        print_warning "API test had some issues, but this might be due to missing API keys"
    fi
    
    cd ..
}

# Function to display next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. 📝 Edit backend/.env with your API keys"
    echo "2. 🌐 Open http://localhost:8000/docs to explore the API"
    echo "3. 🔍 Open http://localhost:3001 for Milvus admin UI (Attu)"
    echo "4. 📊 Check service status: docker-compose ps"
    echo ""
    echo "Available services:"
    echo "- RAG API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/docs"
    echo "- Milvus Admin UI: http://localhost:3001"
    echo "- MinIO Console: http://localhost:9001"
    echo ""
    echo "To test the API:"
    echo "  cd backend && source venv/bin/activate"
    echo "  python examples/api_test.py"
    echo ""
    echo "To run a comprehensive example:"
    echo "  cd backend && source venv/bin/activate"
    echo "  python examples/rag_example.py"
    echo ""
    echo "To stop services:"
    echo "  docker-compose down"
    echo ""
}

# Main setup function
main() {
    echo "🚀 GenAI RAG Boilerplate Setup"
    echo "================================"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ] || [ ! -d "backend" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_requirements
    setup_environment
    setup_python_env
    start_services
    run_example
    show_next_steps
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "check")
        check_requirements
        ;;
    "env")
        setup_environment
        ;;
    "python")
        setup_python_env
        ;;
    "start")
        start_services
        ;;
    "test")
        run_example
        ;;
    "help"|"-h"|"--help")
        echo "GenAI RAG Boilerplate Setup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup    Run full setup (default)"
        echo "  check    Check system requirements only"
        echo "  env      Setup environment files only"
        echo "  python   Setup Python environment only"
        echo "  start    Start Docker services only"
        echo "  test     Run API tests only"
        echo "  help     Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Run '$0 help' for usage information"
        exit 1
        ;;
esac
