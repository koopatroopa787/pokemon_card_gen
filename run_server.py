#!/usr/bin/env python3
"""
Convenience script to run the Pokemon Card Generator API server
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Starting Pokemon Card Generator API Server...")
    print(f"Project root: {project_root}")
    print("\nServer will be available at:")
    print("  - API: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Web UI: http://localhost:8000/ui")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
