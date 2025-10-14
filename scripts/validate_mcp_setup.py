#!/usr/bin/env python3
"""
MCP Setup Validation Script

Tests that the movie recommendation MCP server can be installed and configured correctly.
Run this script to verify your setup before configuring Claude Desktop/Code.
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def run_command(cmd, description, capture_output=True):
    """Run a command and handle errors gracefully."""
    print(f"🔍 {description}...")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                return result.stdout.strip()
            else:
                print(f"❌ {description} - FAILED")
                print(f"   Error: {result.stderr.strip()}")
                return None
        else:
            result = subprocess.run(cmd, shell=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                return True
            else:
                print(f"❌ {description} - FAILED")
                return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return None
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return None

def check_python_version():
    """Check Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (need 3.11+)")
        return False

def check_uv_installation():
    """Check if uv is installed and working."""
    uv_path = run_command("which uv", "Checking uv installation")
    if uv_path:
        version = run_command("uv --version", "Checking uv version")
        if version:
            print(f"   uv path: {uv_path}")
            print(f"   uv version: {version}")
            return uv_path
    return None

def check_project_setup():
    """Check if project dependencies are installed."""
    # Check if pyproject.toml exists
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found - are you in the project directory?")
        return False
    
    # Check if .venv exists or can create one
    result = run_command("uv sync --dry-run", "Checking project dependencies")
    return result is not None

def test_mcp_server_import():
    """Test that MCP server can be imported."""
    result = run_command("uv run python -c 'import src.mcp_server; print(\"Import successful\")'", 
                        "Testing MCP server import")
    return result is not None

def test_mcp_server_startup():
    """Test that MCP server can start (briefly)."""
    print("🔍 Testing MCP server startup...")
    try:
        # Start server and kill after 3 seconds
        result = subprocess.run("timeout 3 uv run python -m src.mcp_server", 
                               shell=True, capture_output=True, text=True)
        # timeout command returns 124 when timing out, which is expected
        if result.returncode in [0, 124]:
            print("✅ MCP server startup - SUCCESS")
            return True
        else:
            print("❌ MCP server startup - FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ MCP server startup - ERROR: {e}")
        return False

def test_mcp_tools():
    """Test MCP tools using mcp-client-for-testing if available."""
    # Check if mcp-client-for-testing is available
    mcp_client = run_command("which mcp-client-for-testing", "Checking MCP testing client")
    if not mcp_client:
        print("⚠️  mcp-client-for-testing not found (optional)")
        print("   Install with: uv tool install mcp-client-for-testing")
        return True  # Not critical
    
    # Test a simple tool call
    config = json.dumps([{
        "name": "movie-recommender",
        "command": "uv",
        "args": ["--directory", ".", "run", "python", "-m", "src.mcp_server"]
    }])
    
    tool_call = json.dumps({
        "name": "search_movies",
        "arguments": {"query": "matrix", "limit": 2}
    })
    
    cmd = f'mcp-client-for-testing --config \'{config}\' --client_log_level WARNING --server_log_level WARNING --tool_call \'{tool_call}\''
    
    result = run_command(cmd, "Testing MCP tool call")
    if result and "movieId" in result:
        print("   Found movie data in response")
        return True
    return False

def generate_config_examples():
    """Generate configuration examples for the user."""
    print("\n📝 Configuration Examples:")
    
    # Get current directory
    current_dir = Path.cwd().absolute()
    
    # Get uv path
    uv_path = run_command("which uv", "Getting uv path")
    if not uv_path:
        uv_path = "/usr/local/bin/uv"
    
    print(f"\n🖥️  Claude Desktop Configuration:")
    print(f"File: ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)")
    print(f"File: %APPDATA%\\Claude\\claude_desktop_config.json (Windows)")
    
    claude_desktop_config = {
        "mcpServers": {
            "movie-recommender": {
                "command": uv_path,
                "args": [
                    "--directory",
                    str(current_dir),
                    "run",
                    "python",
                    "-m",
                    "src.mcp_server"
                ],
                "env": {
                    "OPENAI_API_KEY": "your-openai-api-key-here"
                }
            }
        }
    }
    
    print(json.dumps(claude_desktop_config, indent=2))
    
    print(f"\n💻 Claude Code Configuration:")
    print(f"Run this command in your terminal:")
    print(f"claude mcp add --transport stdio movie-recommender -- uv --directory {current_dir} run python -m src.mcp_server")

def main():
    """Run all validation checks."""
    print("🎬 Movie Recommendation MCP Setup Validator")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("uv Installation", lambda: check_uv_installation() is not None),
        ("Project Setup", check_project_setup),
        ("MCP Server Import", test_mcp_server_import),
        ("MCP Server Startup", test_mcp_server_startup),
        ("MCP Tool Testing", test_mcp_tools),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} - EXCEPTION: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("📊 Validation Summary:")
    print("-" * 30)
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 All checks passed! Your MCP setup is ready.")
        generate_config_examples()
    elif passed >= len(results) - 1:
        print("\n⚠️  Almost ready! Fix any failing checks above.")
        generate_config_examples()
    else:
        print("\n❌ Several issues found. Please fix the failing checks.")
        print("\nCommon fixes:")
        print("- Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("- Install dependencies: uv sync")
        print("- Check you're in the project directory")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)