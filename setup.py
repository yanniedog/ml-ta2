import os
import sys
import subprocess
import platform
import webbrowser
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system().lower() == 'windows' else 'clear')


def print_header():
    """Print a nice header for the setup wizard."""
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}"
          "╔══════════════════════════════════════════════════════════════════════════════╗\n"
          "║                    ML-TA Setup Wizard                    ║\n"
          "║            Machine Learning Technical Analysis System             ║\n"
          "╚══════════════════════════════════════════════════════════════════════════════╝"
          f"{Colors.ENDC}\n")


def run_command(command, cwd=None, show_output=True):
    """Run a shell command and return True if successful."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or Path.cwd(),
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            if show_output:
                print(f"{Colors.FAIL}Error: {result.stderr}{Colors.ENDC}")
            return False
        return True
    except Exception as e:
        if show_output:
            print(f"{Colors.FAIL}Error running command: {e}{Colors.ENDC}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print(f"{Colors.OKBLUE}Checking Python version...{Colors.ENDC}")
    if sys.version_info < (3, 10):
        print(f"{Colors.FAIL}Error: Python 3.10 or higher is required.{Colors.ENDC}")
        print(f"{Colors.WARNING}Your version: {sys.version}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Please install Python 3.10 or later from https://www.python.org/downloads/{Colors.ENDC}")
        return False
    print(f"{Colors.OKGREEN}✓ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible{Colors.ENDC}")
    return True


def create_virtual_environment(venv_path):
    """Create a Python virtual environment."""
    print(f"\n{Colors.OKBLUE}Setting up virtual environment...{Colors.ENDC}")
    
    if venv_path.exists():
        print(f"{Colors.WARNING}Virtual environment already exists at {venv_path}{Colors.ENDC}")
        return True
    
    print(f"{Colors.OKCYAN}Creating virtual environment at {venv_path}...{Colors.ENDC}")
    if not run_command(f"{sys.executable} -m venv {venv_path}", show_output=False):
        print(f"{Colors.FAIL}Failed to create virtual environment.{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✓ Virtual environment created successfully{Colors.ENDC}")
    return True


def install_dependencies(venv_path, project_root):
    """Install required dependencies."""
    print(f"\n{Colors.OKBLUE}Installing dependencies...{Colors.ENDC}")
    
    # Determine the correct pip executable
    pip_cmd = f"{venv_path}/Scripts/pip" if platform.system() == "Windows" else f"{venv_path}/bin/pip"
    
    # Upgrade pip first
    print(f"{Colors.OKCYAN}Upgrading pip...{Colors.ENDC}")
    if not run_command(f"{pip_cmd} install --upgrade pip", show_output=False):
        print(f"{Colors.WARNING}Warning: Failed to upgrade pip, continuing anyway...{Colors.ENDC}")
    
    # Install requirements
    requirements = ["requirements.txt", "requirements-dev.txt"]
    for req in requirements:
        req_path = project_root / req
        if req_path.exists():
            print(f"{Colors.OKCYAN}Installing from {req}...{Colors.ENDC}")
            if not run_command(f"{pip_cmd} install -r {req_path}", show_output=False):
                print(f"{Colors.FAIL}Failed to install dependencies from {req}{Colors.ENDC}")
                return False
    
    print(f"{Colors.OKGREEN}✓ Dependencies installed successfully{Colors.ENDC}")
    return True


def setup_configuration(project_root):
    """Set up configuration files."""
    print(f"\n{Colors.OKBLUE}Setting up configuration...{Colors.ENDC}")
    
    # Create .env file if it doesn't exist
    env_file = project_root / ".env"
    if not env_file.exists():
        print(f"{Colors.OKCYAN}Creating .env file...{Colors.ENDC}")
        with open(env_file, "w") as f:
            f.write("# ML-TA Configuration\n")
            f.write("# Get your Binance API keys from: https://www.binance.com/en/support/faq/360002502072\n")
            f.write("BINANCE_API_KEY=your_api_key_here\n")
            f.write("BINANCE_SECRET_KEY=your_secret_key_here\n")
        print(f"{Colors.OKGREEN}✓ .env file created{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}.env file already exists, skipping...{Colors.ENDC}")
    
    return True


def start_application(venv_path):
    """Start the ML-TA application."""
    print(f"\n{Colors.OKBLUE}Starting ML-TA application...{Colors.ENDC}")
    
    # Determine the correct Python executable
    python_cmd = f"{venv_path}/Scripts/python" if platform.system() == "Windows" else f"{venv_path}/bin/python"
    
    # Start the web interface
    print(f"{Colors.OKCYAN}Launching web interface...{Colors.ENDC}")
    
    if platform.system() == "Windows":
        # Use a different approach for Windows
        os.system(f'start cmd /k "{python_cmd} -m src.web.app"')
    else:
        subprocess.Popen(f"{python_cmd} -m src.web.app", shell=True)
    
    # Open browser after a short delay
    import time
    time.sleep(3)
    webbrowser.open("http://localhost:5000")
    
    print(f"{Colors.OKGREEN}✓ Application started successfully!{Colors.ENDC}")
    print(f"{Colors.OKCYAN}The web interface is now available at: {Colors.UNDERLINE}http://localhost:5000{Colors.ENDC}")
    print(f"{Colors.WARNING}Note: To stop the application, close the terminal window that opened.{Colors.ENDC}")


def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        return 1
    
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / ".venv"
    
    # Create virtual environment
    if not create_virtual_environment(venv_path):
        input("\nPress Enter to exit...")
        return 1
    
    # Install dependencies
    if not install_dependencies(venv_path, project_root):
        input("\nPress Enter to exit...")
        return 1
    
    # Setup configuration
    if not setup_configuration(project_root):
        input("\nPress Enter to exit...")
        return 1
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}=== Setup Complete ==={Colors.ENDC}")
    print(f"\n{Colors.OKCYAN}The ML-TA system has been successfully set up!{Colors.ENDC}")
    
    # Ask to start the application
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}1. Configure your Binance API keys in the .env file (optional){Colors.ENDC}")
    print(f"{Colors.OKCYAN}2. Start the application using the web interface{Colors.ENDC}")
    
    if input(f"\n{Colors.BOLD}Start the application now?{Colors.ENDC} [Y/n] ").lower() != 'n':
        start_application(venv_path)
    else:
        print(f"\n{Colors.OKCYAN}To start the application later, run:{Colors.ENDC}")
        if platform.system() == "Windows":
            print(f"{Colors.OKCYAN}  .\\start.bat{Colors.ENDC}")
        else:
            print(f"{Colors.OKCYAN}  ./start.sh{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Or manually:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}  {venv_path}/Scripts/activate  (on Windows){Colors.ENDC}")
        print(f"{Colors.OKCYAN}  {venv_path}/bin/activate      (on macOS/Linux){Colors.ENDC}")
        print(f"{Colors.OKCYAN}  python -m src.web.app{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}Thank you for using ML-TA!{Colors.ENDC}")
    input(f"{Colors.BOLD}\nPress Enter to exit...{Colors.ENDC}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup cancelled by user.{Colors.ENDC}")
        sys.exit(1)
