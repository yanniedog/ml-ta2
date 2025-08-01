#!/usr/bin/env python3
"""
User Acceptance Testing Framework for ML-TA System

This module implements Phase 10.2 requirements for comprehensive user acceptance
testing with non-technical users, including GUI validation, UX flows, feedback
collection, and accessibility compliance.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import webbrowser
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import get_config


class UserType(Enum):
    """Types of users for UAT."""
    NON_TECHNICAL = "non_technical"
    BUSINESS_USER = "business_user"
    POWER_USER = "power_user"
    ACCESSIBILITY_USER = "accessibility_user"


class TestScenario(Enum):
    """User acceptance test scenarios."""
    FIRST_TIME_SETUP = "first_time_setup"
    BASIC_PREDICTION = "basic_prediction"
    DATA_EXPLORATION = "data_exploration"
    MODEL_MONITORING = "model_monitoring"
    SYSTEM_CONFIGURATION = "system_configuration"
    HELP_AND_SUPPORT = "help_and_support"
    ACCESSIBILITY_NAVIGATION = "accessibility_navigation"


@dataclass
class UATResult:
    """User acceptance test result."""
    test_id: str
    user_type: UserType
    scenario: TestScenario
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    user_satisfaction_score: int  # 1-10 scale
    ease_of_use_score: int       # 1-10 scale
    completion_rate: float       # 0-1
    errors_encountered: List[str]
    feedback: str
    recommendations: List[str]
    accessibility_issues: List[str]


class UserAcceptanceTestFramework:
    """Comprehensive user acceptance testing framework."""
    
    def __init__(self):
        self.config = get_config()
        self.results: List[UATResult] = []
        self.logger = self._setup_logging()
        self.web_server_process = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for UAT."""
        logger = logging.getLogger("uat_framework")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f"uat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def start_web_server(self) -> bool:
        """Start the ML-TA web server for testing."""
        try:
            # Start the web frontend server
            cmd = [sys.executable, "-m", "src.web_frontend"]
            self.web_server_process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for server to start
            time.sleep(5)
            
            # Check if server is running
            if self.web_server_process.poll() is None:
                self.logger.info("Web server started successfully for UAT")
                return True
            else:
                self.logger.error("Failed to start web server")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting web server: {str(e)}")
            return False
    
    def stop_web_server(self):
        """Stop the web server."""
        if self.web_server_process:
            self.web_server_process.terminate()
            self.web_server_process.wait()
            self.logger.info("Web server stopped")
    
    def create_test_scenario_guide(self, scenario: TestScenario, user_type: UserType) -> Dict[str, Any]:
        """Create detailed test scenario guide for users."""
        guides = {
            TestScenario.FIRST_TIME_SETUP: {
                "title": "First-Time System Setup",
                "objective": "Successfully set up and access the ML-TA system for the first time",
                "steps": [
                    "1. Open web browser and navigate to http://localhost:8000",
                    "2. Complete the initial setup wizard",
                    "3. Configure basic system settings",
                    "4. Verify all components are working",
                    "5. Access the main dashboard"
                ],
                "success_criteria": [
                    "System loads without errors",
                    "Setup completes in under 10 minutes",
                    "All dashboard elements are visible and functional",
                    "User can navigate between main sections"
                ],
                "estimated_time": "10-15 minutes"
            },
            
            TestScenario.BASIC_PREDICTION: {
                "title": "Basic Prediction Request",
                "objective": "Successfully request and receive a trading prediction",
                "steps": [
                    "1. Navigate to the Prediction section",
                    "2. Select a trading symbol (e.g., BTCUSDT)",
                    "3. Choose time frame (e.g., 1 hour)",
                    "4. Click 'Generate Prediction'",
                    "5. Review the prediction results",
                    "6. Understand the confidence levels and recommendations"
                ],
                "success_criteria": [
                    "Prediction generates within 5 seconds",
                    "Results are clearly presented and understandable",
                    "Confidence levels and reasoning are provided",
                    "User feels confident in interpreting results"
                ],
                "estimated_time": "5-10 minutes"
            },
            
            TestScenario.DATA_EXPLORATION: {
                "title": "Data Exploration and Visualization",
                "objective": "Explore historical data and understand market trends",
                "steps": [
                    "1. Go to the Data Explorer section",
                    "2. Select a cryptocurrency pair",
                    "3. Choose a date range for analysis",
                    "4. View price charts and technical indicators",
                    "5. Examine feature importance and correlations",
                    "6. Export or save interesting findings"
                ],
                "success_criteria": [
                    "Charts load quickly and are interactive",
                    "Data is presented in an understandable format",
                    "User can identify trends and patterns",
                    "Export functionality works correctly"
                ],
                "estimated_time": "10-15 minutes"
            },
            
            TestScenario.MODEL_MONITORING: {
                "title": "Model Performance Monitoring",
                "objective": "Monitor and understand model performance metrics",
                "steps": [
                    "1. Access the Monitoring dashboard",
                    "2. Review model accuracy metrics",
                    "3. Check system health indicators",
                    "4. Examine recent prediction performance",
                    "5. Understand alert notifications",
                    "6. Review historical performance trends"
                ],
                "success_criteria": [
                    "Metrics are clearly visualized",
                    "User understands what metrics mean",
                    "Alerts are comprehensible",
                    "Performance trends are evident"
                ],
                "estimated_time": "8-12 minutes"
            },
            
            TestScenario.HELP_AND_SUPPORT: {
                "title": "Help System and Documentation",
                "objective": "Successfully find help and resolve questions",
                "steps": [
                    "1. Access the Help section",
                    "2. Search for specific topics",
                    "3. Follow step-by-step tutorials",
                    "4. Access FAQ and troubleshooting guides",
                    "5. Find contact information for support",
                    "6. Test any interactive help features"
                ],
                "success_criteria": [
                    "Help content is easy to find",
                    "Information is clear and actionable",
                    "Search functionality works well",
                    "User feels supported and can self-serve"
                ],
                "estimated_time": "5-10 minutes"
            },
            
            TestScenario.ACCESSIBILITY_NAVIGATION: {
                "title": "Accessibility and Alternative Navigation",
                "objective": "Test system accessibility for users with disabilities",
                "steps": [
                    "1. Navigate using only keyboard (no mouse)",
                    "2. Test with screen reader compatibility",
                    "3. Check color contrast and readability",
                    "4. Verify text scaling and zoom functionality",
                    "5. Test voice navigation if available",
                    "6. Evaluate alternative input methods"
                ],
                "success_criteria": [
                    "All features accessible via keyboard",
                    "Screen reader compatibility",
                    "Sufficient color contrast",
                    "Text scales appropriately",
                    "WCAG 2.1 AA compliance"
                ],
                "estimated_time": "15-20 minutes"
            }
        }
        
        base_guide = guides.get(scenario, {})
        
        # Customize guide based on user type
        if user_type == UserType.NON_TECHNICAL:
            base_guide["additional_notes"] = [
                "Take your time and don't worry about making mistakes",
                "Focus on whether the interface feels intuitive",
                "Note any technical terms that are confusing",
                "Consider how confident you feel using each feature"
            ]
        elif user_type == UserType.ACCESSIBILITY_USER:
            base_guide["additional_notes"] = [
                "Test with your preferred assistive technologies",
                "Note any barriers to accessing information",
                "Evaluate keyboard navigation paths",
                "Check for proper ARIA labels and descriptions"
            ]
        
        return base_guide
    
    def run_guided_user_test(self, user_type: UserType, scenario: TestScenario) -> UATResult:
        """Run a guided user acceptance test."""
        test_id = f"UAT_{user_type.value}_{scenario.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting UAT: {test_id}")
        
        # Get test scenario guide
        guide = self.create_test_scenario_guide(scenario, user_type)
        
        # Display test guide to user
        print("\n" + "="*80)
        print(f"USER ACCEPTANCE TEST: {guide['title']}")
        print("="*80)
        print(f"Objective: {guide['objective']}")
        print(f"Estimated Time: {guide['estimated_time']}")
        print(f"User Type: {user_type.value.replace('_', ' ').title()}")
        print("\nTest Steps:")
        for step in guide['steps']:
            print(f"  {step}")
        
        print("\nSuccess Criteria:")
        for criteria in guide['success_criteria']:
            print(f"  • {criteria}")
        
        if 'additional_notes' in guide:
            print("\nAdditional Notes:")
            for note in guide['additional_notes']:
                print(f"  • {note}")
        
        print("\n" + "-"*80)
        
        # Start timing
        start_time = datetime.now()
        
        # Open browser to the application
        try:
            webbrowser.open("http://localhost:8000")
            self.logger.info("Opened browser for UAT")
        except Exception as e:
            self.logger.warning(f"Could not open browser automatically: {str(e)}")
            print("Please manually open http://localhost:8000 in your browser")
        
        # Interactive feedback collection
        print("\nPlease complete the test scenario and then provide feedback:")
        print("Press Enter when you have completed the test...")
        input()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Collect user feedback
        feedback_data = self._collect_user_feedback(scenario, duration)
        
        # Create test result
        result = UATResult(
            test_id=test_id,
            user_type=user_type,
            scenario=scenario,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=feedback_data['completed_successfully'],
            user_satisfaction_score=feedback_data['satisfaction_score'],
            ease_of_use_score=feedback_data['ease_of_use_score'],
            completion_rate=feedback_data['completion_rate'],
            errors_encountered=feedback_data['errors_encountered'],
            feedback=feedback_data['general_feedback'],
            recommendations=feedback_data['recommendations'],
            accessibility_issues=feedback_data['accessibility_issues']
        )
        
        self.results.append(result)
        self.logger.info(f"Completed UAT: {test_id}")
        
        return result
    
    def _collect_user_feedback(self, scenario: TestScenario, duration: float) -> Dict[str, Any]:
        """Collect detailed user feedback after test completion."""
        print("\n" + "="*60)
        print("USER FEEDBACK COLLECTION")
        print("="*60)
        
        # Basic completion info
        completed = self._get_yes_no_input("Did you successfully complete the test scenario?")
        
        # Ratings (1-10 scale)
        satisfaction = self._get_rating_input(
            "How satisfied are you with the overall experience? (1-10)", 1, 10
        )
        
        ease_of_use = self._get_rating_input(
            "How easy was it to use the system? (1-10)", 1, 10
        )
        
        # Completion percentage
        if not completed:
            completion_rate = self._get_rating_input(
                "What percentage of the test did you complete? (0-100)", 0, 100
            ) / 100
        else:
            completion_rate = 1.0
        
        # Error collection
        errors = []
        if self._get_yes_no_input("Did you encounter any errors or problems?"):
            print("Please describe each error (press Enter with empty line to finish):")
            while True:
                error = input("Error description: ").strip()
                if not error:
                    break
                errors.append(error)
        
        # General feedback
        print("\nGeneral feedback (press Enter with empty line to finish):")
        feedback_lines = []
        while True:
            line = input().strip()
            if not line:
                break
            feedback_lines.append(line)
        general_feedback = " ".join(feedback_lines)
        
        # Recommendations
        recommendations = []
        if self._get_yes_no_input("Do you have any recommendations for improvement?"):
            print("Please list your recommendations (press Enter with empty line to finish):")
            while True:
                rec = input("Recommendation: ").strip()
                if not rec:
                    break
                recommendations.append(rec)
        
        # Accessibility issues
        accessibility_issues = []
        if self._get_yes_no_input("Did you notice any accessibility issues?"):
            print("Please describe accessibility issues (press Enter with empty line to finish):")
            while True:
                issue = input("Accessibility issue: ").strip()
                if not issue:
                    break
                accessibility_issues.append(issue)
        
        return {
            'completed_successfully': completed,
            'satisfaction_score': satisfaction,
            'ease_of_use_score': ease_of_use,
            'completion_rate': completion_rate,
            'errors_encountered': errors,
            'general_feedback': general_feedback,
            'recommendations': recommendations,
            'accessibility_issues': accessibility_issues,
            'test_duration_minutes': duration / 60
        }
    
    def _get_yes_no_input(self, question: str) -> bool:
        """Get yes/no input from user."""
        while True:
            response = input(f"{question} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def _get_rating_input(self, question: str, min_val: int, max_val: int) -> int:
        """Get rating input from user."""
        while True:
            try:
                response = int(input(f"{question}: ").strip())
                if min_val <= response <= max_val:
                    return response
                else:
                    print(f"Please enter a number between {min_val} and {max_val}.")
            except ValueError:
                print("Please enter a valid number.")
    
    def run_comprehensive_uat_suite(self) -> Dict[str, Any]:
        """Run comprehensive user acceptance testing suite."""
        print("\n" + "="*80)
        print("ML-TA COMPREHENSIVE USER ACCEPTANCE TESTING SUITE")
        print("="*80)
        
        # Start web server
        if not self.start_web_server():
            return {"error": "Failed to start web server for UAT"}
        
        try:
            # Define test matrix
            test_matrix = [
                (UserType.NON_TECHNICAL, TestScenario.FIRST_TIME_SETUP),
                (UserType.NON_TECHNICAL, TestScenario.BASIC_PREDICTION),
                (UserType.BUSINESS_USER, TestScenario.DATA_EXPLORATION),
                (UserType.BUSINESS_USER, TestScenario.MODEL_MONITORING),
                (UserType.POWER_USER, TestScenario.SYSTEM_CONFIGURATION),
                (UserType.ACCESSIBILITY_USER, TestScenario.ACCESSIBILITY_NAVIGATION),
                (UserType.NON_TECHNICAL, TestScenario.HELP_AND_SUPPORT)
            ]
            
            # Execute tests
            for user_type, scenario in test_matrix:
                print(f"\n\nPreparing test: {user_type.value} - {scenario.value}")
                input("Press Enter when ready to start this test...")
                
                result = self.run_guided_user_test(user_type, scenario)
                
                # Brief result summary
                print(f"\nTest completed: {result.success}")
                print(f"Satisfaction: {result.user_satisfaction_score}/10")
                print(f"Ease of use: {result.ease_of_use_score}/10")
                
                # Ask if user wants to continue
                if not self._get_yes_no_input("Continue with next test?"):
                    break
            
            # Generate comprehensive report
            report = self._generate_uat_report()
            
            return report
            
        finally:
            # Always stop the web server
            self.stop_web_server()
    
    def _generate_uat_report(self) -> Dict[str, Any]:
        """Generate comprehensive UAT report."""
        if not self.results:
            return {"error": "No UAT results available"}
        
        # Calculate metrics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        
        avg_satisfaction = sum(r.user_satisfaction_score for r in self.results) / total_tests
        avg_ease_of_use = sum(r.ease_of_use_score for r in self.results) / total_tests
        avg_completion_rate = sum(r.completion_rate for r in self.results) / total_tests
        
        # Collect all feedback
        all_errors = []
        all_recommendations = []
        all_accessibility_issues = []
        
        for result in self.results:
            all_errors.extend(result.errors_encountered)
            all_recommendations.extend(result.recommendations)
            all_accessibility_issues.extend(result.accessibility_issues)
        
        # Generate report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate_percent": success_rate,
                "average_satisfaction_score": avg_satisfaction,
                "average_ease_of_use_score": avg_ease_of_use,
                "average_completion_rate": avg_completion_rate
            },
            "detailed_results": [asdict(result) for result in self.results],
            "aggregated_feedback": {
                "total_errors": len(all_errors),
                "unique_errors": list(set(all_errors)),
                "total_recommendations": len(all_recommendations),
                "unique_recommendations": list(set(all_recommendations)),
                "accessibility_issues": list(set(all_accessibility_issues))
            },
            "compliance_assessment": {
                "user_experience_acceptable": success_rate >= 80,
                "satisfaction_target_met": avg_satisfaction >= 7,
                "ease_of_use_target_met": avg_ease_of_use >= 7,
                "accessibility_compliant": len(all_accessibility_issues) == 0,
                "ready_for_non_technical_users": avg_ease_of_use >= 7 and success_rate >= 80
            },
            "recommendations": self._generate_uat_recommendations()
        }
        
        # Save report
        self._save_uat_report(report)
        
        return report
    
    def _generate_uat_recommendations(self) -> List[str]:
        """Generate UAT-based recommendations."""
        recommendations = []
        
        if not self.results:
            return ["No UAT results available for recommendations"]
        
        avg_satisfaction = sum(r.user_satisfaction_score for r in self.results) / len(self.results)
        avg_ease_of_use = sum(r.ease_of_use_score for r in self.results) / len(self.results)
        success_rate = sum(1 for r in self.results if r.success) / len(self.results) * 100
        
        if avg_satisfaction < 7:
            recommendations.append("Improve overall user satisfaction through better UX design")
        
        if avg_ease_of_use < 7:
            recommendations.append("Simplify user interface and improve intuitiveness")
        
        if success_rate < 80:
            recommendations.append("Address usability issues preventing task completion")
        
        # Check for common errors
        all_errors = []
        for result in self.results:
            all_errors.extend(result.errors_encountered)
        
        if all_errors:
            recommendations.append("Address recurring user errors and improve error handling")
        
        # Check accessibility
        accessibility_issues = []
        for result in self.results:
            accessibility_issues.extend(result.accessibility_issues)
        
        if accessibility_issues:
            recommendations.append("Improve accessibility compliance and inclusive design")
        
        return recommendations
    
    def _save_uat_report(self, report: Dict[str, Any]):
        """Save UAT report to file."""
        # Create reports directory
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = reports_dir / f"uat_report_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"UAT report saved: {json_file}")
        
        # Also save a human-readable summary
        summary_file = reports_dir / f"uat_summary_{timestamp}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ML-TA USER ACCEPTANCE TESTING SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            summary = report['summary']
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Success Rate: {summary['success_rate_percent']:.1f}%\n")
            f.write(f"Average Satisfaction: {summary['average_satisfaction_score']:.1f}/10\n")
            f.write(f"Average Ease of Use: {summary['average_ease_of_use_score']:.1f}/10\n\n")
            
            compliance = report['compliance_assessment']
            f.write("COMPLIANCE ASSESSMENT:\n")
            for key, value in compliance.items():
                status = "✓ PASS" if value else "✗ FAIL"
                f.write(f"  {key}: {status}\n")
            
            if report['recommendations']:
                f.write("\nRECOMMENDATIONS:\n")
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")
        
        print(f"\nUAT reports saved:")
        print(f"  Detailed: {json_file}")
        print(f"  Summary: {summary_file}")


def main():
    """Main function to run user acceptance testing."""
    print("ML-TA User Acceptance Testing Framework")
    print("=" * 50)
    
    uat_framework = UserAcceptanceTestFramework()
    
    # Check if user wants to run full suite or individual test
    print("\nChoose testing mode:")
    print("1. Full UAT Suite (recommended)")
    print("2. Individual Test Scenario")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run full suite
        report = uat_framework.run_comprehensive_uat_suite()
        
        if "error" in report:
            print(f"Error: {report['error']}")
            return 1
        
        # Display summary
        print("\n" + "="*60)
        print("UAT SUITE COMPLETION SUMMARY")
        print("="*60)
        
        summary = report['summary']
        print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"Average Satisfaction: {summary['average_satisfaction_score']:.1f}/10")
        print(f"Average Ease of Use: {summary['average_ease_of_use_score']:.1f}/10")
        
        compliance = report['compliance_assessment']
        ready_for_production = all(compliance.values())
        
        if ready_for_production:
            print("\n✓ SYSTEM READY FOR NON-TECHNICAL USERS")
            return 0
        else:
            print("\n✗ SYSTEM NEEDS IMPROVEMENT FOR NON-TECHNICAL USERS")
            return 1
    
    elif choice == "2":
        # Individual test mode
        print("\nAvailable User Types:")
        for i, user_type in enumerate(UserType, 1):
            print(f"{i}. {user_type.value.replace('_', ' ').title()}")
        
        user_choice = int(input("Select user type: ")) - 1
        user_type = list(UserType)[user_choice]
        
        print("\nAvailable Test Scenarios:")
        for i, scenario in enumerate(TestScenario, 1):
            print(f"{i}. {scenario.value.replace('_', ' ').title()}")
        
        scenario_choice = int(input("Select test scenario: ")) - 1
        scenario = list(TestScenario)[scenario_choice]
        
        # Start web server
        if not uat_framework.start_web_server():
            print("Failed to start web server")
            return 1
        
        try:
            # Run individual test
            result = uat_framework.run_guided_user_test(user_type, scenario)
            
            # Display result
            print(f"\nTest Result: {'SUCCESS' if result.success else 'INCOMPLETE'}")
            print(f"Satisfaction: {result.user_satisfaction_score}/10")
            print(f"Ease of Use: {result.ease_of_use_score}/10")
            
            return 0 if result.success else 1
            
        finally:
            uat_framework.stop_web_server()
    
    else:
        print("Invalid choice")
        return 1


if __name__ == "__main__":
    exit(main())
