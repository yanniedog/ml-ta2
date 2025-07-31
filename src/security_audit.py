"""
Security audit and penetration testing framework for ML-TA system.

This module provides comprehensive security testing, vulnerability assessment,
and penetration testing capabilities for the ML-TA trading analysis platform.
"""

import os
import sys
import re
import json
import hashlib
import secrets
import subprocess
import socket
import ssl
import requests
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import base64

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityVulnerability:
    """Security vulnerability container."""
    
    id: str
    title: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # INJECTION, AUTH, CRYPTO, XSS, etc.
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    affected_component: str = ""
    file_path: str = ""
    line_number: int = 0
    evidence: str = ""
    recommendation: str = ""
    references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityScanResult:
    """Security scan result container."""
    
    scan_type: str
    target: str
    start_time: datetime
    end_time: datetime
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time
    
    @property
    def critical_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == "CRITICAL"])
    
    @property
    def high_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == "HIGH"])
    
    @property
    def medium_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == "MEDIUM"])
    
    @property
    def low_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == "LOW"])


class CodeSecurityScanner:
    """Static code security analysis scanner."""
    
    def __init__(self):
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.excluded_paths = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability detection patterns."""
        return {
            'sql_injection': {
                'pattern': r'(execute|query|cursor\.execute)\s*\(\s*["\'].*?%s.*?["\']',
                'severity': 'HIGH',
                'cwe': 'CWE-89',
                'description': 'Potential SQL injection vulnerability'
            },
            'command_injection': {
                'pattern': r'(os\.system|subprocess\.call|subprocess\.run|eval|exec)\s*\(',
                'severity': 'HIGH',
                'cwe': 'CWE-78',
                'description': 'Potential command injection vulnerability'
            },
            'hardcoded_secrets': {
                'pattern': r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                'severity': 'CRITICAL',
                'cwe': 'CWE-798',
                'description': 'Hardcoded credentials detected'
            },
            'weak_crypto': {
                'pattern': r'(md5|sha1|des|rc4)\s*\(',
                'severity': 'MEDIUM',
                'cwe': 'CWE-327',
                'description': 'Weak cryptographic algorithm'
            },
            'unsafe_deserialization': {
                'pattern': r'(pickle\.loads|yaml\.load|eval)\s*\(',
                'severity': 'HIGH',
                'cwe': 'CWE-502',
                'description': 'Unsafe deserialization'
            },
            'path_traversal': {
                'pattern': r'open\s*\(\s*.*?\+.*?\)',
                'severity': 'MEDIUM',
                'cwe': 'CWE-22',
                'description': 'Potential path traversal vulnerability'
            },
            'debug_code': {
                'pattern': r'(print\s*\(|console\.log|debugger|pdb\.set_trace)',
                'severity': 'LOW',
                'cwe': 'CWE-489',
                'description': 'Debug code in production'
            },
            'insecure_random': {
                'pattern': r'random\.(random|randint|choice)',
                'severity': 'MEDIUM',
                'cwe': 'CWE-338',
                'description': 'Insecure random number generation'
            }
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a single file for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for pattern_name, pattern_info in self.vulnerability_patterns.items():
                pattern = pattern_info['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    
                    for match in matches:
                        vuln = SecurityVulnerability(
                            id=f"{pattern_name}_{file_path.name}_{line_num}",
                            title=f"{pattern_name.replace('_', ' ').title()} Vulnerability",
                            description=pattern_info['description'],
                            severity=pattern_info['severity'],
                            category=pattern_name.upper(),
                            cwe_id=pattern_info.get('cwe'),
                            affected_component=str(file_path),
                            file_path=str(file_path),
                            line_number=line_num,
                            evidence=line.strip(),
                            recommendation=self._get_recommendation(pattern_name)
                        )
                        vulnerabilities.append(vuln)
                        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_directory(self, directory: Path) -> SecurityScanResult:
        """Scan directory recursively for security vulnerabilities."""
        start_time = datetime.now()
        all_vulnerabilities = []
        
        logger.info(f"Starting security scan of directory: {directory}")
        
        for file_path in directory.rglob('*.py'):
            # Skip excluded paths
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
            
            file_vulnerabilities = self.scan_file(file_path)
            all_vulnerabilities.extend(file_vulnerabilities)
        
        end_time = datetime.now()
        
        # Create summary
        summary = {
            'total_files_scanned': len(list(directory.rglob('*.py'))),
            'total_vulnerabilities': len(all_vulnerabilities),
            'critical': len([v for v in all_vulnerabilities if v.severity == "CRITICAL"]),
            'high': len([v for v in all_vulnerabilities if v.severity == "HIGH"]),
            'medium': len([v for v in all_vulnerabilities if v.severity == "MEDIUM"]),
            'low': len([v for v in all_vulnerabilities if v.severity == "LOW"])
        }
        
        result = SecurityScanResult(
            scan_type="static_code_analysis",
            target=str(directory),
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=all_vulnerabilities,
            summary=summary
        )
        
        logger.info(f"Security scan completed. Found {len(all_vulnerabilities)} vulnerabilities")
        return result
    
    def _get_recommendation(self, vulnerability_type: str) -> str:
        """Get security recommendation for vulnerability type."""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM methods to prevent SQL injection',
            'command_injection': 'Validate and sanitize all user inputs before executing commands',
            'hardcoded_secrets': 'Use environment variables or secure secret management systems',
            'weak_crypto': 'Use strong cryptographic algorithms like AES-256, SHA-256, or better',
            'unsafe_deserialization': 'Use safe serialization formats like JSON or validate inputs',
            'path_traversal': 'Validate and sanitize file paths, use absolute paths',
            'debug_code': 'Remove debug code before deploying to production',
            'insecure_random': 'Use cryptographically secure random number generators'
        }
        return recommendations.get(vulnerability_type, 'Review and fix the identified security issue')


class NetworkSecurityScanner:
    """Network security scanner for port scanning and service detection."""
    
    def __init__(self):
        self.common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 6379, 8000, 8080, 8443, 9090]
    
    def scan_host(self, host: str, ports: Optional[List[int]] = None) -> SecurityScanResult:
        """Scan host for open ports and services."""
        start_time = datetime.now()
        vulnerabilities = []
        
        if ports is None:
            ports = self.common_ports
        
        logger.info(f"Starting network scan of host: {host}")
        
        open_ports = []
        
        for port in ports:
            if self._is_port_open(host, port):
                open_ports.append(port)
                
                # Check for common vulnerabilities
                service_vulns = self._check_service_vulnerabilities(host, port)
                vulnerabilities.extend(service_vulns)
        
        end_time = datetime.now()
        
        summary = {
            'total_ports_scanned': len(ports),
            'open_ports': len(open_ports),
            'vulnerabilities_found': len(vulnerabilities)
        }
        
        metadata = {
            'open_ports': open_ports,
            'scanned_ports': ports
        }
        
        result = SecurityScanResult(
            scan_type="network_scan",
            target=host,
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=vulnerabilities,
            summary=summary,
            metadata=metadata
        )
        
        logger.info(f"Network scan completed. Found {len(open_ports)} open ports")
        return result
    
    def _is_port_open(self, host: str, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open on the target host."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, socket.error):
            return False
    
    def _check_service_vulnerabilities(self, host: str, port: int) -> List[SecurityVulnerability]:
        """Check for common service vulnerabilities."""
        vulnerabilities = []
        
        # Check for common insecure services
        insecure_services = {
            21: ("FTP", "Unencrypted file transfer protocol"),
            23: ("Telnet", "Unencrypted remote access protocol"),
            25: ("SMTP", "Potentially unsecured email service"),
            80: ("HTTP", "Unencrypted web service"),
            110: ("POP3", "Unencrypted email retrieval"),
            143: ("IMAP", "Unencrypted email access")
        }
        
        if port in insecure_services:
            service_name, description = insecure_services[port]
            
            vuln = SecurityVulnerability(
                id=f"insecure_service_{host}_{port}",
                title=f"Insecure {service_name} Service",
                description=f"{description} detected on port {port}",
                severity="MEDIUM",
                category="NETWORK",
                affected_component=f"{host}:{port}",
                recommendation=f"Consider using secure alternatives or implementing encryption for {service_name}"
            )
            vulnerabilities.append(vuln)
        
        # Check SSL/TLS configuration for HTTPS services
        if port in [443, 8443]:
            ssl_vulns = self._check_ssl_vulnerabilities(host, port)
            vulnerabilities.extend(ssl_vulns)
        
        return vulnerabilities
    
    def _check_ssl_vulnerabilities(self, host: str, port: int) -> List[SecurityVulnerability]:
        """Check SSL/TLS configuration vulnerabilities."""
        vulnerabilities = []
        
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    # Check for weak ciphers
                    if cipher and 'RC4' in cipher[0] or 'DES' in cipher[0]:
                        vuln = SecurityVulnerability(
                            id=f"weak_cipher_{host}_{port}",
                            title="Weak SSL/TLS Cipher",
                            description=f"Weak cipher suite detected: {cipher[0]}",
                            severity="HIGH",
                            category="CRYPTO",
                            affected_component=f"{host}:{port}",
                            recommendation="Configure server to use strong cipher suites"
                        )
                        vulnerabilities.append(vuln)
                    
                    # Check certificate expiration
                    if cert:
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry < 30:
                            severity = "HIGH" if days_until_expiry < 7 else "MEDIUM"
                            vuln = SecurityVulnerability(
                                id=f"cert_expiry_{host}_{port}",
                                title="SSL Certificate Expiring Soon",
                                description=f"Certificate expires in {days_until_expiry} days",
                                severity=severity,
                                category="CRYPTO",
                                affected_component=f"{host}:{port}",
                                recommendation="Renew SSL certificate before expiration"
                            )
                            vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.debug(f"SSL check failed for {host}:{port}: {e}")
        
        return vulnerabilities


class WebApplicationScanner:
    """Web application security scanner."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ML-TA Security Scanner 1.0'
        })
    
    def scan_web_application(self, base_url: str) -> SecurityScanResult:
        """Scan web application for common vulnerabilities."""
        start_time = datetime.now()
        vulnerabilities = []
        
        logger.info(f"Starting web application scan: {base_url}")
        
        # Test for common vulnerabilities
        vulnerabilities.extend(self._test_information_disclosure(base_url))
        vulnerabilities.extend(self._test_security_headers(base_url))
        vulnerabilities.extend(self._test_http_methods(base_url))
        vulnerabilities.extend(self._test_directory_traversal(base_url))
        vulnerabilities.extend(self._test_xss_protection(base_url))
        
        end_time = datetime.now()
        
        summary = {
            'total_tests': 5,
            'vulnerabilities_found': len(vulnerabilities),
            'critical': len([v for v in vulnerabilities if v.severity == "CRITICAL"]),
            'high': len([v for v in vulnerabilities if v.severity == "HIGH"]),
            'medium': len([v for v in vulnerabilities if v.severity == "MEDIUM"]),
            'low': len([v for v in vulnerabilities if v.severity == "LOW"])
        }
        
        result = SecurityScanResult(
            scan_type="web_application_scan",
            target=base_url,
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=vulnerabilities,
            summary=summary
        )
        
        logger.info(f"Web application scan completed. Found {len(vulnerabilities)} vulnerabilities")
        return result
    
    def _test_information_disclosure(self, base_url: str) -> List[SecurityVulnerability]:
        """Test for information disclosure vulnerabilities."""
        vulnerabilities = []
        
        # Test for common sensitive files
        sensitive_paths = [
            '/.env', '/config.yaml', '/config.json', '/backup.sql',
            '/admin', '/debug', '/test', '/.git/config'
        ]
        
        for path in sensitive_paths:
            try:
                response = self.session.get(f"{base_url}{path}", timeout=10)
                
                if response.status_code == 200:
                    vuln = SecurityVulnerability(
                        id=f"info_disclosure_{path.replace('/', '_')}",
                        title="Information Disclosure",
                        description=f"Sensitive file accessible: {path}",
                        severity="MEDIUM",
                        category="INFO_DISCLOSURE",
                        affected_component=f"{base_url}{path}",
                        recommendation="Restrict access to sensitive files and directories"
                    )
                    vulnerabilities.append(vuln)
                    
            except Exception as e:
                logger.debug(f"Error testing path {path}: {e}")
        
        return vulnerabilities
    
    def _test_security_headers(self, base_url: str) -> List[SecurityVulnerability]:
        """Test for missing security headers."""
        vulnerabilities = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            headers = response.headers
            
            # Check for missing security headers
            security_headers = {
                'X-Frame-Options': 'Missing X-Frame-Options header (clickjacking protection)',
                'X-Content-Type-Options': 'Missing X-Content-Type-Options header (MIME sniffing protection)',
                'X-XSS-Protection': 'Missing X-XSS-Protection header (XSS protection)',
                'Strict-Transport-Security': 'Missing HSTS header (HTTPS enforcement)',
                'Content-Security-Policy': 'Missing CSP header (XSS/injection protection)'
            }
            
            for header, description in security_headers.items():
                if header not in headers:
                    vuln = SecurityVulnerability(
                        id=f"missing_header_{header.lower().replace('-', '_')}",
                        title=f"Missing Security Header: {header}",
                        description=description,
                        severity="MEDIUM",
                        category="SECURITY_HEADERS",
                        affected_component=base_url,
                        recommendation=f"Implement {header} security header"
                    )
                    vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.debug(f"Error testing security headers: {e}")
        
        return vulnerabilities
    
    def _test_http_methods(self, base_url: str) -> List[SecurityVulnerability]:
        """Test for dangerous HTTP methods."""
        vulnerabilities = []
        
        dangerous_methods = ['TRACE', 'TRACK', 'DELETE', 'PUT', 'PATCH']
        
        for method in dangerous_methods:
            try:
                response = self.session.request(method, base_url, timeout=10)
                
                if response.status_code not in [405, 501]:  # Method not allowed
                    vuln = SecurityVulnerability(
                        id=f"dangerous_method_{method.lower()}",
                        title=f"Dangerous HTTP Method: {method}",
                        description=f"HTTP {method} method is enabled",
                        severity="MEDIUM",
                        category="HTTP_METHODS",
                        affected_component=base_url,
                        recommendation=f"Disable {method} method if not required"
                    )
                    vulnerabilities.append(vuln)
                    
            except Exception as e:
                logger.debug(f"Error testing method {method}: {e}")
        
        return vulnerabilities
    
    def _test_directory_traversal(self, base_url: str) -> List[SecurityVulnerability]:
        """Test for directory traversal vulnerabilities."""
        vulnerabilities = []
        
        # Simple directory traversal payloads
        payloads = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
        ]
        
        for payload in payloads:
            try:
                # Test common parameters
                params = {'file': payload, 'path': payload, 'page': payload}
                
                for param, value in params.items():
                    response = self.session.get(base_url, params={param: value}, timeout=10)
                    
                    # Check for signs of successful traversal
                    if ('root:' in response.text or 
                        'localhost' in response.text or
                        'Windows' in response.text):
                        
                        vuln = SecurityVulnerability(
                            id=f"directory_traversal_{param}",
                            title="Directory Traversal Vulnerability",
                            description=f"Directory traversal detected via {param} parameter",
                            severity="HIGH",
                            category="PATH_TRAVERSAL",
                            affected_component=base_url,
                            evidence=f"Parameter: {param}, Payload: {value}",
                            recommendation="Validate and sanitize file path parameters"
                        )
                        vulnerabilities.append(vuln)
                        break
                        
            except Exception as e:
                logger.debug(f"Error testing directory traversal: {e}")
        
        return vulnerabilities
    
    def _test_xss_protection(self, base_url: str) -> List[SecurityVulnerability]:
        """Test for XSS protection mechanisms."""
        vulnerabilities = []
        
        # Simple XSS payload
        xss_payload = '<script>alert("XSS")</script>'
        
        try:
            # Test common parameters
            params = {'q': xss_payload, 'search': xss_payload, 'input': xss_payload}
            
            for param, value in params.items():
                response = self.session.get(base_url, params={param: value}, timeout=10)
                
                # Check if payload is reflected without encoding
                if xss_payload in response.text:
                    vuln = SecurityVulnerability(
                        id=f"reflected_xss_{param}",
                        title="Potential Reflected XSS",
                        description=f"Unfiltered user input reflected via {param} parameter",
                        severity="HIGH",
                        category="XSS",
                        affected_component=base_url,
                        evidence=f"Parameter: {param}, Payload reflected",
                        recommendation="Implement proper input validation and output encoding"
                    )
                    vulnerabilities.append(vuln)
                    
        except Exception as e:
            logger.debug(f"Error testing XSS protection: {e}")
        
        return vulnerabilities


class SecurityAuditor:
    """Comprehensive security auditor orchestrating all security tests."""
    
    def __init__(self):
        self.code_scanner = CodeSecurityScanner()
        self.network_scanner = NetworkSecurityScanner()
        self.web_scanner = WebApplicationScanner()
        self.audit_results = []
    
    def run_comprehensive_audit(self, 
                               code_path: Optional[Path] = None,
                               network_targets: Optional[List[str]] = None,
                               web_targets: Optional[List[str]] = None) -> List[SecurityScanResult]:
        """Run comprehensive security audit."""
        logger.info("Starting comprehensive security audit")
        
        results = []
        
        # Code security scan
        if code_path:
            logger.info("Running static code analysis")
            code_result = self.code_scanner.scan_directory(code_path)
            results.append(code_result)
        
        # Network security scan
        if network_targets:
            for target in network_targets:
                logger.info(f"Running network scan for {target}")
                network_result = self.network_scanner.scan_host(target)
                results.append(network_result)
        
        # Web application scan
        if web_targets:
            for target in web_targets:
                logger.info(f"Running web application scan for {target}")
                web_result = self.web_scanner.scan_web_application(target)
                results.append(web_result)
        
        self.audit_results = results
        logger.info(f"Security audit completed. {len(results)} scan results generated")
        
        return results
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security audit report."""
        if not self.audit_results:
            return "No security audit results available."
        
        report = ["ML-TA Security Audit Report", "=" * 50, ""]
        
        # Executive summary
        total_vulns = sum(len(result.vulnerabilities) for result in self.audit_results)
        critical_vulns = sum(result.critical_count for result in self.audit_results)
        high_vulns = sum(result.high_count for result in self.audit_results)
        
        report.extend([
            "Executive Summary:",
            f"  Total Vulnerabilities: {total_vulns}",
            f"  Critical: {critical_vulns}",
            f"  High: {high_vulns}",
            f"  Medium: {sum(result.medium_count for result in self.audit_results)}",
            f"  Low: {sum(result.low_count for result in self.audit_results)}",
            ""
        ])
        
        # Risk assessment
        if critical_vulns > 0:
            risk_level = "CRITICAL"
        elif high_vulns > 5:
            risk_level = "HIGH"
        elif high_vulns > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        report.extend([
            f"Overall Risk Level: {risk_level}",
            ""
        ])
        
        # Detailed results
        for result in self.audit_results:
            report.extend([
                f"Scan Type: {result.scan_type.title()}",
                f"Target: {result.target}",
                f"Duration: {result.duration}",
                f"Vulnerabilities Found: {len(result.vulnerabilities)}",
                ""
            ])
            
            # Top vulnerabilities
            critical_vulns = [v for v in result.vulnerabilities if v.severity == "CRITICAL"]
            high_vulns = [v for v in result.vulnerabilities if v.severity == "HIGH"]
            
            if critical_vulns or high_vulns:
                report.append("Critical/High Severity Issues:")
                
                for vuln in critical_vulns + high_vulns:
                    report.extend([
                        f"  [{vuln.severity}] {vuln.title}",
                        f"    File: {vuln.file_path}:{vuln.line_number}" if vuln.file_path else f"    Component: {vuln.affected_component}",
                        f"    Description: {vuln.description}",
                        f"    Recommendation: {vuln.recommendation}",
                        ""
                    ])
        
        # Recommendations
        report.extend([
            "Security Recommendations:",
            "1. Address all CRITICAL and HIGH severity vulnerabilities immediately",
            "2. Implement security code review processes",
            "3. Set up automated security scanning in CI/CD pipeline",
            "4. Regular security audits and penetration testing",
            "5. Security awareness training for development team",
            "6. Implement security monitoring and incident response procedures",
            ""
        ])
        
        return "\n".join(report)


# Factory functions
def create_code_scanner() -> CodeSecurityScanner:
    """Create code security scanner instance."""
    return CodeSecurityScanner()


def create_network_scanner() -> NetworkSecurityScanner:
    """Create network security scanner instance."""
    return NetworkSecurityScanner()


def create_web_scanner() -> WebApplicationScanner:
    """Create web application scanner instance."""
    return WebApplicationScanner()


def create_security_auditor() -> SecurityAuditor:
    """Create security auditor instance."""
    return SecurityAuditor()


if __name__ == "__main__":
    # Example usage
    print("ML-TA Security Audit Framework")
    
    # Create security auditor
    auditor = create_security_auditor()
    
    # Run comprehensive audit
    current_dir = Path(".")
    results = auditor.run_comprehensive_audit(
        code_path=current_dir,
        network_targets=["localhost"],
        web_targets=["http://localhost:8000"]
    )
    
    # Generate report
    print(auditor.generate_security_report())
