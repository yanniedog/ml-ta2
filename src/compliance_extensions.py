"""
Extended compliance framework for MiFID II, GDPR, and SOX requirements.

This module implements specific compliance checks and procedures for:
- MiFID II (Markets in Financial Instruments Directive II)
- GDPR (General Data Protection Regulation)
- SOX (Sarbanes-Oxley Act)
"""

import os
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.security import ComplianceCheck, ComplianceStandard, SecurityAuditEvent
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataProcessingRecord:
    """GDPR data processing record."""
    processing_id: str
    data_subject_id: str
    data_categories: List[str]
    processing_purpose: str
    legal_basis: str
    retention_period: str
    third_party_sharing: bool = False
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    processed_at: datetime = field(default_factory=datetime.now)


@dataclass
class MiFIDTransaction:
    """MiFID II transaction record."""
    transaction_id: str
    client_id: str
    instrument: str
    transaction_type: str
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    algorithm_used: Optional[str] = None
    decision_maker: str = ""
    execution_quality: Dict[str, Any] = field(default_factory=dict)


class GDPRCompliance:
    """GDPR compliance implementation."""
    
    def __init__(self):
        """Initialize GDPR compliance."""
        self.processing_records = []
        self.consent_records = {}
        self.data_retention_policies = self._load_retention_policies()
        
        logger.info("GDPRCompliance initialized")
    
    def _load_retention_policies(self) -> Dict[str, str]:
        """Load data retention policies."""
        return {
            'trading_data': '7_years',
            'user_profiles': '5_years',
            'model_predictions': '3_years',
            'system_logs': '2_years',
            'audit_trails': '10_years'
        }
    
    def record_data_processing(self, data_subject_id: str, data_categories: List[str],
                              purpose: str, legal_basis: str) -> str:
        """Record data processing activity."""
        processing_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_records)}"
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purpose=purpose,
            legal_basis=legal_basis,
            retention_period=self._get_retention_period(data_categories)
        )
        
        self.processing_records.append(record)
        logger.info(f"Data processing recorded: {processing_id}")
        
        return processing_id
    
    def _get_retention_period(self, data_categories: List[str]) -> str:
        """Get retention period for data categories."""
        max_retention = '1_year'  # Default
        
        for category in data_categories:
            if category in self.data_retention_policies:
                policy_retention = self.data_retention_policies[category]
                # Simple logic to get maximum retention period
                if '10_years' in policy_retention:
                    max_retention = '10_years'
                elif '7_years' in policy_retention and max_retention != '10_years':
                    max_retention = '7_years'
                elif '5_years' in policy_retention and max_retention not in ['10_years', '7_years']:
                    max_retention = '5_years'
        
        return max_retention
    
    def record_consent(self, data_subject_id: str, consent_type: str, 
                      consent_given: bool, purpose: str) -> str:
        """Record consent for data processing."""
        consent_id = f"consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data_subject_id}"
        
        consent_record = {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'consent_type': consent_type,
            'consent_given': consent_given,
            'purpose': purpose,
            'timestamp': datetime.now(),
            'ip_address': None,  # Should be captured from request
            'user_agent': None   # Should be captured from request
        }
        
        self.consent_records[consent_id] = consent_record
        logger.info(f"Consent recorded: {consent_id}")
        
        return consent_id
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR data subject requests."""
        logger.info(f"Processing {request_type} request for {data_subject_id}")
        
        if request_type == 'access':
            return self._handle_access_request(data_subject_id)
        elif request_type == 'rectification':
            return self._handle_rectification_request(data_subject_id)
        elif request_type == 'erasure':
            return self._handle_erasure_request(data_subject_id)
        elif request_type == 'portability':
            return self._handle_portability_request(data_subject_id)
        else:
            return {'status': 'error', 'message': f'Unknown request type: {request_type}'}
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data access request."""
        # Find all processing records for the data subject
        subject_records = [r for r in self.processing_records if r.data_subject_id == data_subject_id]
        
        # Find consent records
        subject_consents = {k: v for k, v in self.consent_records.items() 
                           if v['data_subject_id'] == data_subject_id}
        
        return {
            'status': 'completed',
            'data_subject_id': data_subject_id,
            'processing_records': [
                {
                    'processing_id': r.processing_id,
                    'data_categories': r.data_categories,
                    'purpose': r.processing_purpose,
                    'legal_basis': r.legal_basis,
                    'processed_at': r.processed_at.isoformat()
                }
                for r in subject_records
            ],
            'consent_records': list(subject_consents.values()),
            'generated_at': datetime.now().isoformat()
        }
    
    def _handle_rectification_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data rectification request."""
        # In a real implementation, this would update the data
        return {
            'status': 'pending',
            'message': 'Data rectification request logged for manual review',
            'data_subject_id': data_subject_id
        }
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data erasure request (right to be forgotten)."""
        # Check if erasure is legally possible
        subject_records = [r for r in self.processing_records if r.data_subject_id == data_subject_id]
        
        legal_obligations = []
        for record in subject_records:
            if record.legal_basis in ['legal_obligation', 'public_task']:
                legal_obligations.append(record.processing_purpose)
        
        if legal_obligations:
            return {
                'status': 'rejected',
                'message': 'Erasure not possible due to legal obligations',
                'legal_obligations': legal_obligations
            }
        
        return {
            'status': 'pending',
            'message': 'Data erasure request logged for processing',
            'data_subject_id': data_subject_id
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Generate portable data format
        subject_data = self._handle_access_request(data_subject_id)
        
        return {
            'status': 'completed',
            'data_subject_id': data_subject_id,
            'portable_data': subject_data,
            'format': 'json',
            'generated_at': datetime.now().isoformat()
        }
    
    def run_gdpr_compliance_checks(self) -> List[ComplianceCheck]:
        """Run GDPR compliance checks."""
        checks = []
        
        # Check 1: Data processing records
        checks.append(ComplianceCheck(
            check_id="gdpr_001",
            standard=ComplianceStandard.GDPR,
            requirement="Article 30 - Records of processing activities",
            description="Maintain records of all data processing activities",
            status="pass" if self.processing_records else "fail",
            evidence=[f"Found {len(self.processing_records)} processing records"],
            remediation="Implement data processing logging" if not self.processing_records else ""
        ))
        
        # Check 2: Consent management
        checks.append(ComplianceCheck(
            check_id="gdpr_002",
            standard=ComplianceStandard.GDPR,
            requirement="Article 7 - Conditions for consent",
            description="Proper consent management system",
            status="pass" if self.consent_records else "warning",
            evidence=[f"Found {len(self.consent_records)} consent records"],
            remediation="Implement consent management system" if not self.consent_records else ""
        ))
        
        # Check 3: Data retention policies
        checks.append(ComplianceCheck(
            check_id="gdpr_003",
            standard=ComplianceStandard.GDPR,
            requirement="Article 5 - Storage limitation",
            description="Data retention policies defined",
            status="pass" if self.data_retention_policies else "fail",
            evidence=[f"Defined policies for {len(self.data_retention_policies)} data types"],
            remediation="Define data retention policies" if not self.data_retention_policies else ""
        ))
        
        return checks


class MiFIDIICompliance:
    """MiFID II compliance implementation."""
    
    def __init__(self):
        """Initialize MiFID II compliance."""
        self.transaction_records = []
        self.algorithm_records = {}
        self.best_execution_policies = self._load_best_execution_policies()
        
        logger.info("MiFIDIICompliance initialized")
    
    def _load_best_execution_policies(self) -> Dict[str, Any]:
        """Load best execution policies."""
        return {
            'price_priority': 0.4,
            'speed_priority': 0.3,
            'likelihood_execution': 0.2,
            'settlement_priority': 0.1,
            'venues': ['venue_a', 'venue_b', 'venue_c'],
            'review_frequency': 'quarterly'
        }
    
    def record_transaction(self, client_id: str, instrument: str, 
                          transaction_type: str, quantity: float, 
                          price: float, venue: str, algorithm_used: str = None) -> str:
        """Record MiFID II transaction."""
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.transaction_records)}"
        
        transaction = MiFIDTransaction(
            transaction_id=transaction_id,
            client_id=client_id,
            instrument=instrument,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            venue=venue,
            algorithm_used=algorithm_used,
            decision_maker="algorithm" if algorithm_used else "human"
        )
        
        self.transaction_records.append(transaction)
        logger.info(f"MiFID II transaction recorded: {transaction_id}")
        
        return transaction_id
    
    def record_algorithmic_trading(self, algorithm_id: str, algorithm_type: str,
                                  parameters: Dict[str, Any]) -> str:
        """Record algorithmic trading activity."""
        record_id = f"algo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{algorithm_id}"
        
        algo_record = {
            'record_id': record_id,
            'algorithm_id': algorithm_id,
            'algorithm_type': algorithm_type,
            'parameters': parameters,
            'registered_at': datetime.now(),
            'risk_controls': self._get_risk_controls(algorithm_type),
            'testing_completed': True,  # Should be validated
            'approval_status': 'approved'
        }
        
        self.algorithm_records[record_id] = algo_record
        logger.info(f"Algorithmic trading recorded: {record_id}")
        
        return record_id
    
    def _get_risk_controls(self, algorithm_type: str) -> List[str]:
        """Get required risk controls for algorithm type."""
        base_controls = [
            'maximum_order_size',
            'maximum_daily_volume',
            'price_deviation_limits',
            'kill_switch'
        ]
        
        if algorithm_type == 'high_frequency':
            base_controls.extend([
                'latency_monitoring',
                'order_to_trade_ratio',
                'market_making_obligations'
            ])
        
        return base_controls
    
    def generate_transaction_report(self, start_date: datetime, 
                                   end_date: datetime) -> Dict[str, Any]:
        """Generate MiFID II transaction report."""
        period_transactions = [
            t for t in self.transaction_records 
            if start_date <= t.timestamp <= end_date
        ]
        
        # Calculate execution quality metrics
        total_transactions = len(period_transactions)
        algorithmic_transactions = len([t for t in period_transactions if t.algorithm_used])
        
        venue_distribution = {}
        for transaction in period_transactions:
            venue_distribution[transaction.venue] = venue_distribution.get(transaction.venue, 0) + 1
        
        return {
            'reporting_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_transactions': total_transactions,
                'algorithmic_transactions': algorithmic_transactions,
                'algorithmic_percentage': (algorithmic_transactions / total_transactions * 100) if total_transactions > 0 else 0
            },
            'venue_distribution': venue_distribution,
            'best_execution_analysis': self._analyze_best_execution(period_transactions),
            'generated_at': datetime.now().isoformat()
        }
    
    def _analyze_best_execution(self, transactions: List[MiFIDTransaction]) -> Dict[str, Any]:
        """Analyze best execution compliance."""
        if not transactions:
            return {'status': 'no_data'}
        
        # Simple best execution analysis
        total_value = sum(t.quantity * t.price for t in transactions)
        venue_values = {}
        
        for transaction in transactions:
            venue = transaction.venue
            value = transaction.quantity * transaction.price
            venue_values[venue] = venue_values.get(venue, 0) + value
        
        return {
            'total_value_traded': total_value,
            'venue_value_distribution': venue_values,
            'compliance_status': 'compliant',  # Simplified
            'review_required': False
        }
    
    def run_mifid_compliance_checks(self) -> List[ComplianceCheck]:
        """Run MiFID II compliance checks."""
        checks = []
        
        # Check 1: Transaction reporting
        checks.append(ComplianceCheck(
            check_id="mifid_001",
            standard=ComplianceStandard.MIFID_II,
            requirement="Article 26 - Transaction reporting",
            description="All transactions must be reported",
            status="pass" if self.transaction_records else "fail",
            evidence=[f"Found {len(self.transaction_records)} transaction records"],
            remediation="Implement transaction reporting" if not self.transaction_records else ""
        ))
        
        # Check 2: Algorithmic trading records
        checks.append(ComplianceCheck(
            check_id="mifid_002",
            standard=ComplianceStandard.MIFID_II,
            requirement="Article 17 - Algorithmic trading",
            description="Algorithmic trading must be properly recorded and controlled",
            status="pass" if self.algorithm_records else "warning",
            evidence=[f"Found {len(self.algorithm_records)} algorithm records"],
            remediation="Implement algorithmic trading controls" if not self.algorithm_records else ""
        ))
        
        # Check 3: Best execution policies
        checks.append(ComplianceCheck(
            check_id="mifid_003",
            standard=ComplianceStandard.MIFID_II,
            requirement="Article 27 - Best execution",
            description="Best execution policies must be defined and followed",
            status="pass" if self.best_execution_policies else "fail",
            evidence=[f"Best execution policy defined with {len(self.best_execution_policies)} parameters"],
            remediation="Define best execution policies" if not self.best_execution_policies else ""
        ))
        
        return checks


class SOXCompliance:
    """SOX (Sarbanes-Oxley) compliance implementation."""
    
    def __init__(self):
        """Initialize SOX compliance."""
        self.financial_controls = self._load_financial_controls()
        self.change_logs = []
        self.access_reviews = []
        
        logger.info("SOXCompliance initialized")
    
    def _load_financial_controls(self) -> Dict[str, Any]:
        """Load financial controls framework."""
        return {
            'segregation_of_duties': {
                'enabled': True,
                'roles': ['developer', 'approver', 'deployer'],
                'restrictions': {
                    'same_person_develop_approve': False,
                    'same_person_approve_deploy': False
                }
            },
            'change_management': {
                'approval_required': True,
                'testing_required': True,
                'documentation_required': True,
                'rollback_plan_required': True
            },
            'access_controls': {
                'periodic_review': True,
                'review_frequency': 'quarterly',
                'approval_workflow': True,
                'least_privilege': True
            },
            'audit_trails': {
                'comprehensive_logging': True,
                'log_retention': '7_years',
                'log_integrity': True,
                'regular_review': True
            }
        }
    
    def record_system_change(self, change_id: str, change_type: str, 
                           description: str, requestor: str, approver: str,
                           implementation_date: datetime) -> str:
        """Record system change for SOX compliance."""
        change_record = {
            'change_id': change_id,
            'change_type': change_type,
            'description': description,
            'requestor': requestor,
            'approver': approver,
            'implementation_date': implementation_date,
            'testing_completed': True,  # Should be validated
            'documentation_complete': True,  # Should be validated
            'rollback_plan': True,  # Should be validated
            'recorded_at': datetime.now()
        }
        
        self.change_logs.append(change_record)
        logger.info(f"SOX system change recorded: {change_id}")
        
        return change_id
    
    def conduct_access_review(self, reviewer: str, review_scope: str) -> str:
        """Conduct periodic access review."""
        review_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, this would review actual user access
        review_record = {
            'review_id': review_id,
            'reviewer': reviewer,
            'review_scope': review_scope,
            'review_date': datetime.now(),
            'users_reviewed': 0,  # Would be populated from actual review
            'access_removed': 0,  # Would be populated from actual review
            'exceptions_found': 0,  # Would be populated from actual review
            'status': 'completed'
        }
        
        self.access_reviews.append(review_record)
        logger.info(f"SOX access review completed: {review_id}")
        
        return review_id
    
    def validate_segregation_of_duties(self, user_roles: Dict[str, List[str]]) -> Dict[str, Any]:
        """Validate segregation of duties compliance."""
        violations = []
        
        for user, roles in user_roles.items():
            # Check for conflicting roles
            if 'developer' in roles and 'approver' in roles:
                violations.append({
                    'user': user,
                    'violation': 'Developer cannot be approver',
                    'roles': roles
                })
            
            if 'approver' in roles and 'deployer' in roles:
                violations.append({
                    'user': user,
                    'violation': 'Approver cannot be deployer',
                    'roles': roles
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'users_reviewed': len(user_roles),
            'review_date': datetime.now().isoformat()
        }
    
    def run_sox_compliance_checks(self) -> List[ComplianceCheck]:
        """Run SOX compliance checks."""
        checks = []
        
        # Check 1: Change management
        checks.append(ComplianceCheck(
            check_id="sox_001",
            standard=ComplianceStandard.SOX,
            requirement="Section 404 - Change Management Controls",
            description="All system changes must be properly documented and approved",
            status="pass" if self.change_logs else "fail",
            evidence=[f"Found {len(self.change_logs)} change records"],
            remediation="Implement change management process" if not self.change_logs else ""
        ))
        
        # Check 2: Access reviews
        checks.append(ComplianceCheck(
            check_id="sox_002",
            standard=ComplianceStandard.SOX,
            requirement="Section 404 - Access Control Reviews",
            description="Periodic access reviews must be conducted",
            status="pass" if self.access_reviews else "warning",
            evidence=[f"Found {len(self.access_reviews)} access reviews"],
            remediation="Conduct regular access reviews" if not self.access_reviews else ""
        ))
        
        # Check 3: Financial controls framework
        checks.append(ComplianceCheck(
            check_id="sox_003",
            standard=ComplianceStandard.SOX,
            requirement="Section 404 - Internal Controls",
            description="Financial controls framework must be implemented",
            status="pass" if self.financial_controls else "fail",
            evidence=[f"Financial controls framework with {len(self.financial_controls)} control areas"],
            remediation="Implement financial controls framework" if not self.financial_controls else ""
        ))
        
        return checks


class ComplianceOrchestrator:
    """Orchestrates all compliance frameworks."""
    
    def __init__(self):
        """Initialize compliance orchestrator."""
        self.gdpr = GDPRCompliance()
        self.mifid = MiFIDIICompliance()
        self.sox = SOXCompliance()
        
        logger.info("ComplianceOrchestrator initialized")
    
    def run_all_compliance_checks(self) -> Dict[str, List[ComplianceCheck]]:
        """Run all compliance checks."""
        results = {}
        
        try:
            results['GDPR'] = self.gdpr.run_gdpr_compliance_checks()
        except Exception as e:
            logger.error(f"GDPR compliance check failed: {e}")
            results['GDPR'] = []
        
        try:
            results['MiFID_II'] = self.mifid.run_mifid_compliance_checks()
        except Exception as e:
            logger.error(f"MiFID II compliance check failed: {e}")
            results['MiFID_II'] = []
        
        try:
            results['SOX'] = self.sox.run_sox_compliance_checks()
        except Exception as e:
            logger.error(f"SOX compliance check failed: {e}")
            results['SOX'] = []
        
        return results
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        all_checks = self.run_all_compliance_checks()
        
        # Calculate overall compliance status
        total_checks = sum(len(checks) for checks in all_checks.values())
        passed_checks = sum(
            len([c for c in checks if c.status == 'pass']) 
            for checks in all_checks.values()
        )
        
        compliance_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            'report_generated_at': datetime.now().isoformat(),
            'overall_compliance': {
                'percentage': compliance_percentage,
                'status': 'compliant' if compliance_percentage >= 90 else 'non_compliant',
                'total_checks': total_checks,
                'passed_checks': passed_checks
            },
            'by_standard': {
                standard: {
                    'total_checks': len(checks),
                    'passed_checks': len([c for c in checks if c.status == 'pass']),
                    'failed_checks': len([c for c in checks if c.status == 'fail']),
                    'warnings': len([c for c in checks if c.status == 'warning']),
                    'checks': [
                        {
                            'check_id': c.check_id,
                            'requirement': c.requirement,
                            'status': c.status,
                            'remediation': c.remediation
                        }
                        for c in checks
                    ]
                }
                for standard, checks in all_checks.items()
            }
        }


def create_compliance_system() -> ComplianceOrchestrator:
    """Factory function to create compliance system."""
    return ComplianceOrchestrator()


if __name__ == '__main__':
    # Example usage
    orchestrator = create_compliance_system()
    
    # Record some sample data
    orchestrator.gdpr.record_data_processing(
        data_subject_id="user_123",
        data_categories=["trading_data", "user_profiles"],
        purpose="algorithmic_trading",
        legal_basis="legitimate_interest"
    )
    
    orchestrator.mifid.record_transaction(
        client_id="client_456",
        instrument="EURUSD",
        transaction_type="buy",
        quantity=10000,
        price=1.1234,
        venue="venue_a",
        algorithm_used="momentum_algo"
    )
    
    orchestrator.sox.record_system_change(
        change_id="CHG_001",
        change_type="model_update",
        description="Updated trading algorithm parameters",
        requestor="john.doe",
        approver="jane.smith",
        implementation_date=datetime.now()
    )
    
    # Generate compliance report
    report = orchestrator.generate_compliance_report()
    print(f"Overall compliance: {report['overall_compliance']['percentage']:.1f}%")
    
    for standard, results in report['by_standard'].items():
        print(f"{standard}: {results['passed_checks']}/{results['total_checks']} checks passed")
