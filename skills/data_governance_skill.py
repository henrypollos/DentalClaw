"""
Data Governance Skill for DentalClaw
=====================================

Comprehensive data quality auditing for dental imaging datasets.
Covers 6 major categories of data quality checks.

Author: DentalClaw Team
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_erosion, label


# Handle imports based on how the module is being run
try:
    from skills.base_skill import BaseSkill
except ImportError:
    try:
        from base_skill import BaseSkill
    except ImportError:
        # If base_skill is not available, create a minimal BaseSkill
        class BaseSkill:
            """Minimal BaseSkill stub for compatibility."""
            pass
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class DataQualityIssue:
    """Represents a single data quality issue."""
    case_id: str
    issue_category: str
    issue_type: str
    severity: str
    file_path: str
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditStats:
    """Statistics from the audit."""
    total_cases: int = 0
    total_issues: int = 0
    file_integrity_issues: int = 0
    pairing_issues: int = 0
    annotation_completeness_issues: int = 0
    labeling_standards_issues: int = 0
    structural_reasonableness_issues: int = 0
    split_integrity_issues: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Data Governance Auditor
# ==============================================================================

class DataGovernanceAuditor(BaseSkill):
    """
    Comprehensive data quality auditor for dental imaging datasets.
    
    Performs 6 categories of quality checks:
    1. File Integrity - Check file presence and readability
    2. Image-Annotation Pairing - Verify matching and consistency
    3. Annotation Completeness - Validate mask content
    4. Labeling Standards - Check tooth numbering
    5. Structural Reasonableness - Detect artifacts
    6. Split Integrity - Prevent train-test leakage
    """
    
    def __init__(self,
                 dataset_path: str,
                 image_subdir: str = 'Radiographs',
                 mask_subdir: str = 'Segmentation/teeth_mask',
                 valid_labels: Optional[Set[int]] = None,
                 min_component_size: int = 5,
                 output_dir: Optional[str] = None):
        """
        Initialize the auditor.
        
        Args:
            dataset_path: Root path to the dataset
            image_subdir: Subdirectory containing images
            mask_subdir: Subdirectory containing masks
            valid_labels: Set of valid label values (default: 0-48)
            min_component_size: Minimum pixels for a valid component
            output_dir: Directory to save reports
        """
        super().__init__()
        
        self.dataset_path = Path(dataset_path)
        self.image_dir = self.dataset_path / image_subdir
        self.mask_dir = self.dataset_path / mask_subdir
        self.output_dir = Path(output_dir) if output_dir else self.dataset_path / 'audit_results'
        
        self.valid_labels = valid_labels or set(range(0, 49))
        self.min_component_size = min_component_size
        
        self.issues: List[DataQualityIssue] = []
        self.case_mapping: Dict[str, Tuple[str, str]] = {}
        self.stats = AuditStats()
        
        logger.info(f"Initialized DataGovernanceAuditor for {self.dataset_path}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _build_case_mapping(self) -> Dict[str, Tuple[str, str]]:
        """Build case_id -> (image_file, mask_file) mapping."""
        logger.info("Building case mapping...")
        
        image_files = list(self.image_dir.glob('*')) if self.image_dir.exists() else []
        mask_files = list(self.mask_dir.glob('*')) if self.mask_dir.exists() else []
        
        image_cases = {}
        for f in image_files:
            if f.is_file():
                case_id = f.stem.lower()
                image_cases[case_id] = f.name
        
        mask_cases = {}
        for f in mask_files:
            if f.is_file():
                case_id = f.stem.lower()
                mask_cases[case_id] = f.name
        
        matching = {}
        for case_id in image_cases:
            if case_id in mask_cases:
                matching[case_id] = (image_cases[case_id], mask_cases[case_id])
        
        logger.info(f"Found {len(image_cases)} images, {len(mask_cases)} masks, {len(matching)} matching pairs")
        self.case_mapping = matching
        return matching
    
    def _log_issue(self, case_id: str, category: str, issue_type: str,
                   severity: str, file_path: str, description: str):
        """Log a data quality issue."""
        issue = DataQualityIssue(
            case_id=case_id,
            issue_category=category,
            issue_type=issue_type,
            severity=severity,
            file_path=file_path,
            description=description
        )
        self.issues.append(issue)
        
        # Update stats
        self.stats.total_issues += 1
        if category == 'file_integrity':
            self.stats.file_integrity_issues += 1
        elif category == 'pairing':
            self.stats.pairing_issues += 1
        elif category == 'annotation_completeness':
            self.stats.annotation_completeness_issues += 1
        elif category == 'labeling_standards':
            self.stats.labeling_standards_issues += 1
        elif category == 'structural_reasonableness':
            self.stats.structural_reasonableness_issues += 1
        elif category == 'split_integrity':
            self.stats.split_integrity_issues += 1
    
    # =========================================================================
    # (1) FILE INTEGRITY CHECKS
    # =========================================================================
    
    def check_file_integrity(self):
        """Check if files are present and readable."""
        logger.info("=" * 80)
        logger.info("CHECK 1: FILE INTEGRITY")
        logger.info("=" * 80)
        
        self._build_case_mapping()
        
        # Check for images without masks
        all_case_ids = set(self.case_mapping.keys())
        image_files = list(self.image_dir.glob('*')) if self.image_dir.exists() else []
        
        for img_file in image_files:
            if not img_file.is_file():
                continue
            case_id = img_file.stem.lower()
            if case_id not in all_case_ids:
                self._log_issue(
                    case_id=case_id,
                    category='file_integrity',
                    issue_type='missing_annotation_mask',
                    severity='critical',
                    file_path=str(img_file),
                    description=f"Image {img_file.name} has no corresponding mask"
                )
        
        # Check readability
        for case_id, (img_file, mask_file) in self.case_mapping.items():
            img_path = self.image_dir / img_file
            mask_path = self.mask_dir / mask_file
            
            # Check image readability
            try:
                img = Image.open(img_path)
                img.load()
            except (UnidentifiedImageError, OSError, Exception) as e:
                self._log_issue(
                    case_id=case_id,
                    category='file_integrity',
                    issue_type='image_unreadable',
                    severity='critical',
                    file_path=str(img_path),
                    description=f"Cannot read image file: {str(e)[:80]}"
                )
            
            # Check mask readability
            try:
                mask = Image.open(mask_path)
                mask.load()
            except (UnidentifiedImageError, OSError, Exception) as e:
                self._log_issue(
                    case_id=case_id,
                    category='file_integrity',
                    issue_type='mask_unreadable',
                    severity='critical',
                    file_path=str(mask_path),
                    description=f"Cannot read mask file: {str(e)[:80]}"
                )
        
        logger.info(f"File integrity check complete.")
    
    # =========================================================================
    # (2) IMAGE-ANNOTATION PAIRING CHECKS
    # =========================================================================
    
    def check_image_annotation_pairing(self):
        """Verify image and mask pairing and consistency."""
        logger.info("=" * 80)
        logger.info("CHECK 2: IMAGE-ANNOTATION PAIRING")
        logger.info("=" * 80)
        
        for case_id, (img_file, mask_file) in self.case_mapping.items():
            img_path = self.image_dir / img_file
            mask_path = self.mask_dir / mask_file
            
            # Get dimensions
            try:
                img = Image.open(img_path)
                img.load()
                img_size = img.size
            except:
                continue
            
            try:
                mask = Image.open(mask_path)
                mask.load()
                mask_size = mask.size
            except:
                continue
            
            # Check dimension match
            if img_size != mask_size:
                self._log_issue(
                    case_id=case_id,
                    category='pairing',
                    issue_type='dimension_mismatch',
                    severity='critical',
                    file_path=str(mask_path),
                    description=f"Image size {img_size} != mask size {mask_size}"
                )
        
        logger.info(f"Image-annotation pairing check complete.")
    
    # =========================================================================
    # (3) ANNOTATION COMPLETENESS CHECKS
    # =========================================================================
    
    def check_annotation_completeness(self):
        """Verify mask content and label validity."""
        logger.info("=" * 80)
        logger.info("CHECK 3: ANNOTATION COMPLETENESS")
        logger.info("=" * 80)
        
        for case_id, (img_file, mask_file) in self.case_mapping.items():
            mask_path = self.mask_dir / mask_file
            
            try:
                mask = Image.open(mask_path)
                mask.load()
                mask_data = np.array(mask)
            except:
                continue
            
            # Check if mask is empty
            if mask_data.max() == 0:
                self._log_issue(
                    case_id=case_id,
                    category='annotation_completeness',
                    issue_type='mask_empty',
                    severity='warning',
                    file_path=str(mask_path),
                    description="Mask is empty (all zeros)"
                )
                continue
            
            # Check for invalid label values
            unique_labels = set(np.unique(mask_data))
            invalid_labels = unique_labels - self.valid_labels
            
            if invalid_labels:
                self._log_issue(
                    case_id=case_id,
                    category='annotation_completeness',
                    issue_type='invalid_label_value',
                    severity='critical',
                    file_path=str(mask_path),
                    description=f"Found invalid labels: {invalid_labels}"
                )
            
            # Check for duplicated label regions (same label in multiple disconnected areas)
            for label_val in unique_labels:
                if label_val == 0:
                    continue
                
                label_mask = (mask_data == label_val)
                labeled_array, num_features = label(label_mask)
                
                if num_features > 1:
                    self._log_issue(
                        case_id=case_id,
                        category='annotation_completeness',
                        issue_type='duplicated_label_regions',
                        severity='warning',
                        file_path=str(mask_path),
                        description=f"Label {label_val} in {num_features} disconnected regions"
                    )
        
        logger.info(f"Annotation completeness check complete.")
    
    # =========================================================================
    # (4) LABELING STANDARDS CHECKS
    # =========================================================================
    
    def check_labeling_standards(self):
        """Verify tooth numbering and class schema consistency."""
        logger.info("=" * 80)
        logger.info("CHECK 4: LABELING STANDARDS / TOOTH NUMBERING")
        logger.info("=" * 80)
        
        all_present_labels = set()
        
        for case_id, (img_file, mask_file) in self.case_mapping.items():
            mask_path = self.mask_dir / mask_file
            
            try:
                mask = Image.open(mask_path)
                mask.load()
                mask_data = np.array(mask)
                unique_labels = set(np.unique(mask_data))
                all_present_labels.update(unique_labels)
            except:
                continue
        
        all_present_labels.discard(0)  # Remove background
        
        # Check completeness
        expected_labels = set(range(1, 49))
        missing_labels = expected_labels - all_present_labels
        
        if missing_labels and len(missing_labels) > 5:
            self._log_issue(
                case_id='DATASET',
                category='labeling_standards',
                issue_type='incomplete_tooth_schema',
                severity='warning',
                file_path=str(self.mask_dir),
                description=f"Missing {len(missing_labels)} tooth labels"
            )
        
        # Check for unexpected labels
        unexpected_labels = all_present_labels - expected_labels
        if unexpected_labels:
            self._log_issue(
                case_id='DATASET',
                category='labeling_standards',
                issue_type='unexpected_labels',
                severity='warning',
                file_path=str(self.mask_dir),
                description=f"Unexpected labels found: {unexpected_labels}"
            )
        
        logger.info(f"Labeling standards check complete.")
    
    # =========================================================================
    # (5) STRUCTURAL REASONABLENESS CHECKS
    # =========================================================================
    
    def check_structural_reasonableness(self):
        """Detect small fragments and unreasonable structures."""
        logger.info("=" * 80)
        logger.info("CHECK 5: STRUCTURAL REASONABLENESS")
        logger.info("=" * 80)
        
        for case_id, (img_file, mask_file) in self.case_mapping.items():
            mask_path = self.mask_dir / mask_file
            
            try:
                mask = Image.open(mask_path)
                mask.load()
                mask_data = np.array(mask)
            except:
                continue
            
            if mask_data.max() == 0:
                continue
            
            # Check for small fragments
            for label_val in np.unique(mask_data):
                if label_val == 0:
                    continue
                
                label_mask = (mask_data == label_val)
                labeled_array, num_features = label(label_mask)
                
                for feature_id in range(1, num_features + 1):
                    component = (labeled_array == feature_id)
                    component_size = component.sum()
                    
                    if component_size < self.min_component_size:
                        self._log_issue(
                            case_id=case_id,
                            category='structural_reasonableness',
                            issue_type='small_disconnected_components',
                            severity='warning',
                            file_path=str(mask_path),
                            description=f"Label {label_val} has fragment ({component_size} < {self.min_component_size} pixels)"
                        )
                        break
        
        logger.info(f"Structural reasonableness check complete.")
    
    # =========================================================================
    # (6) SPLIT INTEGRITY CHECKS
    # =========================================================================
    
    def check_split_integrity(self):
        """Check for train-test leakage and split consistency."""
        logger.info("=" * 80)
        logger.info("CHECK 6: SPLIT INTEGRITY")
        logger.info("=" * 80)
        
        train_path = self.dataset_path / 'train.txt'
        val_path = self.dataset_path / 'val.txt'
        test_path = self.dataset_path / 'test.txt'
        
        splits = {
            'train': train_path,
            'val': val_path,
            'test': test_path
        }
        
        split_cases = {}
        
        for split_name, split_file in splits.items():
            if not split_file.exists():
                logger.info(f"Split file {split_name}.txt not found")
                continue
            
            with open(split_file) as f:
                cases = {line.strip() for line in f if line.strip()}
            
            split_cases[split_name] = cases
            
            # Verify all cases exist
            for case_id in cases:
                if case_id.lower() not in self.case_mapping:
                    self._log_issue(
                        case_id=case_id,
                        category='split_integrity',
                        issue_type='case_not_in_dataset',
                        severity='critical',
                        file_path=str(split_file),
                        description=f"Case in {split_name} but not in dataset"
                    )
        
        # Check for overlaps
        if 'train' in split_cases and 'val' in split_cases:
            overlap = split_cases['train'] & split_cases['val']
            for case_id in overlap:
                self._log_issue(
                    case_id=case_id,
                    category='split_integrity',
                    issue_type='split_leakage',
                    severity='critical',
                    file_path=str(self.dataset_path),
                    description="Case in both train and val splits"
                )
        
        if 'train' in split_cases and 'test' in split_cases:
            overlap = split_cases['train'] & split_cases['test']
            for case_id in overlap:
                self._log_issue(
                    case_id=case_id,
                    category='split_integrity',
                    issue_type='split_leakage',
                    severity='critical',
                    file_path=str(self.dataset_path),
                    description="Case in both train and test splits"
                )
        
        logger.info(f"Split integrity check complete.")
    
    # =========================================================================
    # MAIN AUDIT METHOD
    # =========================================================================
    
    def run_full_audit(self):
        """Execute all 6 data quality checks."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPREHENSIVE DATA QUALITY AUDIT")
        logger.info("=" * 80 + "\n")
        
        self.stats.total_cases = len(self.case_mapping) if self.case_mapping else 0
        
        self.check_file_integrity()
        self.check_image_annotation_pairing()
        self.check_annotation_completeness()
        self.check_labeling_standards()
        self.check_structural_reasonableness()
        self.check_split_integrity()
        
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT COMPLETE")
        logger.info("=" * 80 + "\n")
    
    def save_reports(self):
        """Save audit reports to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        report_data = {
            'dataset': str(self.dataset_path),
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats.to_dict(),
            'issues': [issue.to_dict() for issue in self.issues]
        }
        
        json_path = self.output_dir / 'audit_report.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Report saved to {json_path}")
        
        # Save as CSV
        if self.issues:
            issues_df = pd.DataFrame([issue.to_dict() for issue in self.issues])
            csv_path = self.output_dir / 'audit_issues.csv'
            issues_df.to_csv(csv_path, index=False)
            logger.info(f"Issues saved to {csv_path}")
        
        return json_path
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 100)
        print("DATA QUALITY AUDIT SUMMARY")
        print("=" * 100)
        print(f"\nDataset: {self.dataset_path}")
        print(f"Total Cases: {self.stats.total_cases}")
        print(f"Total Issues: {self.stats.total_issues}")
        print(f"\nIssues by Category:")
        print(f"  1. File Integrity: {self.stats.file_integrity_issues}")
        print(f"  2. Image-Annotation Pairing: {self.stats.pairing_issues}")
        print(f"  3. Annotation Completeness: {self.stats.annotation_completeness_issues}")
        print(f"  4. Labeling Standards: {self.stats.labeling_standards_issues}")
        print(f"  5. Structural Reasonableness: {self.stats.structural_reasonableness_issues}")
        print(f"  6. Split Integrity: {self.stats.split_integrity_issues}")
        print("=" * 100 + "\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        dataset_path = "/data/data2/yiyang/TDD"
    else:
        dataset_path = sys.argv[1]
    
    auditor = DataGovernanceAuditor(
        dataset_path=dataset_path,
        image_subdir='Radiographs',
        mask_subdir='Segmentation/teeth_mask',
        output_dir=f"{dataset_path}/audit_results"
    )
    
    auditor.run_full_audit()
    auditor.print_summary()
    auditor.save_reports()