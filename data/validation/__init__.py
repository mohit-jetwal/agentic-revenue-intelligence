"""Dataset validation.

Two layers, answering two different questions:

* ``checks`` - *is the data internally consistent?* Business invariants like the
  inventory identity and the revenue identity. Failures here mean a generator bug.
* ``statistical`` - *does the data contain the relationships it was built to
  contain?* Elasticity recovery, promotion uplift, cross-price signs, stockout
  censoring. Failures here mean the data is internally consistent but useless
  for modelling, which is the more dangerous outcome because nothing else would
  reveal it.

``report`` composes both into a markdown report and a JSON summary.
"""

from data.validation.checks import CheckResult, CheckSuite, Severity, run_all_checks
from data.validation.report import ValidationReport, validate_dataset, write_report

__all__ = [
    "CheckResult",
    "CheckSuite",
    "Severity",
    "ValidationReport",
    "run_all_checks",
    "validate_dataset",
    "write_report",
]
