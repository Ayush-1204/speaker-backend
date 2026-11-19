"""Clear per-household T_HIGH / T_LOW written by the old auto-calibration.

The pre-fix auto-calibration (calibration.find_eer_threshold) picked thresholds
near the top of the score range and stored them in household_configs, which
overrode the field-tuned global thresholds.  Context-gate settings (quiet hours,
geofence, party mode) are left untouched.

Usage (from the speaker-backend directory, with DATABASE_URL pointing at the DB):
    python -m scripts.reset_household_thresholds            # dry run: list values
    python -m scripts.reset_household_thresholds --apply    # clear them
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reset_household_thresholds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write changes (default is a dry run)")
    args = parser.parse_args()

    from app.database.models import HouseholdConfig
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        rows = db.query(HouseholdConfig).filter(
            (HouseholdConfig.t_high.isnot(None)) | (HouseholdConfig.t_low.isnot(None))
        ).all()
        if not rows:
            logger.info("No households have stored thresholds. Nothing to do.")
            return
        for row in rows:
            logger.info(
                "parent_id=%s | t_high=%s | t_low=%s | calibrated_at=%s | samples=%s",
                row.parent_id, row.t_high, row.t_low, row.calibrated_at, row.calibration_sample_count,
            )
            if args.apply:
                row.t_high = None
                row.t_low = None
                row.calibrated_at = None
                row.calibration_sample_count = None
        if args.apply:
            db.commit()
            logger.info("Cleared thresholds for %d household(s).", len(rows))
        else:
            logger.info("Dry run: %d household(s) would be cleared. Re-run with --apply.", len(rows))
    finally:
        db.close()


if __name__ == "__main__":
    main()
