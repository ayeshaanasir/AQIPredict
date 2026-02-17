from feature_pipeline import run_feature_pipeline
from datetime import datetime, timedelta, timezone
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backfill_historical_data(months: int = 6) -> None:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30 * months)

    current_date = start_date
    chunk_size = 5  
    total_chunks = int((end_date - start_date).days / chunk_size) + 1
    processed = 0

    logger.info(f"Starting backfill: {start_date.date()} → {end_date.date()} ({months} months)")
    logger.info(f"Estimated chunks: {total_chunks}")

    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=chunk_size), end_date)

        logger.info(f"\n{'='*60}")
        logger.info(f"Chunk {processed + 1}/{total_chunks}: {current_date.date()} → {chunk_end.date()}")
        logger.info(f"{'='*60}")

        try:
            run_feature_pipeline(current_date, chunk_end)
            logger.info("✓ Chunk processed successfully")
            processed += 1
        except Exception as e:
            logger.error(f"✗ Error processing chunk: {e}")

        current_date = chunk_end + timedelta(hours=1)

        time.sleep(2)

    logger.info(f"\n✓ Backfill completed! Processed {processed}/{total_chunks} chunks.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill historical AQI data")
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Number of months to backfill (default: 1)"
    )
    args = parser.parse_args()

    backfill_historical_data(months=args.months)
