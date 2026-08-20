import argparse
import sys

import pandas as pd
import pandera as pa
from pandera import Check


DRIFT_THRESHOLD = 0.20  # 20% relative drift on selected numeric means
NUMERIC_DRIFT_COLUMNS = ["age", "income", "tenure_months"]


class IncomingSchema(pa.DataFrameModel):
    customer_id: pa.typing.Series[int] = pa.Field(gt=0)
    age: pa.typing.Series[int] = pa.Field(ge=18, le=100)
    income: pa.typing.Series[float] = pa.Field(ge=0, le=1_000_000)
    tenure_months: pa.typing.Series[int] = pa.Field(ge=0, le=600)
    has_defaulted: pa.typing.Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True


def validate_schema(df: pd.DataFrame) -> None:
    """Raises if schema validation fails."""
    IncomingSchema.validate(df, lazy=True)


def relative_drift(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else 1.0
    return abs(current - baseline) / abs(baseline)


def validate_statistical_bounds(
    baseline_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    drift_threshold: float = DRIFT_THRESHOLD,
) -> None:
    """Raises ValueError if simple drift checks exceed threshold."""
    failures = []

    for col in NUMERIC_DRIFT_COLUMNS:
        baseline_mean = float(baseline_df[col].mean())
        incoming_mean = float(incoming_df[col].mean())
        drift = relative_drift(incoming_mean, baseline_mean)

        if drift > drift_threshold:
            failures.append(
                {
                    "column": col,
                    "baseline_mean": baseline_mean,
                    "incoming_mean": incoming_mean,
                    "relative_drift": drift,
                    "threshold": drift_threshold,
                }
            )

    if failures:
        detail = "\n".join(
            [
                f"- {f['column']}: baseline_mean={f['baseline_mean']:.3f}, "
                f"incoming_mean={f['incoming_mean']:.3f}, "
                f"relative_drift={f['relative_drift']:.3f} > threshold={f['threshold']:.3f}"
                for f in failures
            ]
        )
        raise ValueError("Statistical drift check failed:\n" + detail)


def run_quality_gate(baseline_path: str, incoming_path: str) -> None:
    baseline_df = pd.read_csv(baseline_path)
    incoming_df = pd.read_csv(incoming_path)

    # 1) schema/type/range/null checks
    validate_schema(incoming_df)

    # 2) simple statistical drift checks
    validate_statistical_bounds(baseline_df, incoming_df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime data quality gating demo")
    parser.add_argument("--baseline", required=True, help="Path to baseline CSV")
    parser.add_argument("--incoming", required=True, help="Path to incoming CSV")
    args = parser.parse_args()

    print("[INFO] Running data quality gate...")
    print(f"[INFO] Baseline file: {args.baseline}")
    print(f"[INFO] Incoming file: {args.incoming}")

    try:
        run_quality_gate(args.baseline, args.incoming)
    except pa.errors.SchemaErrors as e:
        print("[BLOCKED] Schema validation failed. Pipeline execution is stopped.")
        print("\nFailure cases:")
        print(e.failure_cases)
        return 1
    except Exception as e:
        print("[BLOCKED] Quality gate failed. Pipeline execution is stopped.")
        print(str(e))
        return 1

    print("[PASSED] Quality gate passed. Pipeline may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
