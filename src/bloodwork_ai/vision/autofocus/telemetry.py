from __future__ import annotations

import csv
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


@dataclass
class TelemetryEvent:
    """Single telemetry event for autofocus operations."""

    # Core identifiers
    timestamp: float
    tile_id: str
    session_id: Optional[str] = None

    # Spatial coordinates
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    z_predicted_um: Optional[float] = None
    z_af_um: Optional[float] = None

    # Performance metrics
    elapsed_ms: Optional[float] = None
    metric_values: Optional[Dict[str, float]] = None
    sample_count: Optional[int] = None

    # System state
    temperature_c: Optional[float] = None
    illumination_profile: Optional[str] = None
    led_currents: Optional[List[float]] = None
    exposure_ms: Optional[float] = None

    # Quality indicators
    surface_residual_um: Optional[float] = None
    focus_error_estimate_um: Optional[float] = None
    metric_snr: Optional[float] = None

    # Status and flags
    status: str = "unknown"  # success, failed, timeout, error
    flags: List[str] = field(default_factory=list)
    algorithm_version: str = "1.0"
    config_hash: Optional[str] = None

    # Regulatory traceability
    operator_id: Optional[str] = None
    instrument_id: Optional[str] = None
    software_version: Optional[str] = None

    def __post_init__(self):
        if self.session_id is None:
            # Generate session ID based on timestamp and some context
            session_data = f"{self.timestamp}_{self.tile_id}"
            self.session_id = hashlib.md5(session_data.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_csv_row(self) -> Dict[str, str]:
        """Convert to CSV-friendly row."""
        data = self.to_dict()
        # Convert complex fields to JSON strings
        if data.get('metric_values'):
            data['metric_values'] = json.dumps(data['metric_values'])
        if data.get('led_currents'):
            data['led_currents'] = json.dumps(data['led_currents'])
        if data.get('flags'):
            data['flags'] = json.dumps(data['flags'])

        # Convert all values to strings
        return {k: str(v) if v is not None else '' for k, v in data.items()}


class ProductionTelemetryLogger:
    """Production-grade telemetry logging system for autofocus operations."""

    def __init__(self,
                 output_dir: Union[str, Path],
                 enable_csv: bool = True,
                 enable_sqlite: bool = True,
                 enable_json: bool = False,
                 max_memory_events: int = 10000,
                 flush_interval_s: float = 60.0):
        """Initialize telemetry logger.

        Args:
            output_dir: Directory for telemetry files
            enable_csv: Enable CSV output
            enable_sqlite: Enable SQLite database
            enable_json: Enable JSON line output
            max_memory_events: Max events to keep in memory
            flush_interval_s: Automatic flush interval
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_csv = enable_csv
        self.enable_sqlite = enable_sqlite and HAS_SQLITE
        self.enable_json = enable_json
        self.max_memory_events = max_memory_events
        self.flush_interval_s = flush_interval_s

        # In-memory buffer
        self.events: List[TelemetryEvent] = []
        self.lock = threading.Lock()

        # File handles
        self._csv_file = None
        self._csv_writer = None
        self._json_file = None
        self._sqlite_conn = None

        # Timing
        self._last_flush = time.time()

        # Initialize outputs
        self._initialize_outputs()

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

    def _initialize_outputs(self):
        """Initialize output files and databases."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.enable_csv:
            csv_path = self.output_dir / f"autofocus_telemetry_{timestamp}.csv"
            self._csv_file = open(csv_path, 'w', newline='')
            # CSV writer will be created when first event is logged

        if self.enable_json:
            json_path = self.output_dir / f"autofocus_telemetry_{timestamp}.jsonl"
            self._json_file = open(json_path, 'w')

        if self.enable_sqlite:
            db_path = self.output_dir / f"autofocus_telemetry_{timestamp}.db"
            self._sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._create_sqlite_schema()

    def _create_sqlite_schema(self):
        """Create SQLite database schema."""
        if not self._sqlite_conn:
            return

        schema = '''
        CREATE TABLE IF NOT EXISTS autofocus_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            tile_id TEXT,
            session_id TEXT,
            x_um REAL,
            y_um REAL,
            z_predicted_um REAL,
            z_af_um REAL,
            elapsed_ms REAL,
            temperature_c REAL,
            illumination_profile TEXT,
            exposure_ms REAL,
            surface_residual_um REAL,
            focus_error_estimate_um REAL,
            metric_snr REAL,
            status TEXT,
            flags TEXT,
            algorithm_version TEXT,
            config_hash TEXT,
            operator_id TEXT,
            instrument_id TEXT,
            software_version TEXT,
            metric_values_json TEXT,
            led_currents_json TEXT,
            sample_count INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_timestamp ON autofocus_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tile_id ON autofocus_events(tile_id);
        CREATE INDEX IF NOT EXISTS idx_session_id ON autofocus_events(session_id);
        CREATE INDEX IF NOT EXISTS idx_status ON autofocus_events(status);
        '''

        self._sqlite_conn.executescript(schema)
        self._sqlite_conn.commit()

    def log_event(self, event: TelemetryEvent) -> None:
        """Log a telemetry event."""
        with self.lock:
            self.events.append(event)

            # Limit memory usage
            if len(self.events) > self.max_memory_events:
                self.events = self.events[-self.max_memory_events//2:]

            # Check if we need to flush
            if (time.time() - self._last_flush) > self.flush_interval_s:
                self._flush_to_disk()

    def log_autofocus_operation(self,
                               tile_id: str,
                               x_um: float,
                               y_um: float,
                               z_af_um: float,
                               elapsed_ms: float,
                               status: str = "success",
                               **kwargs) -> None:
        """Convenience method to log autofocus operation."""
        event = TelemetryEvent(
            timestamp=time.time(),
            tile_id=tile_id,
            x_um=x_um,
            y_um=y_um,
            z_af_um=z_af_um,
            elapsed_ms=elapsed_ms,
            status=status,
            **kwargs
        )
        self.log_event(event)

    def _flush_to_disk(self) -> None:
        """Flush events to disk."""
        if not self.events:
            return

        events_to_flush = self.events.copy()

        # Write to CSV
        if self.enable_csv and self._csv_file:
            self._write_csv_events(events_to_flush)

        # Write to JSON Lines
        if self.enable_json and self._json_file:
            self._write_json_events(events_to_flush)

        # Write to SQLite
        if self.enable_sqlite and self._sqlite_conn:
            self._write_sqlite_events(events_to_flush)

        self._last_flush = time.time()

    def _write_csv_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to CSV file."""
        if not events:
            return

        # Initialize CSV writer with first event's fields
        if self._csv_writer is None:
            fieldnames = list(events[0].to_csv_row().keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()

        for event in events:
            self._csv_writer.writerow(event.to_csv_row())

        self._csv_file.flush()

    def _write_json_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to JSON Lines file."""
        for event in events:
            json.dump(event.to_dict(), self._json_file)
            self._json_file.write('\n')

        self._json_file.flush()

    def _write_sqlite_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to SQLite database."""
        for event in events:
            data = event.to_dict()

            # Convert complex fields to JSON
            data['metric_values_json'] = (json.dumps(data.pop('metric_values', None))
                                        if data.get('metric_values') else None)
            data['led_currents_json'] = (json.dumps(data.pop('led_currents', None))
                                       if data.get('led_currents') else None)
            data['flags'] = json.dumps(data.get('flags', []))

            # Filter data to only include columns that exist in the schema
            schema_columns = {
                'timestamp', 'tile_id', 'session_id', 'x_um', 'y_um',
                'z_predicted_um', 'z_af_um', 'elapsed_ms', 'temperature_c',
                'illumination_profile', 'exposure_ms', 'surface_residual_um',
                'focus_error_estimate_um', 'metric_snr', 'status', 'flags',
                'algorithm_version', 'config_hash', 'operator_id', 'instrument_id',
                'software_version', 'metric_values_json', 'led_currents_json', 'sample_count'
            }

            filtered_data = {k: v for k, v in data.items() if k in schema_columns}

            # Insert into database
            columns = ', '.join(filtered_data.keys())
            placeholders = ', '.join(['?'] * len(filtered_data))
            query = f"INSERT INTO autofocus_events ({columns}) VALUES ({placeholders})"

            self._sqlite_conn.execute(query, list(filtered_data.values()))

        self._sqlite_conn.commit()

    def _background_flush(self) -> None:
        """Background thread for periodic flushing."""
        while True:
            time.sleep(self.flush_interval_s)
            with self.lock:
                self._flush_to_disk()

    def get_statistics(self, time_window_s: Optional[float] = None) -> Dict[str, Any]:
        """Get telemetry statistics.

        Args:
            time_window_s: Time window for statistics (None for all data)

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            if time_window_s:
                cutoff_time = time.time() - time_window_s
                events = [e for e in self.events if e.timestamp >= cutoff_time]
            else:
                events = self.events

        if not events:
            return {"status": "no_data", "event_count": 0}

        # Basic statistics
        elapsed_times = [e.elapsed_ms for e in events if e.elapsed_ms is not None]
        success_count = sum(1 for e in events if e.status == "success")

        # Performance statistics
        import numpy as np

        stats = {
            "event_count": len(events),
            "success_rate": success_count / len(events) if events else 0,
            "time_window_s": time_window_s,
        }

        if elapsed_times:
            stats.update({
                "avg_elapsed_ms": float(np.mean(elapsed_times)),
                "median_elapsed_ms": float(np.median(elapsed_times)),
                "p95_elapsed_ms": float(np.percentile(elapsed_times, 95)),
                "p99_elapsed_ms": float(np.percentile(elapsed_times, 99)),
                "max_elapsed_ms": float(np.max(elapsed_times)),
                "throughput_target_met": float(np.percentile(elapsed_times, 95)) <= 150.0
            })

        # Temperature statistics
        temperatures = [e.temperature_c for e in events if e.temperature_c is not None]
        if temperatures:
            stats.update({
                "avg_temperature_c": float(np.mean(temperatures)),
                "temperature_range_c": float(np.ptp(temperatures))
            })

        # Focus accuracy statistics
        focus_errors = [e.focus_error_estimate_um for e in events
                       if e.focus_error_estimate_um is not None]
        if focus_errors:
            stats.update({
                "avg_focus_error_um": float(np.mean(focus_errors)),
                "p95_focus_error_um": float(np.percentile(focus_errors, 95))
            })

        return stats

    def export_data(self,
                   output_path: Union[str, Path],
                   format: str = "csv",
                   time_range: Optional[Tuple[float, float]] = None) -> None:
        """Export telemetry data to file.

        Args:
            output_path: Output file path
            format: Export format ("csv", "json", "excel")
            time_range: Optional (start_time, end_time) filter
        """
        with self.lock:
            events = self.events.copy()

        if time_range:
            start_time, end_time = time_range
            events = [e for e in events if start_time <= e.timestamp <= end_time]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "csv":
            self._export_csv(events, output_path)
        elif format.lower() == "json":
            self._export_json(events, output_path)
        elif format.lower() == "excel":
            self._export_excel(events, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, events: List[TelemetryEvent], path: Path) -> None:
        """Export events to CSV."""
        if not events:
            return

        with open(path, 'w', newline='') as f:
            fieldnames = list(events[0].to_csv_row().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow(event.to_csv_row())

    def _export_json(self, events: List[TelemetryEvent], path: Path) -> None:
        """Export events to JSON."""
        data = [event.to_dict() for event in events]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_excel(self, events: List[TelemetryEvent], path: Path) -> None:
        """Export events to Excel (requires pandas and openpyxl)."""
        try:
            import pandas as pd

            data = [event.to_dict() for event in events]
            df = pd.DataFrame(data)
            df.to_excel(path, index=False)
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export")

    def close(self) -> None:
        """Close telemetry logger and flush remaining data."""
        with self.lock:
            self._flush_to_disk()

        if self._csv_file:
            self._csv_file.close()
        if self._json_file:
            self._json_file.close()
        if self._sqlite_conn:
            self._sqlite_conn.close()


class RegulatoryLogger:
    """Specialized logger for regulatory compliance and traceability."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create separate log for regulatory events
        self.audit_log_path = self.output_dir / "regulatory_audit.log"
        self.audit_file = open(self.audit_log_path, 'a')

    def log_regulatory_event(self,
                           event_type: str,
                           tile_id: str,
                           details: Dict[str, Any],
                           operator_id: Optional[str] = None) -> None:
        """Log regulatory event with full traceability."""
        timestamp = datetime.now().isoformat()

        audit_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "tile_id": tile_id,
            "operator_id": operator_id,
            "details": details,
            "software_version": "1.0",  # Should come from actual version
            "instrument_id": "INSTR_001"  # Should come from system config
        }

        # Write to audit log
        self.audit_file.write(json.dumps(audit_entry) + '\n')
        self.audit_file.flush()

    def log_autofocus_decision(self,
                             tile_id: str,
                             z_af_um: float,
                             confidence_metrics: Dict[str, float],
                             operator_id: Optional[str] = None) -> None:
        """Log autofocus decision for regulatory traceability."""
        self.log_regulatory_event(
            event_type="AUTOFOCUS_DECISION",
            tile_id=tile_id,
            details={
                "focus_position_um": z_af_um,
                "confidence_metrics": confidence_metrics,
                "decision_algorithm": "BloodSmearAutofocus_v1.0"
            },
            operator_id=operator_id
        )

    def close(self) -> None:
        """Close regulatory logger."""
        if self.audit_file:
            self.audit_file.close()