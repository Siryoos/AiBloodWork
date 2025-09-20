# Deployment Guide

## Overview

This comprehensive guide covers deploying the dual-lens autofocus system in production hematology environments. Learn how to properly configure, deploy, monitor, and maintain the system for reliable operation.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Hardware Integration](#hardware-integration)
6. [Calibration Procedures](#calibration-procedures)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Backup & Recovery](#backup--recovery)

## System Requirements

### Minimum Hardware Requirements

#### Computing Platform
- **CPU**: Intel i5-8000 series or AMD Ryzen 5 3000 series (minimum)
- **RAM**: 8GB DDR4 (16GB recommended for high-throughput)
- **Storage**: 100GB available SSD space
- **GPU**: Optional, improves focus processing performance
- **Network**: Gigabit Ethernet (for telemetry and remote monitoring)

#### Camera System
- **Dual-lens microscope** with motorized lens turret
- **Focus motor**: Sub-micrometer precision, <100ms settling time
- **Camera**: Minimum 2MP, preferably >5MP for accurate focus detection
- **Illumination**: LED with brightness control

#### Environmental
- **Temperature**: 18-28°C operating range
- **Humidity**: 30-70% RH, non-condensing
- **Vibration**: <0.1g RMS for optimal accuracy
- **Power**: Uninterruptible power supply (UPS) recommended

### Software Dependencies

#### Python Environment
```bash
# Python 3.8 or higher required
python --version  # Should show Python 3.8+

# Required packages
pip install numpy>=1.19.0
pip install scipy>=1.5.0
pip install opencv-python>=4.5.0
pip install scikit-image>=0.17.0
pip install matplotlib>=3.3.0  # For calibration tools
```

#### Operating System
- **Linux**: Ubuntu 20.04+ LTS (recommended)
- **Windows**: Windows 10/11 Professional
- **macOS**: 10.15+ (development only)

## Pre-deployment Checklist

### Hardware Validation

Use this checklist to validate hardware before deployment:

```python
# Hardware validation script
from autofocus.deployment import HardwareValidator

validator = HardwareValidator()

# Run complete hardware validation
validation_results = validator.run_full_validation()

print("Hardware Validation Results:")
for component, result in validation_results.items():
    status = "✓ PASS" if result['pass'] else "✗ FAIL"
    print(f"  {component}: {status}")

    if not result['pass']:
        print(f"    Error: {result['error']}")
        print(f"    Solution: {result['solution']}")
```

#### Critical Validations
- [ ] Lens turret motorized switching (<50ms)
- [ ] Focus motor precision (±0.1μm)
- [ ] Camera resolution and frame rate
- [ ] Illumination control range
- [ ] Temperature sensor accuracy
- [ ] Stage movement precision
- [ ] Network connectivity
- [ ] Power supply stability

### Software Pre-checks

```python
# Software environment validation
from autofocus.deployment import EnvironmentValidator

env_validator = EnvironmentValidator()

# Check Python environment
python_check = env_validator.validate_python_environment()
print(f"Python Environment: {'✓ PASS' if python_check['valid'] else '✗ FAIL'}")

# Check dependencies
deps_check = env_validator.validate_dependencies()
print(f"Dependencies: {'✓ PASS' if deps_check['all_present'] else '✗ FAIL'}")

# Check hardware drivers
drivers_check = env_validator.validate_hardware_drivers()
print(f"Hardware Drivers: {'✓ PASS' if drivers_check['all_loaded'] else '✗ FAIL'}")
```

## Installation

### Standard Installation

```bash
# 1. Clone the repository
git clone https://github.com/bloodwork-ai/dual-lens-autofocus.git
cd dual-lens-autofocus

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# 3. Install package
pip install -e .

# 4. Install additional production dependencies
pip install -r requirements-production.txt

# 5. Verify installation
python -c "from autofocus.dual_lens_optimized import create_optimized_dual_lens_system; print('✓ Installation successful')"
```

### Production Installation

For production deployments, use the production-ready installation:

```bash
# Production installation with all optimizations
pip install bloodwork-autofocus[production]

# Install system service components
sudo python setup.py install_service

# Enable and start the service
sudo systemctl enable bloodwork-autofocus
sudo systemctl start bloodwork-autofocus
```

### Docker Deployment

For containerized deployments:

```dockerfile
# Dockerfile for production deployment
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -e .[production]

# Create user for security
RUN useradd -m -s /bin/bash autofocus
USER autofocus

# Expose monitoring port
EXPOSE 8080

# Start application
CMD ["python", "-m", "autofocus.production.server"]
```

Build and deploy:
```bash
# Build container
docker build -t bloodwork-autofocus:latest .

# Run container
docker run -d \
  --name autofocus-production \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/autofocus/config:/app/config \
  -v /opt/autofocus/data:/app/data \
  bloodwork-autofocus:latest
```

## Configuration

### Production Configuration File

Create a production configuration file `/opt/autofocus/config/production.yaml`:

```yaml
# Production configuration for dual-lens autofocus system
system:
  name: "Hematology Scanner #1"
  location: "Lab Room A"
  optimization_level: "ULTRA_FAST"
  enable_telemetry: true
  log_level: "INFO"

# Hardware configuration
hardware:
  camera:
    device_id: 0
    resolution: [2048, 1536]
    exposure_ms: 10
    gain: 1.0

  lens_turret:
    driver: "motorized_turret_v2"
    port: "/dev/ttyUSB0"
    switch_time_ms: 25

  focus_motor:
    driver: "piezo_focus_v3"
    port: "/dev/ttyUSB1"
    precision_um: 0.01
    speed_um_per_s: 500

  illumination:
    driver: "led_controller"
    port: "/dev/ttyUSB2"
    max_brightness: 255
    default_brightness: 128

# Lens profiles
lenses:
  lens_a:
    name: "Scanning Lens 20x"
    magnification: 20.0
    na: 0.4
    z_range_um: [-20.0, 20.0]
    focus_speed_um_per_s: 500.0
    settle_time_ms: 5.0

  lens_b:
    name: "Analysis Lens 60x"
    magnification: 60.0
    na: 1.2
    z_range_um: [-15.0, 15.0]
    focus_speed_um_per_s: 300.0
    settle_time_ms: 8.0

# Parfocal mapping
parfocal:
  model_type: "ADAPTIVE"
  calibration_points: 25
  temperature_compensation: true
  confidence_threshold: 0.95
  learning_rate: 0.1

# Performance settings
performance:
  max_handoff_time_ms: 300
  max_mapping_error_um: 1.0
  cache_size: 100
  thread_pool_workers: 4
  enable_concurrent_operations: true
  enable_predictive_focus: true

# Monitoring and alerts
monitoring:
  enable_performance_tracking: true
  alert_thresholds:
    handoff_time_ms: 350
    mapping_error_um: 1.5
    success_rate: 0.98

  telemetry:
    enable: true
    endpoint: "https://monitor.bloodwork-ai.com/telemetry"
    api_key: "${TELEMETRY_API_KEY}"
    upload_interval_s: 60

# Maintenance
maintenance:
  auto_calibration:
    enable: true
    interval_hours: 24
    temperature_change_threshold_c: 2.0

  backup:
    enable: true
    interval_hours: 6
    retention_days: 30
    location: "/opt/autofocus/backups"

# Security
security:
  enable_api_auth: true
  api_key: "${API_KEY}"
  allowed_ips: ["192.168.1.0/24", "10.0.0.0/8"]
```

### Loading Configuration

```python
from autofocus.config import ProductionConfig

# Load production configuration
config = ProductionConfig.from_file("/opt/autofocus/config/production.yaml")

# Create system with production config
system = config.create_optimized_system()

print(f"System initialized: {config.system.name}")
print(f"Optimization level: {config.system.optimization_level}")
```

## Hardware Integration

### Lens Controller Integration

```python
# Example lens controller integration
from autofocus.hardware import LensTurretController

class ProductionLensTurret(LensTurretController):
    """Production lens turret integration."""

    def __init__(self, port: str, switch_time_ms: float):
        self.port = port
        self.switch_time_ms = switch_time_ms
        self._current_lens = None
        self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize hardware connection."""
        # Initialize serial connection
        import serial
        self.serial = serial.Serial(
            self.port,
            baudrate=115200,
            timeout=1.0
        )

        # Send initialization commands
        self.serial.write(b"INIT\n")
        response = self.serial.readline()

        if b"OK" not in response:
            raise RuntimeError("Failed to initialize lens turret")

    def switch_lens(self, lens_id: LensID) -> bool:
        """Switch to specified lens."""
        try:
            command = f"SWITCH_{lens_id.value.upper()}\n"
            self.serial.write(command.encode())

            # Wait for completion
            response = self.serial.readline()

            if b"COMPLETE" in response:
                self._current_lens = lens_id
                return True
            else:
                return False

        except Exception as e:
            print(f"Lens switch error: {e}")
            return False

    def get_current_lens(self) -> Optional[LensID]:
        """Get currently active lens."""
        return self._current_lens

# Register hardware controller
system.camera.register_lens_controller(ProductionLensTurret("/dev/ttyUSB0", 25.0))
```

### Focus Motor Integration

```python
# Example focus motor integration
from autofocus.hardware import FocusMotorController

class PiezoFocusMotor(FocusMotorController):
    """Piezo focus motor integration for production."""

    def __init__(self, port: str, precision_um: float):
        self.port = port
        self.precision_um = precision_um
        self._position_um = 0.0
        self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize piezo controller."""
        import serial
        self.serial = serial.Serial(
            self.port,
            baudrate=57600,
            timeout=0.5
        )

        # Initialize piezo controller
        self.serial.write(b"*IDN?\n")
        response = self.serial.readline()
        print(f"Focus controller: {response.decode().strip()}")

        # Set precision mode
        self.serial.write(f"PREC {self.precision_um}\n".encode())

    def set_position(self, z_um: float) -> bool:
        """Set focus position."""
        try:
            # Send position command
            command = f"MOV {z_um:.3f}\n"
            self.serial.write(command.encode())

            # Wait for movement completion
            while True:
                self.serial.write(b"MOV?\n")
                response = self.serial.readline()

                if b"READY" in response:
                    self._position_um = z_um
                    return True
                elif b"ERROR" in response:
                    return False

                time.sleep(0.001)  # 1ms polling

        except Exception as e:
            print(f"Focus move error: {e}")
            return False

    def get_position(self) -> float:
        """Get current focus position."""
        return self._position_um

# Register focus motor
system.camera.register_focus_motor(PiezoFocusMotor("/dev/ttyUSB1", 0.01))
```

## Calibration Procedures

### Initial System Calibration

```python
from autofocus.calibration import ProductionCalibration

def perform_initial_calibration(system):
    """Perform comprehensive initial calibration."""

    calibrator = ProductionCalibration(system)

    print("🔧 Starting initial system calibration...")

    # 1. Hardware verification
    print("\n1. Hardware verification...")
    hw_results = calibrator.verify_hardware()

    if not hw_results['all_pass']:
        print("❌ Hardware verification failed!")
        for component, result in hw_results['components'].items():
            if not result['pass']:
                print(f"  {component}: {result['error']}")
        return False

    print("✅ Hardware verification passed")

    # 2. Lens characterization
    print("\n2. Lens characterization...")
    lens_results = calibrator.characterize_lenses()

    for lens_id, result in lens_results.items():
        print(f"  {lens_id.value}: Range {result['z_range_um']}, "
              f"Speed {result['speed_um_per_s']:.1f}μm/s")

    # 3. Parfocal mapping calibration
    print("\n3. Parfocal mapping calibration...")
    mapping_result = calibrator.calibrate_parfocal_mapping(
        num_points=30,
        z_range=(-8, 8),
        enable_temperature_compensation=True
    )

    print(f"  Model: {mapping_result['model_type']}")
    print(f"  RMS error: {mapping_result['rms_error_um']:.3f}μm")
    print(f"  Max error: {mapping_result['max_error_um']:.3f}μm")

    # 4. Performance validation
    print("\n4. Performance validation...")
    perf_results = calibrator.validate_performance(num_tests=50)

    print(f"  Average handoff time: {perf_results['avg_handoff_time_ms']:.1f}ms")
    print(f"  Success rate: {perf_results['success_rate']:.1%}")
    print(f"  Target achievement: {perf_results['target_met_rate']:.1%}")

    # 5. Save calibration
    print("\n5. Saving calibration...")
    calibration_file = "/opt/autofocus/calibration/initial_calibration.json"
    calibrator.save_calibration(calibration_file)

    print(f"✅ Initial calibration complete: {calibration_file}")

    # Performance assessment
    if (perf_results['avg_handoff_time_ms'] <= 300 and
        perf_results['success_rate'] >= 0.99 and
        mapping_result['rms_error_um'] <= 0.1):
        print("\n🎉 System ready for production deployment!")
        return True
    else:
        print("\n⚠️  System needs optimization before production")
        return False

# Run initial calibration
calibration_success = perform_initial_calibration(system)
```

### Routine Calibration

```python
def perform_routine_calibration(system):
    """Perform routine calibration maintenance."""

    print("🔄 Starting routine calibration...")

    # Check if recalibration is needed
    current_stats = system.get_performance_stats()

    needs_recalibration = (
        current_stats['avg_handoff_time_ms'] > 350 or
        current_stats['success_rate'] < 0.98 or
        current_stats['avg_mapping_error_um'] > 0.2
    )

    if needs_recalibration:
        print("📊 Performance degradation detected, recalibrating...")

        # Quick recalibration
        quick_calibration_points = generate_calibration_data(15)
        result = system.parfocal_mapping.calibrate_enhanced(quick_calibration_points)

        print(f"✅ Recalibration complete: {result['rms_error_um']:.3f}μm RMS")

        # Save updated calibration
        system.save_calibration("/opt/autofocus/calibration/routine_update.json")
    else:
        print("✅ System performance within specifications")

# Schedule routine calibration
import schedule

schedule.every(24).hours.do(perform_routine_calibration, system)
```

## Production Deployment

### Service Configuration

Create a systemd service for Linux deployment:

```ini
# /etc/systemd/system/bloodwork-autofocus.service
[Unit]
Description=Bloodwork AI Dual-Lens Autofocus Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=autofocus
Group=autofocus
WorkingDirectory=/opt/autofocus
Environment=PYTHONPATH=/opt/autofocus
ExecStart=/opt/autofocus/venv/bin/python -m autofocus.production.server
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=autofocus

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/autofocus/data /opt/autofocus/logs

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable bloodwork-autofocus
sudo systemctl start bloodwork-autofocus
sudo systemctl status bloodwork-autofocus
```

### Production Server

```python
# autofocus/production/server.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from autofocus.dual_lens_optimized import create_optimized_dual_lens_system
from autofocus.config import ProductionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/autofocus/logs/autofocus.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bloodwork AI Autofocus API",
    description="Production API for dual-lens autofocus system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
autofocus_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize autofocus system on startup."""
    global autofocus_system

    try:
        # Load production configuration
        config = ProductionConfig.from_file("/opt/autofocus/config/production.yaml")

        # Create optimized system
        autofocus_system = config.create_optimized_system()

        logger.info("Autofocus system initialized successfully")

        # Start background monitoring
        asyncio.create_task(background_monitoring())

    except Exception as e:
        logger.error(f"Failed to initialize autofocus system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global autofocus_system

    if autofocus_system:
        autofocus_system.close()
        logger.info("Autofocus system closed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if autofocus_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Quick system check
    try:
        stats = autofocus_system.get_performance_stats()

        return {
            "status": "healthy",
            "system_time": time.time(),
            "performance": {
                "avg_handoff_time_ms": stats.get('avg_handoff_time_ms', 0),
                "success_rate": stats.get('success_rate', 0),
                "cache_hit_rate": stats.get('cache_hit_rate', 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {e}")

@app.post("/handoff")
async def perform_handoff(z_source: float):
    """Perform A→B lens handoff."""
    if autofocus_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = autofocus_system.handoff_a_to_b_optimized(z_source)

        return {
            "success": result.success,
            "z_target": result.z_target,
            "elapsed_ms": result.elapsed_ms,
            "mapping_error_um": result.mapping_error_um,
            "lens_switch_ms": result.lens_switch_ms,
            "focus_move_ms": result.focus_move_ms,
            "focus_search_ms": result.focus_search_ms,
            "validation_ms": result.validation_ms
        }
    except Exception as e:
        logger.error(f"Handoff error: {e}")
        raise HTTPException(status_code=500, detail=f"Handoff failed: {e}")

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics."""
    if autofocus_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        stats = autofocus_system.get_performance_stats()
        mapping_report = autofocus_system.parfocal_mapping.get_mapping_accuracy_report()

        return {
            "performance": stats,
            "mapping": mapping_report,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

async def background_monitoring():
    """Background monitoring and maintenance tasks."""
    while True:
        try:
            # Check system health
            stats = autofocus_system.get_performance_stats()

            # Log performance metrics
            logger.info(f"Performance: {stats['avg_handoff_time_ms']:.1f}ms avg, "
                       f"{stats['success_rate']:.1%} success rate")

            # Check for alerts
            if stats['avg_handoff_time_ms'] > 350:
                logger.warning(f"High handoff time: {stats['avg_handoff_time_ms']:.1f}ms")

            if stats['success_rate'] < 0.98:
                logger.warning(f"Low success rate: {stats['success_rate']:.1%}")

            # Routine maintenance
            if should_perform_routine_calibration():
                logger.info("Starting routine calibration...")
                perform_routine_calibration(autofocus_system)

        except Exception as e:
            logger.error(f"Background monitoring error: {e}")

        # Sleep for 60 seconds
        await asyncio.sleep(60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Monitoring & Maintenance

### Production Monitoring Dashboard

```python
# Simple monitoring dashboard
import streamlit as st
import requests
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.title("Dual-Lens Autofocus System Monitor")

# Configuration
API_BASE_URL = "http://localhost:8080"

# Auto-refresh
if st.checkbox("Auto-refresh (30s)", value=True):
    time.sleep(30)
    st.experimental_rerun()

# Get system status
try:
    health_response = requests.get(f"{API_BASE_URL}/health")
    health_data = health_response.json()

    stats_response = requests.get(f"{API_BASE_URL}/stats")
    stats_data = stats_response.json()

    # Status indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "System Status",
            health_data['status'].upper(),
            delta="Online" if health_data['status'] == 'healthy' else "Error"
        )

    with col2:
        handoff_time = health_data['performance']['avg_handoff_time_ms']
        st.metric(
            "Avg Handoff Time",
            f"{handoff_time:.1f}ms",
            delta=f"Target: ≤300ms"
        )

    with col3:
        success_rate = health_data['performance']['success_rate']
        st.metric(
            "Success Rate",
            f"{success_rate:.1%}",
            delta=f"Target: ≥99%"
        )

    with col4:
        cache_hit_rate = health_data['performance']['cache_hit_rate']
        st.metric(
            "Cache Hit Rate",
            f"{cache_hit_rate:.1%}",
            delta=f"Target: ≥50%"
        )

    # Performance trends (simulated data)
    st.subheader("Performance Trends")

    # Generate sample trend data
    times = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
    handoff_times = [handoff_time + np.random.normal(0, 20) for _ in times]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=handoff_times,
        mode='lines+markers',
        name='Handoff Time',
        line=dict(color='blue')
    ))

    fig.add_hline(y=300, line_dash="dash", line_color="red",
                  annotation_text="Target: 300ms")

    fig.update_layout(
        title="Handoff Time Trend (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Handoff Time (ms)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Mapping accuracy
    st.subheader("Mapping Accuracy")

    mapping_data = stats_data['mapping']['calibration']

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "RMS Error",
            f"{mapping_data['rms_error_um']:.3f}μm",
            delta="Target: ≤0.1μm"
        )

    with col2:
        st.metric(
            "Max Error",
            f"{mapping_data['max_error_um']:.3f}μm",
            delta="Target: ≤1.0μm"
        )

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to autofocus system")
except Exception as e:
    st.error(f"Error: {e}")

# Manual controls
st.subheader("Manual Controls")

col1, col2 = st.columns(2)

with col1:
    if st.button("Perform Test Handoff"):
        try:
            test_z = 3.0
            response = requests.post(f"{API_BASE_URL}/handoff", params={"z_source": test_z})
            result = response.json()

            if result['success']:
                st.success(f"Handoff successful: {result['elapsed_ms']:.1f}ms")
            else:
                st.error("Handoff failed")
        except Exception as e:
            st.error(f"Test failed: {e}")

with col2:
    if st.button("Force Calibration"):
        st.info("Calibration would be triggered here")
```

### Automated Maintenance

```python
# Automated maintenance script
import schedule
import time
import logging
from autofocus.maintenance import MaintenanceManager

logger = logging.getLogger(__name__)

class ProductionMaintenance:
    """Automated maintenance for production systems."""

    def __init__(self, system):
        self.system = system
        self.maintenance_manager = MaintenanceManager(system)

    def daily_health_check(self):
        """Perform daily health check."""
        logger.info("Starting daily health check...")

        # Performance check
        stats = self.system.get_performance_stats()

        # Health metrics
        health_score = self._calculate_health_score(stats)

        logger.info(f"System health score: {health_score:.1f}/100")

        if health_score < 80:
            logger.warning("System health below threshold, scheduling maintenance")
            self.schedule_maintenance()

    def weekly_calibration_check(self):
        """Check if calibration update is needed."""
        logger.info("Checking calibration accuracy...")

        mapping_report = self.system.parfocal_mapping.get_mapping_accuracy_report()

        needs_recalibration = (
            mapping_report['calibration']['rms_error_um'] > 0.15 or
            len(mapping_report['recent_performance']['validation_points']) > 100
        )

        if needs_recalibration:
            logger.info("Calibration update recommended")
            self.perform_calibration_update()

    def monthly_full_calibration(self):
        """Perform comprehensive monthly calibration."""
        logger.info("Starting monthly full calibration...")

        # Backup current calibration
        self.system.save_calibration("/opt/autofocus/backups/pre_monthly_calibration.json")

        # Perform full calibration
        calibration_success = perform_initial_calibration(self.system)

        if calibration_success:
            logger.info("Monthly calibration completed successfully")
        else:
            logger.error("Monthly calibration failed, restoring backup")
            self.system.load_calibration("/opt/autofocus/backups/pre_monthly_calibration.json")

    def _calculate_health_score(self, stats):
        """Calculate overall system health score."""
        score = 100

        # Handoff time penalty
        if stats['avg_handoff_time_ms'] > 300:
            score -= min(20, (stats['avg_handoff_time_ms'] - 300) / 10)

        # Success rate penalty
        if stats['success_rate'] < 0.99:
            score -= (0.99 - stats['success_rate']) * 100

        # Cache efficiency bonus/penalty
        if stats['cache_hit_rate'] > 0.7:
            score += 5
        elif stats['cache_hit_rate'] < 0.3:
            score -= 10

        return max(0, score)

# Schedule maintenance tasks
maintenance = ProductionMaintenance(system)

schedule.every().day.at("02:00").do(maintenance.daily_health_check)
schedule.every().week.do(maintenance.weekly_calibration_check)
schedule.every().month.do(maintenance.monthly_full_calibration)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Deployment Issues

#### 1. Hardware Connection Problems

**Symptoms:**
- System fails to initialize
- Hardware validation errors
- Communication timeouts

**Diagnosis:**
```python
from autofocus.diagnostics import HardwareDiagnostics

diagnostics = HardwareDiagnostics()

# Check all hardware connections
connection_results = diagnostics.check_hardware_connections()

for device, result in connection_results.items():
    if not result['connected']:
        print(f"❌ {device}: {result['error']}")
        print(f"   Solution: {result['solution']}")
```

**Solutions:**
1. Verify USB/serial connections
2. Check device permissions (`chmod 666 /dev/ttyUSB*`)
3. Validate baud rates and communication protocols
4. Test with hardware vendor diagnostic tools

#### 2. Permission Issues

**Symptoms:**
- Service fails to start
- File access errors
- Hardware device access denied

**Solutions:**
```bash
# Fix file permissions
sudo chown -R autofocus:autofocus /opt/autofocus
sudo chmod -R 755 /opt/autofocus
sudo chmod 666 /dev/ttyUSB*

# Add user to dialout group
sudo usermod -a -G dialout autofocus

# Restart service
sudo systemctl restart bloodwork-autofocus
```

#### 3. Performance Degradation

**Symptoms:**
- Handoff times increasing over time
- Lower success rates
- Thermal drift issues

**Diagnosis:**
```python
# Performance trend analysis
from autofocus.diagnostics import PerformanceDiagnostics

perf_diag = PerformanceDiagnostics(system)

# Analyze recent performance
trend_analysis = perf_diag.analyze_performance_trends()

print(f"Performance trend: {trend_analysis['trend']}")
print(f"Primary cause: {trend_analysis['primary_cause']}")
print(f"Recommended action: {trend_analysis['recommendation']}")
```

**Solutions:**
1. Recalibrate parfocal mapping
2. Check for mechanical wear
3. Validate temperature compensation
4. Clear performance caches and restart

## Backup & Recovery

### Automated Backup System

```python
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

class BackupManager:
    """Automated backup and recovery system."""

    def __init__(self, backup_location="/opt/autofocus/backups"):
        self.backup_location = Path(backup_location)
        self.backup_location.mkdir(parents=True, exist_ok=True)

    def create_system_backup(self, system):
        """Create complete system backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_location / f"backup_{timestamp}"
        backup_dir.mkdir()

        # Backup calibration data
        calibration_backup = backup_dir / "calibration.json"
        system.save_calibration(str(calibration_backup))

        # Backup configuration
        config_backup = backup_dir / "config.yaml"
        shutil.copy("/opt/autofocus/config/production.yaml", config_backup)

        # Backup performance data
        stats = system.get_performance_stats()
        stats_backup = backup_dir / "performance_stats.json"
        with open(stats_backup, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        # Backup logs
        logs_backup = backup_dir / "logs"
        logs_backup.mkdir()
        shutil.copytree("/opt/autofocus/logs", logs_backup, dirs_exist_ok=True)

        print(f"✅ System backup created: {backup_dir}")
        return backup_dir

    def restore_system_backup(self, system, backup_path):
        """Restore system from backup."""
        backup_dir = Path(backup_path)

        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        try:
            # Restore calibration
            calibration_file = backup_dir / "calibration.json"
            if calibration_file.exists():
                system.load_calibration(str(calibration_file))
                print("✅ Calibration restored")

            # Restore configuration
            config_file = backup_dir / "config.yaml"
            if config_file.exists():
                shutil.copy(config_file, "/opt/autofocus/config/production.yaml")
                print("✅ Configuration restored")

            print(f"✅ System restored from: {backup_path}")

        except Exception as e:
            print(f"❌ Restore failed: {e}")
            raise

    def cleanup_old_backups(self, retention_days=30):
        """Remove backups older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        for backup_dir in self.backup_location.glob("backup_*"):
            try:
                # Parse timestamp from directory name
                timestamp_str = backup_dir.name.replace("backup_", "")
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if backup_date < cutoff_date:
                    shutil.rmtree(backup_dir)
                    print(f"🗑️  Removed old backup: {backup_dir}")

            except ValueError:
                # Skip directories that don't match expected format
                continue

# Automated backup scheduling
backup_manager = BackupManager()

def automated_backup():
    """Automated backup function."""
    try:
        backup_path = backup_manager.create_system_backup(system)
        backup_manager.cleanup_old_backups(retention_days=30)

        logger.info(f"Automated backup completed: {backup_path}")

    except Exception as e:
        logger.error(f"Automated backup failed: {e}")

# Schedule backups every 6 hours
schedule.every(6).hours.do(automated_backup)
```

### Disaster Recovery

```python
def disaster_recovery_procedure():
    """Complete disaster recovery procedure."""

    print("🚨 Starting disaster recovery procedure...")

    # 1. Identify latest backup
    backup_manager = BackupManager()
    backup_dirs = sorted(backup_manager.backup_location.glob("backup_*"))

    if not backup_dirs:
        print("❌ No backups found! Manual recovery required.")
        return False

    latest_backup = backup_dirs[-1]
    print(f"📁 Using latest backup: {latest_backup}")

    # 2. Reinitialize system
    try:
        config = ProductionConfig.from_file("/opt/autofocus/config/production.yaml")
        recovered_system = config.create_optimized_system()

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

    # 3. Restore from backup
    try:
        backup_manager.restore_system_backup(recovered_system, latest_backup)

    except Exception as e:
        print(f"❌ Failed to restore backup: {e}")
        return False

    # 4. Validate recovered system
    try:
        # Quick performance test
        test_result = recovered_system.handoff_a_to_b_optimized(0.0)

        if test_result.success:
            print("✅ System recovery validation passed")
            return True
        else:
            print("❌ System recovery validation failed")
            return False

    except Exception as e:
        print(f"❌ Recovery validation error: {e}")
        return False

# Recovery can be triggered manually or automatically
if __name__ == "__main__":
    success = disaster_recovery_procedure()
    if success:
        print("🎉 Disaster recovery completed successfully!")
    else:
        print("💥 Disaster recovery failed - manual intervention required")
```

## Conclusion

This comprehensive deployment guide provides everything needed to successfully deploy the dual-lens autofocus system in production hematology environments. Key deployment best practices:

1. **Thorough pre-deployment validation** of hardware and software
2. **Proper configuration management** with environment-specific settings
3. **Robust monitoring and alerting** for production reliability
4. **Automated maintenance and calibration** procedures
5. **Comprehensive backup and recovery** systems

The system is designed for high reliability and performance in production environments, with built-in monitoring, self-healing capabilities, and comprehensive error handling.

For additional support:
- [Performance Profiling Guide](performance_profiling.md)
- [Testing Framework Documentation](testing_guide.md)
- [API Reference](api_reference.md)
- [Troubleshooting Guide](troubleshooting.md)