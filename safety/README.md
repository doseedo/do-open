# Safety Monitor & Rollback Manager (Agent 17)

Comprehensive safety monitoring system for the self-expanding music generation framework. Ensures system stability during parameter expansions with git-based checkpointing, comprehensive testing, and instant rollback capabilities.

## Overview

**CRITICAL PRINCIPLE**: System must NEVER break existing functionality. Rollback is better than broken system.

The Safety Monitor provides:
- **Git-based checkpointing** for code state management
- **Registry snapshots** to track parameter changes
- **Model snapshots** to backup trained models
- **Comprehensive testing** of existing parameters, generator stability, and quality
- **Quality monitoring** to detect degradation
- **Conflict detection** to identify parameter naming/functional conflicts
- **Instant rollback** to restore system to known-good state
- **Detailed reporting** for audit and debugging

## Architecture

```
SafetyMonitor
├── GitOperations          # Git commands with retry logic
├── RegistrySnapshot       # Parameter registry snapshots
├── ModelSnapshot          # Model file snapshots
├── QualityMonitor         # Reconstruction quality tracking
├── ComprehensiveTestSuite # System validation tests
└── ParameterTester        # Parameter-specific tests
```

## Quick Start

```python
from safety import SafetyMonitor

# Create monitor
monitor = SafetyMonitor()

# 1. Create checkpoint before expansion
checkpoint_id = monitor.checkpoint_system(
    description="Pre-expansion: Adding harmony.voice_leading"
)

# 2. Add new parameter (done by expansion agent)
# ... parameter expansion code ...

# 3. Monitor the expansion
result = monitor.monitor_expansion(
    param_name='harmony.voice_leading',
    checkpoint_id=checkpoint_id
)

# 4. Decision
if result.safe:
    print("✅ Expansion approved")
else:
    print("❌ Rolling back...")
    monitor.rollback_to_checkpoint(checkpoint_id)
```

## Key Features

### 1. Checkpointing

Creates comprehensive snapshots before expansions:

```python
checkpoint_id = monitor.checkpoint_system(description="Phase 1: Harmony expansion")
```

Checkpoint includes:
- Git commit hash
- Parameter registry state (all 165+ parameters)
- Model files list (checksums + metadata)
- Test suite baseline results
- Quality baseline metrics

### 2. Expansion Monitoring

Runs 5 critical checks after parameter expansion:

```python
result = monitor.monitor_expansion('new.parameter', checkpoint_id)
```

**Checks:**
1. **Existing Parameters** - All existing parameters still work
2. **Generator Stability** - No crashes on random parameter combinations
3. **Quality Maintained** - Reconstruction quality hasn't degraded
4. **New Parameter Effective** - New parameter actually affects output
5. **No Conflicts** - No naming/functional conflicts with existing parameters

**Safety Levels:**
- `SAFE` - All checks pass, no warnings
- `WARNING` - All critical checks pass, some warnings
- `UNSAFE` - Critical checks failed, rollback recommended
- `CRITICAL` - System broken, immediate rollback required

### 3. Rollback Capabilities

**Full system rollback:**
```python
monitor.rollback_to_checkpoint(checkpoint_id)
```

Restores:
- Code state (git reset)
- Parameter registry
- Model files
- System modules (reload)

**Parameter-only rollback:**
```python
monitor.rollback_expansion('problematic.parameter')
```

Removes:
- Parameter from registry
- Associated model file
- Reloads system

### 4. Quality Monitoring

Tracks reconstruction quality over time:

```python
# Establish baseline (run once)
baseline = monitor.quality_monitor.establish_baseline()

# Test current quality (run after expansion)
quality_test = monitor.quality_monitor.test_current_quality()
```

Quality metrics:
- Average reconstruction quality (0-1)
- Degradation from baseline
- Per-file quality scores

### 5. Conflict Detection

Detects parameter conflicts:

```python
conflicts = monitor.parameter_tester.test_parameter_conflicts('new.parameter')
```

Detects:
- **Duplicate names** (exact match)
- **Name similarity** (>80% similar)
- **Functional overlap** (high correlation)

## Configuration

Customize monitoring behavior:

```python
from safety import SafetyConfig

config = SafetyConfig()

# Quality thresholds
config.min_baseline_quality = 0.6
config.quality_degradation_tolerance = 0.05
config.critical_quality_threshold = 0.4

# Test configuration
config.num_stability_tests = 10
config.num_quality_test_files = 20

# Conflict detection
config.name_similarity_threshold = 0.8
config.parameter_correlation_threshold = 0.95

# Checkpoint management
config.max_checkpoints = 50
config.checkpoint_retention_days = 30

monitor = SafetyMonitor(config)
```

## Usage Patterns

### Pattern 1: Single Parameter Expansion

```python
monitor = SafetyMonitor()

# Checkpoint
cp = monitor.checkpoint_system("Adding harmony.tension")

# Expand (simulated)
# ... add parameter, train model, deploy ...

# Monitor
result = monitor.monitor_expansion('harmony.tension', cp)

# Decide
if not result.safe:
    monitor.rollback_to_checkpoint(cp)
```

### Pattern 2: Batch Expansion

```python
monitor = SafetyMonitor()

# Checkpoint once
cp = monitor.checkpoint_system("Batch: 10 harmony parameters")

# Expand multiple parameters
for param in new_parameters:
    # Add parameter
    # Train model
    # Monitor individually
    result = monitor.monitor_expansion(param.name, cp)

    if not result.safe:
        # Rollback just this parameter
        monitor.rollback_expansion(param.name)
```

### Pattern 3: Phased Expansion

```python
monitor = SafetyMonitor()

phases = [
    ('Phase 1: Harmony', harmony_params),
    ('Phase 2: Rhythm', rhythm_params),
    ('Phase 3: Texture', texture_params)
]

for phase_name, params in phases:
    # Checkpoint per phase
    cp = monitor.checkpoint_system(phase_name)

    # Expand phase
    for param in params:
        # ... expansion ...
        result = monitor.monitor_expansion(param, cp)

        if not result.safe:
            # Rollback entire phase
            monitor.rollback_to_checkpoint(cp)
            break
```

## Reports and Logging

### Generate Report

```python
# Overall report
report = monitor.generate_report()
print(report)

# Specific checkpoint
report = monitor.generate_report(checkpoint_id='abc123_1234567890')
```

### View History

```python
# All checkpoints
checkpoints = monitor.get_checkpoint_history()
for cp in checkpoints:
    print(f"{cp.id}: {cp.description} ({cp.status.value})")

# All expansions
expansions = monitor.get_expansion_log()
for exp in expansions:
    print(f"{exp.parameter_name}: {exp.safety_level.value}")
```

### Access Detailed Results

```python
result = monitor.monitor_expansion('param.name', checkpoint_id)

# Check results
print(f"Safe: {result.safe}")
print(f"Safety level: {result.safety_level.value}")

# Issues
for issue in result.issues:
    print(f"❌ {issue}")

# Warnings
for warning in result.warnings:
    print(f"⚠️ {warning}")

# Quality metrics
print(f"Current quality: {result.quality_metrics['current_quality']:.3f}")
print(f"Baseline: {result.quality_metrics['baseline_quality']:.3f}")
print(f"Degradation: {result.quality_metrics['degradation']:.3f}")

# Test details
print(f"Existing params: {result.test_results['existing_params']['pass']}")
print(f"Generator stable: {result.test_results['generator_stability']['pass']}")
```

## Error Handling

The Safety Monitor uses defensive error handling:

- **Git failures** → Continues with snapshots, warns about missing git state
- **Import errors** → Gracefully degrades, provides limited functionality
- **Test failures** → Captured and reported, doesn't crash system
- **Rollback errors** → Critical errors raised, system state logged

Always check logs for warnings and errors.

## Git Operations

Git commands include retry logic for network failures:

- **Max retries**: 4 attempts
- **Backoff**: 2s, 4s, 8s, 16s
- **Operations**: push, fetch, pull
- **Branch validation**: Ensures branch starts with 'claude/' and ends with session ID

## File Structure

```
safety/
├── __init__.py              # Package exports
├── safety_monitor.py        # Main SafetyMonitor class (3500+ lines)
├── example_usage.py         # Usage examples
├── README.md               # This file
├── checkpoints/            # Checkpoint metadata
│   └── {checkpoint_id}.json
├── snapshots/
│   ├── registry/           # Registry snapshots
│   │   └── {checkpoint_id}.json
│   └── models/             # Model snapshots
│       ├── {checkpoint_id}.json
│       └── {checkpoint_id}/ # Model backups
└── logs/                   # Monitoring logs
    └── expansion_{param}_{timestamp}.json
```

## Integration with Expansion Pipeline

The Safety Monitor integrates with the expansion pipeline:

```
1. Gap Detection (Agent X)
   ↓
2. CHECKPOINT ← SafetyMonitor.checkpoint_system()
   ↓
3. Parameter Proposal (Agent Y)
   ↓
4. Code Generation (Agent Z)
   ↓
5. Training (Agent W)
   ↓
6. MONITORING ← SafetyMonitor.monitor_expansion()
   ↓
7. Decision:
   - SAFE → Continue
   - UNSAFE → SafetyMonitor.rollback_to_checkpoint()
```

## Testing

Run the example script:

```bash
# Run default example
python safety/example_usage.py

# Run specific example
python safety/example_usage.py 1  # Safe expansion
python safety/example_usage.py 2  # Rollback
python safety/example_usage.py 3  # Checkpoint management
python safety/example_usage.py 4  # Quality monitoring
python safety/example_usage.py 5  # Conflict detection

# Run all examples
python safety/example_usage.py all
```

## Best Practices

1. **Always checkpoint before expansion** - Even "safe" expansions can have unexpected effects
2. **Monitor immediately after deployment** - Don't wait to discover issues
3. **Check warnings** - Even safe expansions may have warnings worth addressing
4. **Keep checkpoint retention reasonable** - Old checkpoints consume disk space
5. **Verify rollbacks** - Always verify system state after rollback
6. **Review logs** - Check `safety/logs/` for detailed expansion history
7. **Test in isolation** - When possible, test new parameters in isolated environment first

## Troubleshooting

### "No test data found"
- Ensure `test_data/midi/*.mid` exists
- System will use degraded quality monitoring
- Generate test data or copy sample MIDI files

### "Git checkpoint failed"
- Check git repository status
- Ensure working directory is clean
- System continues with NO_GIT checkpoint (limited rollback)

### "Registry import failed"
- Check `parameters/universal_registry.py` exists
- Verify registry is properly initialized
- System provides limited functionality

### "Model snapshot empty"
- Normal if no models trained yet
- Models will be snapshotted as they're created

### "Rollback verification failed"
- Review verification issues in logs
- May indicate incomplete rollback
- Manually verify system state

## Performance

Typical operation times:
- **Checkpoint creation**: 1-5 seconds
- **Expansion monitoring**: 10-30 seconds
- **Rollback**: 2-10 seconds
- **Quality baseline**: 30-60 seconds (one-time)

Performance depends on:
- Number of parameters (165 → 515 → 800+)
- Number of models
- Size of test dataset
- Git repository size

## Version History

- **v1.0.0** (2024) - Initial release
  - Git-based checkpointing
  - Comprehensive monitoring
  - Instant rollback
  - Quality tracking
  - Conflict detection

## Author

Agent 17 - Safety Monitor & Rollback Manager

## License

Part of the self-expanding music generation framework.
