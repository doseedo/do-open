# Human-in-Loop Oversight Interface (Agent 29)

Web dashboard for monitoring and managing the self-expanding inverse music generation system.

## Features

### 📊 Real-Time Dashboard
- Live system statistics and health monitoring
- Parameter count tracking (current: 165, target: 515+)
- Pending proposals queue
- Quality metrics visualization
- Real-time notifications via WebSocket

### ✅ Approval Workflow
- Review LLM-proposed parameters
- Batch approve/reject operations
- Priority-based filtering
- Detailed proposal information
- Gap detection context

### 📈 Quality Metrics
- Reconstruction error tracking
- Musical coherence measurement
- Genre accuracy monitoring
- Model confidence scoring
- Synthetic data validation rates
- Visual gauges and trend charts

### 📝 Audit Logging
- Complete action history
- User attribution
- Proposal lifecycle tracking
- Exportable logs

## Installation

```bash
# Install required dependencies
pip install flask flask-socketio flask-cors sqlalchemy numpy

# Or use requirements file
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Web Server

```bash
# From the interface directory
python human_oversight.py --server

# Custom host/port
python human_oversight.py --server --host 0.0.0.0 --port 5000

# Debug mode
python human_oversight.py --server --debug
```

### 2. Access the Dashboard

Open your browser to: `http://localhost:5000`

## Programmatic Usage

### Create Expansion Proposal

```python
from interface.human_oversight import (
    HumanOversightEngine,
    ExpansionSource,
    QualityMetric
)

# Initialize engine
engine = HumanOversightEngine()

# Create proposal
proposal_id = engine.create_proposal(
    parameter_name="swing_intensity",
    parameter_path="rhythm.jazz.swing_intensity",
    parameter_type="probability",
    description="Controls swing feel intensity (0.0-1.0)",
    source=ExpansionSource.GAP_DETECTION,
    llm_reasoning="Gap detected in reconstruction of swing-heavy jazz pieces",
    expected_impact="Will improve swing feel reconstruction by 15%",
    priority=80
)

print(f"Created proposal: {proposal_id}")
```

### Approve/Reject Proposals

```python
# Approve proposal
engine.approve_proposal(
    proposal_id=proposal_id,
    reviewer_notes="Great addition for jazz reconstruction",
    user_name="John Doe"
)

# Reject proposal
engine.reject_proposal(
    proposal_id=another_id,
    rejection_reason="Too domain-specific, may affect other genres",
    user_name="Jane Smith"
)
```

### Record Quality Metrics

```python
# Record metric
engine.record_metric(
    proposal_id=proposal_id,
    metric_type=QualityMetric.RECONSTRUCTION_ERROR,
    metric_value=0.08,  # 8% error
    context={'test_set': 'jazz_corpus_v1'}
)

engine.record_metric(
    proposal_id=proposal_id,
    metric_type=QualityMetric.MUSICAL_COHERENCE,
    metric_value=0.92,  # 92% coherence
    context={'evaluator': 'human_panel'}
)
```

### Track Training Data

```python
# Record synthetic training data
engine.record_training_data(
    proposal_id=proposal_id,
    midi_file_path="synthetic/swing_001.mid",
    parameter_value={"swing_intensity": 0.75},
    synthetic_score=0.88,
    validation_passed=True,
    generation_time=2.3  # seconds
)
```

### Get Statistics

```python
# Get comprehensive statistics
stats = engine.get_statistics()

print(f"Total parameters: {stats['current_parameters']}")
print(f"Pending proposals: {stats['proposals_by_status']['pending']}")
print(f"System health: {stats['system_health']['health_score']:.1f}/100")
```

## REST API

### Endpoints

#### `GET /api/statistics`
Get system statistics

```bash
curl http://localhost:5000/api/statistics
```

#### `GET /api/proposals`
Get proposals with filtering

```bash
# All proposals
curl http://localhost:5000/api/proposals

# Filter by status
curl http://localhost:5000/api/proposals?status=pending

# Filter by source
curl http://localhost:5000/api/proposals?source=gap_detection
```

#### `POST /api/proposals`
Create new proposal

```bash
curl -X POST http://localhost:5000/api/proposals \
  -H "Content-Type: application/json" \
  -d '{
    "parameter_name": "voicing_complexity",
    "parameter_path": "harmony.jazz.voicing_complexity",
    "parameter_type": "continuous",
    "description": "Controls chord voicing complexity",
    "source": "llm_proposal",
    "priority": 75
  }'
```

#### `POST /api/proposals/{id}/approve`
Approve proposal

```bash
curl -X POST http://localhost:5000/api/proposals/{proposal_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "reviewer_notes": "Excellent proposal",
    "user_name": "John Doe"
  }'
```

#### `POST /api/proposals/{id}/reject`
Reject proposal

```bash
curl -X POST http://localhost:5000/api/proposals/{proposal_id}/reject \
  -H "Content-Type: application/json" \
  -d '{
    "rejection_reason": "Too domain-specific",
    "user_name": "Jane Smith"
  }'
```

#### `GET /api/proposals/{id}/metrics`
Get proposal metrics

```bash
curl http://localhost:5000/api/proposals/{proposal_id}/metrics
```

#### `POST /api/proposals/{id}/metrics`
Record metric

```bash
curl -X POST http://localhost:5000/api/proposals/{proposal_id}/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "metric_type": "reconstruction_error",
    "metric_value": 0.08,
    "context": {"test_set": "jazz_v1"}
  }'
```

#### `GET /api/audit`
Get audit logs

```bash
# All logs
curl http://localhost:5000/api/audit

# Filter by action
curl http://localhost:5000/api/audit?action_type=APPROVE_PROPOSAL

# Filter by proposal
curl http://localhost:5000/api/audit?proposal_id={id}
```

## WebSocket Events

### Connect
```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected');
    socket.emit('subscribe', { room: 'general' });
});
```

### Notifications
```javascript
socket.on('notification', (data) => {
    console.log('Event:', data.event_type);
    console.log('Data:', data.data);
});
```

## Database Schema

### Tables

#### `expansion_proposals`
- Stores parameter expansion proposals
- Tracks status, priority, review information
- Links to metrics and training data

#### `metric_records`
- Quality metrics for proposals
- Pass/fail status against thresholds
- Context and timestamps

#### `training_data_records`
- Synthetic training data tracking
- Validation status
- Generation metadata

#### `system_metrics`
- System-wide metrics
- Time-series data
- Categorized for analysis

#### `audit_logs`
- Complete action history
- User attribution
- Searchable and exportable

## Quality Thresholds

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| Reconstruction Error | ≤ 0.15 | Lower is better |
| Musical Coherence | ≥ 0.70 | Higher is better |
| Genre Accuracy | ≥ 0.80 | Higher is better |
| Feature Coverage | ≥ 0.60 | Higher is better |
| Model Confidence | ≥ 0.75 | Higher is better |
| Synthetic Quality | ≥ 0.70 | Higher is better |
| Validation Score | ≥ 0.80 | Higher is better |

## Architecture

```
interface/
├── human_oversight.py      # Core engine + Flask app (2600+ lines)
├── templates/              # HTML templates
│   ├── dashboard.html      # Main dashboard
│   ├── proposals.html      # Proposal management
│   ├── metrics.html        # Metrics visualization
│   └── audit.html          # Audit logs
├── static/
│   ├── css/
│   │   └── dashboard.css   # Styling
│   └── js/
│       ├── dashboard.js    # Dashboard logic
│       ├── proposals.js    # Proposal management
│       ├── metrics.js      # Metrics visualization
│       └── audit.js        # Audit log viewer
└── human_oversight.db      # SQLite database (auto-created)
```

## Integration with System

### 1. Gap Detection Integration

When the system detects a reconstruction gap:

```python
# In gap detection module
from interface.human_oversight import HumanOversightEngine, ExpansionSource

engine = HumanOversightEngine()

# Create proposal from gap
proposal_id = engine.create_proposal(
    parameter_name=detected_param_name,
    parameter_path=detected_param_path,
    parameter_type=inferred_type,
    description=llm_description,
    source=ExpansionSource.GAP_DETECTION,
    gap_context={
        'failed_midi': failed_midi_path,
        'reconstruction_error': error_value,
        'missing_features': missing_features_list
    },
    llm_reasoning=llm_reasoning,
    priority=calculate_priority(error_value)
)
```

### 2. Parameter Registry Integration

Automatically syncs with universal parameter registry:

```python
from parameters.universal_registry import REGISTRY

# System automatically checks current parameter count
current_count = len(REGISTRY.get_all_parameters())
```

### 3. Training Pipeline Integration

After approval, integrate with training pipeline:

```python
# Monitor approved proposals
approved = engine.get_proposals(status=ExpansionStatus.APPROVED)

for proposal in approved:
    # Generate synthetic data
    synthetic_midi = generate_synthetic_data(proposal)

    # Record training data
    engine.record_training_data(
        proposal_id=proposal['proposal_id'],
        midi_file_path=synthetic_midi,
        parameter_value=extracted_value,
        synthetic_score=quality_score,
        validation_passed=True
    )

    # Update status
    engine.update_proposal_status(
        proposal_id=proposal['proposal_id'],
        new_status=ExpansionStatus.TRAINING
    )
```

## Security Considerations

1. **Authentication**: Add authentication middleware for production
2. **API Keys**: Implement API key authentication for REST endpoints
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **Input Validation**: All inputs are validated before database insertion
5. **SQL Injection**: SQLAlchemy ORM prevents SQL injection
6. **XSS**: Templates escape user input by default

## Performance

- **Database**: SQLite with connection pooling
- **Caching**: Statistics cached for 5 seconds
- **WebSocket**: Async notifications for real-time updates
- **Pagination**: API supports limit/offset pagination
- **Efficient Queries**: Indexed database columns

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :5000

# Use different port
python human_oversight.py --server --port 8080
```

### WebSocket not connecting
- Check CORS settings
- Verify Socket.IO client version matches server
- Check firewall rules

### Database locked
- Ensure only one server instance is running
- Check file permissions on database file

## Future Enhancements

1. **User Authentication**: Multi-user support with roles
2. **Email Notifications**: Alert users of pending proposals
3. **Proposal Comments**: Discussion threads on proposals
4. **A/B Testing**: Compare parameter variants
5. **Model Performance**: Track XGBoost model performance over time
6. **Advanced Analytics**: ML-powered insights and recommendations
7. **Mobile App**: Native mobile interface
8. **Slack/Discord Integration**: Notifications in team channels

## License

MIT License - See main repository for details

## Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review audit logs for debugging

## Author

Agent 29 - Human-in-Loop Interface
Part of the 35-Agent Self-Expanding Music Generation System
