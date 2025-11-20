// Human Oversight Dashboard - Main JavaScript

// Initialize Socket.IO connection
const socket = io();

// Global state
let statusChart = null;
let timelineChart = null;
let notifications = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');

    // Initialize WebSocket
    initializeWebSocket();

    // Load initial data
    loadStatistics();
    loadRecentActivity();

    // Initialize charts
    initializeCharts();

    // Set up auto-refresh
    setInterval(loadStatistics, 30000); // Refresh every 30 seconds
    setInterval(loadRecentActivity, 60000); // Refresh every minute
});

// ============================================================================
// WEBSOCKET HANDLING
// ============================================================================

function initializeWebSocket() {
    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
        socket.emit('subscribe', { room: 'general' });
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    socket.on('notification', (data) => {
        console.log('Received notification:', data);
        handleNotification(data);
    });

    socket.on('connected', (data) => {
        console.log('Server confirmed connection:', data);
    });
}

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connectionBadge');
    const status = document.getElementById('systemStatus');

    if (connected) {
        badge.className = 'badge bg-success ms-2';
        badge.innerHTML = '<i class="bi bi-wifi"></i> Live';
        status.textContent = 'Connected';
    } else {
        badge.className = 'badge bg-danger ms-2';
        badge.innerHTML = '<i class="bi bi-wifi-off"></i> Offline';
        status.textContent = 'Disconnected';
    }
}

function handleNotification(notification) {
    addNotification(notification);

    // Update relevant UI based on notification type
    if (notification.event_type === 'new_proposal') {
        loadStatistics();
    } else if (notification.event_type === 'proposal_approved') {
        loadStatistics();
        loadRecentActivity();
    } else if (notification.event_type === 'metric_recorded') {
        loadStatistics();
    }
}

function addNotification(notification) {
    notifications.unshift(notification);

    // Keep only last 50 notifications
    if (notifications.length > 50) {
        notifications.pop();
    }

    renderNotifications();
}

function renderNotifications() {
    const container = document.getElementById('notifications');

    if (notifications.length === 0) {
        container.innerHTML = `
            <p class="text-muted text-center">
                <i class="bi bi-info-circle"></i>
                Waiting for notifications...
            </p>
        `;
        return;
    }

    container.innerHTML = notifications.map(notif => {
        const time = new Date(notif.timestamp).toLocaleTimeString();
        const type = getNotificationType(notif.event_type);

        return `
            <div class="notification-item ${type}">
                <div class="d-flex justify-content-between">
                    <strong>${formatEventType(notif.event_type)}</strong>
                    <span class="notification-time">${time}</span>
                </div>
                <div class="mt-1">
                    ${formatNotificationData(notif)}
                </div>
            </div>
        `;
    }).join('');
}

function getNotificationType(eventType) {
    if (eventType.includes('approved') || eventType.includes('success')) return 'success';
    if (eventType.includes('rejected') || eventType.includes('failed')) return 'error';
    if (eventType.includes('pending')) return 'warning';
    return '';
}

function formatEventType(eventType) {
    return eventType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatNotificationData(notif) {
    const data = notif.data;

    if (notif.event_type === 'new_proposal') {
        return `New proposal: <strong>${data.parameter_name}</strong>`;
    } else if (notif.event_type === 'proposal_approved') {
        return `Approved: <strong>${data.parameter_name}</strong>`;
    } else if (notif.event_type === 'proposal_rejected') {
        return `Rejected: <strong>${data.parameter_name}</strong>`;
    } else if (notif.event_type === 'metric_recorded') {
        return `Metric recorded: <strong>${data.metric_type}</strong> = ${data.metric_value.toFixed(3)}`;
    }

    return JSON.stringify(data);
}

function clearNotifications() {
    notifications = [];
    renderNotifications();
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const stats = await response.json();

        updateStatisticsUI(stats);
        updateCharts(stats);
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

function updateStatisticsUI(stats) {
    // Total parameters
    const totalParams = stats.current_parameters || 165;
    document.getElementById('totalParameters').textContent = totalParams;
    const progress = (totalParams / 515) * 100;
    document.getElementById('parameterProgress').style.width = `${progress}%`;

    // Pending proposals
    const pending = stats.proposals_by_status?.pending || 0;
    document.getElementById('pendingProposals').textContent = pending;

    // Approved today (approximate - we'll use approved count)
    const approved = stats.proposals_by_status?.approved || 0;
    document.getElementById('approvedToday').textContent = approved;

    // Health score
    const health = stats.system_health?.health_score || 95;
    document.getElementById('healthScore').textContent = Math.round(health);
    document.getElementById('healthProgress').style.width = `${health}%`;

    // Quality metrics
    updateQualityMetrics(stats.quality_metrics || {});

    // Training data
    const trainingData = stats.training_data || {};
    document.getElementById('totalSamples').textContent = trainingData.total || 0;
    document.getElementById('validatedSamples').textContent = trainingData.validated || 0;
    const validationRate = (trainingData.validation_rate || 0) * 100;
    document.getElementById('validationRate').textContent = `${validationRate.toFixed(1)}%`;
}

function updateQualityMetrics(metrics) {
    // Reconstruction quality (inverse of error)
    const reconError = metrics.reconstruction_error?.avg || 0;
    const reconQuality = Math.max(0, 1 - reconError);
    updateMetricBar('reconQuality', reconQuality);

    // Musical coherence
    const coherence = metrics.musical_coherence?.avg || 0;
    updateMetricBar('coherence', coherence);

    // Genre accuracy
    const accuracy = metrics.genre_accuracy?.avg || 0;
    updateMetricBar('genreAccuracy', accuracy);

    // Model confidence
    const confidence = metrics.model_confidence?.avg || 0;
    updateMetricBar('modelConf', confidence);

    // Validation score
    const validation = metrics.validation_score?.avg || 0;
    updateMetricBar('validationScore', validation);
}

function updateMetricBar(metricId, value) {
    const valueElement = document.getElementById(metricId);
    const barElement = document.getElementById(metricId + 'Bar');

    if (valueElement) {
        valueElement.textContent = (value * 100).toFixed(1) + '%';
    }

    if (barElement) {
        barElement.style.width = `${value * 100}%`;
    }
}

async function loadRecentActivity() {
    try {
        const response = await fetch('/api/proposals?limit=10');
        const proposals = await response.json();

        renderRecentActivity(proposals);
    } catch (error) {
        console.error('Error loading recent activity:', error);
    }
}

function renderRecentActivity(proposals) {
    const tbody = document.getElementById('recentActivity');

    if (proposals.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">
                    No recent activity
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = proposals.map(proposal => {
        const time = formatTimeAgo(proposal.created_at);
        const statusBadge = getStatusBadge(proposal.status);
        const sourceBadge = getSourceBadge(proposal.source);

        return `
            <tr>
                <td>${time}</td>
                <td>
                    <strong>${proposal.parameter_name}</strong>
                    <br>
                    <small class="text-muted">${proposal.parameter_path}</small>
                </td>
                <td>${statusBadge}</td>
                <td>${sourceBadge}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="viewProposal('${proposal.proposal_id}')">
                        <i class="bi bi-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function formatTimeAgo(dateString) {
    if (!dateString) return 'Unknown';

    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

function getStatusBadge(status) {
    const badges = {
        'pending': '<span class="badge status-pending">Pending</span>',
        'approved': '<span class="badge status-approved">Approved</span>',
        'rejected': '<span class="badge status-rejected">Rejected</span>',
        'implemented': '<span class="badge status-implemented">Implemented</span>',
        'active': '<span class="badge status-active">Active</span>',
        'failed': '<span class="badge status-failed">Failed</span>',
        'training': '<span class="badge status-training">Training</span>',
    };

    return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
}

function getSourceBadge(source) {
    const badges = {
        'gap_detection': '<span class="badge source-gap">Gap Detection</span>',
        'llm_proposal': '<span class="badge source-llm">LLM</span>',
        'human_request': '<span class="badge source-human">Human</span>',
        'genre_analysis': '<span class="badge source-genre">Genre</span>',
    };

    return badges[source] || `<span class="badge bg-secondary">${source}</span>`;
}

// ============================================================================
// CHARTS
// ============================================================================

function initializeCharts() {
    // Status chart
    const statusCtx = document.getElementById('statusChart');
    if (statusCtx) {
        statusChart = new Chart(statusCtx, {
            type: 'doughnut',
            data: {
                labels: ['Pending', 'Approved', 'Rejected', 'Implemented', 'Active', 'Failed'],
                datasets: [{
                    data: [0, 0, 0, 0, 0, 0],
                    backgroundColor: [
                        '#ffc107',
                        '#198754',
                        '#dc3545',
                        '#0d6efd',
                        '#20c997',
                        '#6c757d'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Timeline chart
    const timelineCtx = document.getElementById('timelineChart');
    if (timelineCtx) {
        timelineChart = new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Parameters',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }
}

function updateCharts(stats) {
    // Update status chart
    if (statusChart && stats.proposals_by_status) {
        const statuses = stats.proposals_by_status;
        statusChart.data.datasets[0].data = [
            statuses.pending || 0,
            statuses.approved || 0,
            statuses.rejected || 0,
            statuses.implemented || 0,
            statuses.active || 0,
            statuses.failed || 0
        ];
        statusChart.update();
    }

    // Update timeline chart (simplified - just show current parameter count)
    if (timelineChart) {
        const currentParams = stats.current_parameters || 165;
        const now = new Date().toLocaleTimeString();

        // Keep last 20 data points
        if (timelineChart.data.labels.length > 20) {
            timelineChart.data.labels.shift();
            timelineChart.data.datasets[0].data.shift();
        }

        timelineChart.data.labels.push(now);
        timelineChart.data.datasets[0].data.push(currentParams);
        timelineChart.update();
    }
}

// ============================================================================
// UI ACTIONS
// ============================================================================

function refreshActivity() {
    loadRecentActivity();
}

function viewProposal(proposalId) {
    window.location.href = `/proposals?id=${proposalId}`;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showToast(message, type = 'info') {
    // Simple toast notification (could be enhanced with Bootstrap toasts)
    console.log(`[${type.toUpperCase()}] ${message}`);
}
