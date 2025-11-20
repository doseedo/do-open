// Metrics Visualization JavaScript

let metricTrendsChart = null;
let passFailChart = null;
let distributionChart = null;

document.addEventListener('DOMContentLoaded', () => {
    console.log('Metrics page initializing...');
    initializeCharts();
    loadMetrics();
});

async function loadMetrics() {
    try {
        const response = await fetch('/api/statistics');
        const stats = await response.json();

        updateMetricsSummary(stats.quality_metrics || {});
        updateCharts(stats.quality_metrics || {});
        updateMetricsTable(stats.quality_metrics || {});
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

function updateMetricsSummary(metrics) {
    // Reconstruction Error
    if (metrics.reconstruction_error) {
        const avg = metrics.reconstruction_error.avg;
        document.getElementById('reconErrorValue').textContent = avg.toFixed(3);
        updateGauge('reconErrorGauge', avg, 0.15, true); // Lower is better
    }

    // Musical Coherence
    if (metrics.musical_coherence) {
        const avg = metrics.musical_coherence.avg;
        document.getElementById('coherenceValue').textContent = (avg * 100).toFixed(1) + '%';
        updateGauge('coherenceGauge', avg, 0.70, false); // Higher is better
    }

    // Genre Accuracy
    if (metrics.genre_accuracy) {
        const avg = metrics.genre_accuracy.avg;
        document.getElementById('accuracyValue').textContent = (avg * 100).toFixed(1) + '%';
        updateGauge('accuracyGauge', avg, 0.80, false); // Higher is better
    }
}

function updateGauge(canvasId, value, threshold, lowerIsBetter) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 10;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 20;
    ctx.strokeStyle = '#e9ecef';
    ctx.stroke();

    // Determine color based on threshold
    let color;
    const passed = lowerIsBetter ? value <= threshold : value >= threshold;
    color = passed ? '#198754' : '#dc3545';

    // Draw value arc
    ctx.beginPath();
    const angleRange = Math.PI; // 180 degrees
    const angle = Math.PI + (value * angleRange);
    ctx.arc(centerX, centerY, radius, Math.PI, angle);
    ctx.lineWidth = 20;
    ctx.strokeStyle = color;
    ctx.stroke();

    // Draw threshold marker
    const thresholdAngle = Math.PI + (threshold * angleRange);
    ctx.beginPath();
    const markerStartX = centerX + (radius - 15) * Math.cos(thresholdAngle);
    const markerStartY = centerY + (radius - 15) * Math.sin(thresholdAngle);
    const markerEndX = centerX + (radius + 15) * Math.cos(thresholdAngle);
    const markerEndY = centerY + (radius + 15) * Math.sin(thresholdAngle);
    ctx.moveTo(markerStartX, markerStartY);
    ctx.lineTo(markerEndX, markerEndY);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#6c757d';
    ctx.stroke();
}

function initializeCharts() {
    // Metric Trends Chart
    const trendsCtx = document.getElementById('metricTrendsChart');
    if (trendsCtx) {
        metricTrendsChart = new Chart(trendsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Reconstruction Quality',
                        data: [],
                        borderColor: '#198754',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Musical Coherence',
                        data: [],
                        borderColor: '#0dcaf0',
                        backgroundColor: 'rgba(13, 202, 240, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Genre Accuracy',
                        data: [],
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4
                    }
                ]
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
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
    }

    // Pass/Fail Chart
    const passFailCtx = document.getElementById('passFailChart');
    if (passFailCtx) {
        passFailChart = new Chart(passFailCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Passed',
                        data: [],
                        backgroundColor: '#198754'
                    },
                    {
                        label: 'Failed',
                        data: [],
                        backgroundColor: '#dc3545'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    x: {
                        stacked: true
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Distribution Chart
    const distCtx = document.getElementById('distributionChart');
    if (distCtx) {
        distributionChart = new Chart(distCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Score',
                    data: [],
                    backgroundColor: [
                        '#0d6efd',
                        '#198754',
                        '#0dcaf0',
                        '#ffc107',
                        '#dc3545'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
    }
}

function updateCharts(metrics) {
    // Update metric trends (simplified - just show current values)
    if (metricTrendsChart) {
        const now = new Date().toLocaleTimeString();

        // Keep last 20 data points
        if (metricTrendsChart.data.labels.length > 20) {
            metricTrendsChart.data.labels.shift();
            metricTrendsChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        metricTrendsChart.data.labels.push(now);

        // Reconstruction Quality (inverse of error)
        const reconQuality = metrics.reconstruction_error
            ? Math.max(0, 1 - metrics.reconstruction_error.avg)
            : 0;
        metricTrendsChart.data.datasets[0].data.push(reconQuality);

        // Musical Coherence
        const coherence = metrics.musical_coherence?.avg || 0;
        metricTrendsChart.data.datasets[1].data.push(coherence);

        // Genre Accuracy
        const accuracy = metrics.genre_accuracy?.avg || 0;
        metricTrendsChart.data.datasets[2].data.push(accuracy);

        metricTrendsChart.update();
    }

    // Update pass/fail chart
    if (passFailChart) {
        const labels = [];
        const passed = [];
        const failed = [];

        for (const [key, value] of Object.entries(metrics)) {
            labels.push(formatMetricName(key));
            passed.push(value.passed || 0);
            failed.push(value.failed || 0);
        }

        passFailChart.data.labels = labels;
        passFailChart.data.datasets[0].data = passed;
        passFailChart.data.datasets[1].data = failed;
        passFailChart.update();
    }

    // Update distribution chart
    if (distributionChart) {
        const labels = [];
        const values = [];

        for (const [key, value] of Object.entries(metrics)) {
            labels.push(formatMetricName(key));
            values.push(value.avg || 0);
        }

        distributionChart.data.labels = labels;
        distributionChart.data.datasets[0].data = values;
        distributionChart.update();
    }
}

function updateMetricsTable(metrics) {
    const tbody = document.getElementById('metricsTable');

    if (Object.keys(metrics).length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center text-muted">
                    No metrics recorded yet
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = Object.entries(metrics).map(([key, value]) => {
        const passRate = value.count > 0
            ? ((value.passed / value.count) * 100).toFixed(1)
            : '0.0';

        return `
            <tr>
                <td><strong>${formatMetricName(key)}</strong></td>
                <td>${value.avg.toFixed(3)}</td>
                <td>${value.min.toFixed(3)}</td>
                <td>${value.max.toFixed(3)}</td>
                <td>${value.count}</td>
                <td><span class="badge bg-success">${value.passed}</span></td>
                <td><span class="badge bg-danger">${value.failed}</span></td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1 me-2" style="height: 20px;">
                            <div class="progress-bar ${passRate >= 80 ? 'bg-success' : passRate >= 60 ? 'bg-warning' : 'bg-danger'}"
                                 style="width: ${passRate}%">
                                ${passRate}%
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

function formatMetricName(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function refreshMetrics() {
    loadMetrics();
}
