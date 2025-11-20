// Audit Log JavaScript

let currentLogs = [];
let currentFilters = {
    action_type: '',
    proposal_id: '',
    limit: 100
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('Audit page initializing...');
    loadAuditLog();
});

async function loadAuditLog() {
    try {
        const params = new URLSearchParams();
        if (currentFilters.action_type) params.append('action_type', currentFilters.action_type);
        if (currentFilters.proposal_id) params.append('proposal_id', currentFilters.proposal_id);
        params.append('limit', currentFilters.limit);

        const response = await fetch(`/api/audit?${params}`);
        currentLogs = await response.json();

        renderAuditLog();
    } catch (error) {
        console.error('Error loading audit log:', error);
        showError('Failed to load audit log');
    }
}

function renderAuditLog() {
    const tbody = document.getElementById('auditLogTable');
    const countElement = document.getElementById('entryCount');

    countElement.textContent = currentLogs.length;

    if (currentLogs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted p-5">
                    <i class="bi bi-inbox" style="font-size: 3rem; opacity: 0.3;"></i>
                    <p class="mt-3">No audit entries found</p>
                    <p class="text-muted">Try adjusting your filters</p>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = currentLogs.map(log => {
        const timestamp = new Date(log.timestamp).toLocaleString();
        const actionBadge = getActionBadge(log.action_type);

        return `
            <tr>
                <td>${timestamp}</td>
                <td>${actionBadge}</td>
                <td>${log.action_description}</td>
                <td>${log.user_name || log.user_id || '<em>System</em>'}</td>
                <td>
                    ${log.proposal_id
                        ? `<code>${log.proposal_id}</code>`
                        : '<span class="text-muted">—</span>'}
                </td>
                <td>
                    ${log.details
                        ? `<button class="btn btn-sm btn-outline-secondary" onclick="viewDetails(${log.id})">
                               <i class="bi bi-info-circle"></i>
                           </button>`
                        : '<span class="text-muted">—</span>'}
                </td>
            </tr>
        `;
    }).join('');
}

function getActionBadge(actionType) {
    const badges = {
        'CREATE_PROPOSAL': '<span class="badge bg-primary">Create</span>',
        'APPROVE_PROPOSAL': '<span class="badge bg-success">Approve</span>',
        'REJECT_PROPOSAL': '<span class="badge bg-danger">Reject</span>',
        'UPDATE_STATUS': '<span class="badge bg-info">Update</span>',
        'RECORD_METRIC': '<span class="badge bg-secondary">Metric</span>'
    };

    return badges[actionType] || `<span class="badge bg-secondary">${actionType}</span>`;
}

function applyFilters() {
    currentFilters.action_type = document.getElementById('filterActionType').value;
    currentFilters.proposal_id = document.getElementById('filterProposalId').value;
    currentFilters.limit = parseInt(document.getElementById('filterLimit').value);

    loadAuditLog();
}

function viewDetails(logId) {
    const log = currentLogs.find(l => l.id === logId);
    if (!log) return;

    const details = {
        'ID': log.id,
        'Timestamp': new Date(log.timestamp).toLocaleString(),
        'Action Type': log.action_type,
        'Description': log.action_description,
        'User ID': log.user_id || 'N/A',
        'User Name': log.user_name || 'N/A',
        'Proposal ID': log.proposal_id || 'N/A',
        'Details': log.details || {}
    };

    document.getElementById('detailsContent').textContent = JSON.stringify(details, null, 2);

    const modal = new bootstrap.Modal(document.getElementById('detailsModal'));
    modal.show();
}

function refreshAuditLog() {
    loadAuditLog();
}

function exportAuditLog() {
    // Simple CSV export
    const csv = ['Timestamp,Action,Description,User,Proposal ID'];

    currentLogs.forEach(log => {
        const timestamp = new Date(log.timestamp).toISOString();
        const user = log.user_name || log.user_id || 'System';
        const proposalId = log.proposal_id || '';

        csv.push(`"${timestamp}","${log.action_type}","${log.action_description}","${user}","${proposalId}"`);
    });

    const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit_log_${new Date().toISOString()}.csv`;
    a.click();
}

function showError(message) {
    alert('Error: ' + message);
}
