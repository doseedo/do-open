// Proposals Management JavaScript

let currentProposals = [];
let selectedProposals = new Set();
let currentFilters = {
    status: '',
    source: '',
    sortBy: 'created_desc',
    limit: 100,
    offset: 0
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('Proposals page initializing...');
    loadProposals();

    // Setup event listeners
    document.getElementById('reviewAction').addEventListener('change', (e) => {
        const reasonGroup = document.getElementById('reasonGroup');
        const notesGroup = document.getElementById('notesGroup');

        if (e.target.value === 'reject') {
            reasonGroup.style.display = 'block';
            notesGroup.style.display = 'none';
        } else {
            reasonGroup.style.display = 'none';
            notesGroup.style.display = 'block';
        }
    });
});

async function loadProposals() {
    try {
        const params = new URLSearchParams();
        if (currentFilters.status) params.append('status', currentFilters.status);
        if (currentFilters.source) params.append('source', currentFilters.source);
        params.append('limit', currentFilters.limit);
        params.append('offset', currentFilters.offset);

        const response = await fetch(`/api/proposals?${params}`);
        currentProposals = await response.json();

        renderProposals();
    } catch (error) {
        console.error('Error loading proposals:', error);
        showError('Failed to load proposals');
    }
}

function renderProposals() {
    const container = document.getElementById('proposalsList');
    const countElement = document.getElementById('proposalCount');

    countElement.textContent = currentProposals.length;

    if (currentProposals.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-inbox"></i>
                <h5>No Proposals Found</h5>
                <p class="text-muted">Try adjusting your filters or create a new proposal</p>
            </div>
        `;
        return;
    }

    // Sort proposals
    sortProposals();

    container.innerHTML = currentProposals.map(proposal => {
        const priority = getPriority(proposal.priority);
        const isSelected = selectedProposals.has(proposal.proposal_id);

        return `
            <div class="card proposal-card priority-${priority}" data-proposal-id="${proposal.proposal_id}">
                <div class="card-body">
                    <div class="proposal-header">
                        <div class="d-flex align-items-center">
                            ${proposal.status === 'pending' ? `
                                <input type="checkbox" class="form-check-input me-3"
                                       ${isSelected ? 'checked' : ''}
                                       onchange="toggleSelection('${proposal.proposal_id}')">
                            ` : ''}
                            <div>
                                <h5 class="proposal-title">${proposal.parameter_name}</h5>
                                <div class="proposal-meta">
                                    <i class="bi bi-tag"></i> ${proposal.parameter_path}
                                    <span class="ms-2">
                                        <i class="bi bi-clock"></i> ${formatTimeAgo(proposal.created_at)}
                                    </span>
                                    <span class="ms-2">
                                        <i class="bi bi-flag"></i> Priority: ${proposal.priority}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div>
                            ${getStatusBadge(proposal.status)}
                            ${getSourceBadge(proposal.source)}
                        </div>
                    </div>

                    <p class="proposal-description text-truncate-2">
                        ${proposal.description}
                    </p>

                    ${proposal.llm_reasoning ? `
                        <div class="alert alert-info py-2 px-3 mb-2">
                            <strong><i class="bi bi-lightbulb"></i> LLM Reasoning:</strong>
                            <span class="text-truncate-2 d-block">${proposal.llm_reasoning}</span>
                        </div>
                    ` : ''}

                    <div class="proposal-actions">
                        <button class="btn btn-sm btn-primary" onclick="viewProposalDetails('${proposal.proposal_id}')">
                            <i class="bi bi-eye"></i> Details
                        </button>

                        ${proposal.status === 'pending' ? `
                            <button class="btn btn-sm btn-success" onclick="reviewProposal('${proposal.proposal_id}', 'approve')">
                                <i class="bi bi-check-circle"></i> Approve
                            </button>
                            <button class="btn btn-sm btn-danger" onclick="reviewProposal('${proposal.proposal_id}', 'reject')">
                                <i class="bi bi-x-circle"></i> Reject
                            </button>
                        ` : ''}

                        ${proposal.status === 'approved' || proposal.status === 'implemented' ? `
                            <button class="btn btn-sm btn-info" onclick="viewMetrics('${proposal.proposal_id}')">
                                <i class="bi bi-graph-up"></i> Metrics
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function sortProposals() {
    const [field, direction] = currentFilters.sortBy.split('_');

    currentProposals.sort((a, b) => {
        let compareA, compareB;

        if (field === 'created') {
            compareA = new Date(a.created_at);
            compareB = new Date(b.created_at);
        } else if (field === 'priority') {
            compareA = a.priority;
            compareB = b.priority;
        } else {
            return 0;
        }

        if (direction === 'desc') {
            return compareB - compareA;
        } else {
            return compareA - compareB;
        }
    });
}

function applyFilters() {
    currentFilters.status = document.getElementById('filterStatus').value;
    currentFilters.source = document.getElementById('filterSource').value;
    currentFilters.sortBy = document.getElementById('sortBy').value;
    currentFilters.offset = 0;

    selectedProposals.clear();
    updateBatchButtons();

    loadProposals();
}

function toggleSelection(proposalId) {
    if (selectedProposals.has(proposalId)) {
        selectedProposals.delete(proposalId);
    } else {
        selectedProposals.add(proposalId);
    }

    updateBatchButtons();
}

function updateBatchButtons() {
    const count = selectedProposals.size;
    document.getElementById('selectedCount').textContent = count;

    const approveBtn = document.getElementById('batchApproveBtn');
    const rejectBtn = document.getElementById('batchRejectBtn');

    if (count > 0) {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
    } else {
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
    }
}

async function batchApprove() {
    if (selectedProposals.size === 0) return;

    if (!confirm(`Approve ${selectedProposals.size} proposals?`)) return;

    const notes = prompt('Optional notes for approval:');

    try {
        const response = await fetch('/api/proposals/batch/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                proposal_ids: Array.from(selectedProposals),
                reviewer_notes: notes,
                user_name: 'Dashboard User'
            })
        });

        const result = await response.json();
        showSuccess(`Approved ${result.success_count} proposals`);

        selectedProposals.clear();
        updateBatchButtons();
        loadProposals();
    } catch (error) {
        console.error('Error batch approving:', error);
        showError('Failed to batch approve');
    }
}

async function batchReject() {
    if (selectedProposals.size === 0) return;

    const reason = prompt(`Reject ${selectedProposals.size} proposals?\n\nReason:`);
    if (!reason) return;

    try {
        const response = await fetch('/api/proposals/batch/reject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                proposal_ids: Array.from(selectedProposals),
                rejection_reason: reason,
                user_name: 'Dashboard User'
            })
        });

        const result = await response.json();
        showSuccess(`Rejected ${result.success_count} proposals`);

        selectedProposals.clear();
        updateBatchButtons();
        loadProposals();
    } catch (error) {
        console.error('Error batch rejecting:', error);
        showError('Failed to batch reject');
    }
}

async function submitProposal() {
    const data = {
        parameter_name: document.getElementById('paramName').value,
        parameter_path: document.getElementById('paramPath').value,
        parameter_type: document.getElementById('paramType').value,
        description: document.getElementById('paramDesc').value,
        source: document.getElementById('paramSource').value,
        priority: parseInt(document.getElementById('paramPriority').value),
        llm_reasoning: document.getElementById('paramReasoning').value,
        expected_impact: document.getElementById('paramImpact').value
    };

    try {
        const response = await fetch('/api/proposals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        showSuccess('Proposal created successfully!');

        // Close modal
        bootstrap.Modal.getInstance(document.getElementById('createProposalModal')).hide();

        // Reset form
        document.getElementById('createProposalForm').reset();

        // Reload proposals
        loadProposals();
    } catch (error) {
        console.error('Error creating proposal:', error);
        showError('Failed to create proposal');
    }
}

async function viewProposalDetails(proposalId) {
    try {
        const response = await fetch(`/api/proposals/${proposalId}`);
        const proposal = await response.json();

        const metricsResponse = await fetch(`/api/proposals/${proposalId}/metrics`);
        const metrics = await metricsResponse.json();

        const content = `
            <div class="row">
                <div class="col-md-8">
                    <h4>${proposal.parameter_name}</h4>
                    <p><strong>Path:</strong> ${proposal.parameter_path}</p>
                    <p><strong>Type:</strong> ${proposal.parameter_type}</p>
                    <p><strong>Description:</strong> ${proposal.description}</p>

                    ${proposal.llm_reasoning ? `
                        <div class="alert alert-info">
                            <h6>LLM Reasoning:</h6>
                            <p>${proposal.llm_reasoning}</p>
                        </div>
                    ` : ''}

                    ${proposal.expected_impact ? `
                        <div class="alert alert-success">
                            <h6>Expected Impact:</h6>
                            <p>${proposal.expected_impact}</p>
                        </div>
                    ` : ''}

                    ${proposal.reviewer_notes ? `
                        <div class="alert alert-primary">
                            <h6>Reviewer Notes:</h6>
                            <p>${proposal.reviewer_notes}</p>
                        </div>
                    ` : ''}

                    ${proposal.rejection_reason ? `
                        <div class="alert alert-danger">
                            <h6>Rejection Reason:</h6>
                            <p>${proposal.rejection_reason}</p>
                        </div>
                    ` : ''}
                </div>
                <div class="col-md-4">
                    <h6>Metadata</h6>
                    <p><strong>Status:</strong> ${getStatusBadge(proposal.status)}</p>
                    <p><strong>Source:</strong> ${getSourceBadge(proposal.source)}</p>
                    <p><strong>Priority:</strong> ${proposal.priority}</p>
                    <p><strong>Created:</strong> ${new Date(proposal.created_at).toLocaleString()}</p>
                    ${proposal.reviewed_at ? `<p><strong>Reviewed:</strong> ${new Date(proposal.reviewed_at).toLocaleString()}</p>` : ''}
                    ${proposal.implemented_at ? `<p><strong>Implemented:</strong> ${new Date(proposal.implemented_at).toLocaleString()}</p>` : ''}

                    <h6 class="mt-3">Quality Metrics</h6>
                    ${metrics.length > 0 ? metrics.map(m => `
                        <div class="mb-2">
                            <small>${m.metric_type}:</small>
                            <strong>${m.metric_value.toFixed(3)}</strong>
                            ${m.passed ? '<span class="badge bg-success">✓</span>' : '<span class="badge bg-danger">✗</span>'}
                        </div>
                    `).join('') : '<p class="text-muted">No metrics recorded</p>'}
                </div>
            </div>
        `;

        document.getElementById('proposalDetailContent').innerHTML = content;
        document.getElementById('detailModalTitle').textContent = `Proposal: ${proposal.parameter_name}`;

        const modal = new bootstrap.Modal(document.getElementById('proposalDetailModal'));
        modal.show();
    } catch (error) {
        console.error('Error loading proposal details:', error);
        showError('Failed to load proposal details');
    }
}

function reviewProposal(proposalId, action) {
    const proposal = currentProposals.find(p => p.proposal_id === proposalId);

    document.getElementById('reviewProposalId').value = proposalId;
    document.getElementById('reviewAction').value = action;
    document.getElementById('reviewProposalInfo').innerHTML = `
        <div class="alert alert-secondary">
            <strong>${proposal.parameter_name}</strong>
            <p class="mb-0">${proposal.parameter_path}</p>
        </div>
    `;

    // Trigger change event to show/hide fields
    document.getElementById('reviewAction').dispatchEvent(new Event('change'));

    const modal = new bootstrap.Modal(document.getElementById('reviewModal'));
    modal.show();
}

async function submitReview() {
    const proposalId = document.getElementById('reviewProposalId').value;
    const action = document.getElementById('reviewAction').value;
    const notes = document.getElementById('reviewNotes').value;
    const reason = document.getElementById('rejectionReason').value;

    try {
        const endpoint = action === 'approve' ? 'approve' : 'reject';
        const body = action === 'approve'
            ? { reviewer_notes: notes, user_name: 'Dashboard User' }
            : { rejection_reason: reason, user_name: 'Dashboard User' };

        const response = await fetch(`/api/proposals/${proposalId}/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (response.ok) {
            showSuccess(`Proposal ${action}ed successfully`);
            bootstrap.Modal.getInstance(document.getElementById('reviewModal')).hide();
            loadProposals();
        } else {
            showError(`Failed to ${action} proposal`);
        }
    } catch (error) {
        console.error('Error submitting review:', error);
        showError('Failed to submit review');
    }
}

function refreshProposals() {
    loadProposals();
}

function exportProposals() {
    // Simple CSV export
    const csv = ['Parameter Name,Path,Type,Status,Source,Priority,Created'];

    currentProposals.forEach(p => {
        csv.push(`"${p.parameter_name}","${p.parameter_path}","${p.parameter_type}","${p.status}","${p.source}",${p.priority},"${p.created_at}"`);
    });

    const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `proposals_${new Date().toISOString()}.csv`;
    a.click();
}

// Utility functions
function getPriority(priority) {
    if (priority >= 75) return 'high';
    if (priority >= 40) return 'medium';
    return 'low';
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
        'training': '<span class="badge status-training">Training</span>'
    };
    return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
}

function getSourceBadge(source) {
    const badges = {
        'gap_detection': '<span class="badge source-gap">Gap Detection</span>',
        'llm_proposal': '<span class="badge source-llm">LLM</span>',
        'human_request': '<span class="badge source-human">Human</span>',
        'genre_analysis': '<span class="badge source-genre">Genre</span>'
    };
    return badges[source] || `<span class="badge bg-secondary">${source}</span>`;
}

function showSuccess(message) {
    alert(message); // Can be replaced with toast notifications
}

function showError(message) {
    alert('Error: ' + message);
}

function viewMetrics(proposalId) {
    window.location.href = `/metrics?proposal=${proposalId}`;
}
