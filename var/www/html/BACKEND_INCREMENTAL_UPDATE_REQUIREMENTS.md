# Backend API Changes for Incremental Track Display

## Overview
This document describes the required changes to the Python backend API to support incremental track display in monophonic mode.

## Current Behavior
- Backend generates all voices sequentially
- Task status endpoint only returns results when ALL voices are complete
- Frontend waits for entire generation to finish before displaying any tracks

## Required Changes

### 1. Initial Response (`POST /api/generate-ace-step`)

The initial response should include the expected number of voices:

```json
{
  "task_id": "abc123",
  "expected_voices": 4,
  "status": "processing"
}
```

**New field:**
- `expected_voices`: Integer - The number of voices that will be generated in monophonic mode

### 2. Task Status Endpoint (`GET /api/generate-ace-step/task/{taskId}`)

The status endpoint should return partial results as voices complete:

#### While Processing (with partial results):
```json
{
  "status": "processing",
  "completed_voices": [
    "/download/task_abc123/voice_0.wav",
    "/download/task_abc123/voice_1.wav"
  ],
  "total_voices": 4,
  "progress": 0.5
}
```

#### When Complete:
```json
{
  "status": "completed",
  "result": [
    "/download/task_abc123/voice_0.wav",
    "/download/task_abc123/voice_1.wav",
    "/download/task_abc123/voice_2.wav",
    "/download/task_abc123/voice_3.wav"
  ]
}
```

**New fields:**
- `completed_voices`: Array of strings - File paths of voices that have completed generation (only present while processing)
- `total_voices`: Integer - Total number of voices being generated
- `progress`: Float (0-1) - Generation progress percentage

### 3. Implementation Notes

#### Task State Management
The backend should maintain task state that includes:
- Array of completed voice file paths
- Total expected voices
- Current progress

#### Voice Completion Updates
When each voice finishes generating:
1. Save the generated audio file
2. Append the file path to `completed_voices` array in task state
3. Update progress percentage
4. Frontend will detect new items in `completed_voices` on next poll

#### Example Python Implementation (Pseudocode)

```python
class TaskState:
    def __init__(self, task_id, total_voices):
        self.task_id = task_id
        self.status = "processing"
        self.completed_voices = []
        self.total_voices = total_voices
        self.error = None

    def add_completed_voice(self, file_path):
        self.completed_voices.append(file_path)

    def mark_complete(self):
        self.status = "completed"

    def to_dict(self):
        if self.status == "completed":
            return {
                "status": "completed",
                "result": self.completed_voices
            }
        else:
            return {
                "status": self.status,
                "completed_voices": self.completed_voices,
                "total_voices": self.total_voices,
                "progress": len(self.completed_voices) / self.total_voices
            }

# In the generation task:
async def generate_monophonic_voices(task_id, params):
    task_state = get_task_state(task_id)

    for voice_idx in range(params.num_voices):
        # Generate single voice
        voice_audio = generate_voice(voice_idx, params)

        # Save to file
        file_path = f"/download/{task_id}/voice_{voice_idx}.wav"
        save_audio(voice_audio, file_path)

        # Update task state (this allows incremental updates)
        task_state.add_completed_voice(file_path)
        update_task_state(task_id, task_state)

    # Mark as complete
    task_state.mark_complete()
    update_task_state(task_id, task_state)
```

## Benefits

1. **Better UX**: Users see tracks appear as they're generated instead of waiting
2. **Progress Visibility**: Clear indication of generation progress
3. **Perceived Performance**: Feels faster even though generation time is the same
4. **Early Access**: Users can start listening to completed tracks while others generate

## Backward Compatibility

The changes are backward compatible:
- Frontend still works if backend doesn't provide `completed_voices`
- Existing non-monophonic mode continues to work as before
- If `completed_voices` is not present, frontend falls back to original behavior

## Testing

Test cases:
1. Generate with monophonic mode enabled → Should see placeholders, then tracks appear one by one
2. Generate without monophonic mode → Should work as before (all tracks at once)
3. Network interruption during generation → Should handle gracefully
4. Backend doesn't support incremental updates → Should fall back to original behavior
