import React from 'react';
import { BaseEdge, getSmoothStepPath } from '@xyflow/react';

export function DSPAudioEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd }) {
  const [edgePath] = getSmoothStepPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
    borderRadius: 10,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        stroke: '#4fc3f7',
        strokeWidth: 2.5,
        opacity: 0.8,
      }}
    />
  );
}

export function DSPModulationEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd }) {
  const [edgePath] = getSmoothStepPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
    borderRadius: 10,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        stroke: '#ba68c8',
        strokeWidth: 1.5,
        strokeDasharray: '6 3',
        opacity: 0.7,
      }}
    />
  );
}
