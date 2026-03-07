import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { CATEGORY_COLORS } from './dspNodeDefinitions';
import styles from './DSPGraphEditor.module.css';

const DSPNodeComponent = memo(({ data, selected }) => {
  const { nodeData, category, ports } = data;
  const color = CATEGORY_COLORS[category] || '#90a4ae';
  const params = nodeData.params || {};
  const paramEntries = Object.entries(params);

  return (
    <div
      className={`${styles.graphNode} ${selected ? styles.graphNodeSelected : ''}`}
      style={{ '--node-color': color }}
    >
      {/* Audio input handles (left side) */}
      {ports.inputs.map((port, i) => (
        <Handle
          key={port.id}
          type="target"
          position={Position.Left}
          id={port.id}
          className={styles.handleAudioIn}
          style={{ top: ports.inputs.length === 1 ? '50%' : `${30 + i * 25}%` }}
          title={port.label}
        />
      ))}

      {/* Param modulation input handles (top) */}
      {ports.paramInputs.map((port, i) => (
        <Handle
          key={port.id}
          type="target"
          position={Position.Top}
          id={port.id}
          className={styles.handleModIn}
          style={{ left: `${(i + 1) * 100 / (ports.paramInputs.length + 1)}%` }}
          title={`Mod: ${port.label}`}
        />
      ))}

      {/* Header */}
      <div className={styles.graphNodeHeader} style={{ background: color }}>
        <span className={styles.graphNodeType}>{nodeData.type}</span>
      </div>

      {/* Node ID */}
      <div className={styles.graphNodeId}>{nodeData.id}</div>

      {/* Params */}
      <div className={styles.graphNodeParams}>
        {paramEntries.map(([key, val]) => {
          const isBound = typeof val === 'string' && val.startsWith('@');
          return (
            <div key={key} className={styles.graphNodeParam}>
              <span className={styles.graphParamKey}>{key}</span>
              {isBound ? (
                <span className={styles.graphParamBound}>
                  <i className="fa-solid fa-link" style={{ fontSize: 7 }} /> {val.slice(1)}
                </span>
              ) : (
                <span className={styles.graphParamVal}>
                  {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(2)) : String(val)}
                </span>
              )}
            </div>
          );
        })}
        {paramEntries.length === 0 && (
          <div className={styles.graphParamEmpty}>no params</div>
        )}
      </div>

      {/* Audio output handles (right side) */}
      {ports.outputs.map((port, i) => (
        <Handle
          key={port.id}
          type="source"
          position={Position.Right}
          id={port.id}
          className={port.type === 'modulation' ? styles.handleModOut : styles.handleAudioOut}
          style={{ top: ports.outputs.length === 1 ? '50%' : `${30 + i * 25}%` }}
          title={port.label}
        />
      ))}
    </div>
  );
});

DSPNodeComponent.displayName = 'DSPNodeComponent';
export default DSPNodeComponent;
