import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import styles from './DSPGraphEditor.module.css';

const DSPIONode = memo(({ data }) => {
  const isInput = data.direction === 'input';

  return (
    <div className={styles.ioNode}>
      <div className={styles.ioNodeIcon}>
        <i className={`fa-solid ${isInput ? 'fa-right-to-bracket' : 'fa-right-from-bracket'}`} />
      </div>
      <span className={styles.ioNodeLabel}>{data.label}</span>

      {isInput ? (
        <Handle
          type="source"
          position={Position.Right}
          id="audio_out"
          className={styles.handleAudioOut}
          style={{ top: '50%' }}
        />
      ) : (
        <Handle
          type="target"
          position={Position.Left}
          id="audio_in"
          className={styles.handleAudioIn}
          style={{ top: '50%' }}
        />
      )}
    </div>
  );
});

DSPIONode.displayName = 'DSPIONode';
export default DSPIONode;
