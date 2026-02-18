import React from 'react';

/**
 * TimelineTick Component - Single tick mark on timeline
 * Pure presentational component (memoized for performance)
 */
const TimelineTick = React.memo(({ time, position, label, isMajor = true }) => {
  return (
    <>
      <div
        className={`tick ${isMajor ? 'major' : 'minor'}`}
        style={{
          position: 'absolute',
          left: `${position}px`,
          height: isMajor ? '100%' : '50%',
          width: '1px'
        }}
      />
      {isMajor && (
        <div
          className="tick-label"
          style={{
            position: 'absolute',
            top: '10px',
            left: `${position + 3}px`,
            fontSize: '10px',
            color: '#ccc'
          }}
        >
          {label}
        </div>
      )}
    </>
  );
});

TimelineTick.displayName = 'TimelineTick';

export default TimelineTick;
