import React, { useRef, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

const menuStyle = {
  position: 'fixed',
  zIndex: 10000,
  background: '#1e1e3a',
  border: '1px solid rgba(186, 156, 255, 0.25)',
  borderRadius: 8,
  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  minWidth: 160,
  padding: '4px 0',
  overflow: 'hidden',
};

const itemStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  width: '100%',
  padding: '7px 14px',
  border: 'none',
  background: 'none',
  color: 'rgba(255,255,255,0.8)',
  fontSize: 12,
  cursor: 'pointer',
  textAlign: 'left',
  fontFamily: 'inherit',
  transition: 'background 0.1s',
};

const separatorStyle = {
  height: 1,
  background: 'rgba(255,255,255,0.08)',
  margin: '4px 0',
};

const ContextMenu = ({ x, y, items, onClose }) => {
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ x, y });

  // Adjust position to keep menu on screen
  useEffect(() => {
    if (!menuRef.current) return;
    const rect = menuRef.current.getBoundingClientRect();
    const adjusted = { x, y };
    if (x + rect.width > window.innerWidth - 8) adjusted.x = window.innerWidth - rect.width - 8;
    if (y + rect.height > window.innerHeight - 8) adjusted.y = window.innerHeight - rect.height - 8;
    if (adjusted.x < 8) adjusted.x = 8;
    if (adjusted.y < 8) adjusted.y = 8;
    setPos(adjusted);
  }, [x, y]);

  // Close on escape or click outside
  useEffect(() => {
    const handleEscape = (e) => { if (e.key === 'Escape') onClose(); };
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) onClose();
    };
    document.addEventListener('keydown', handleEscape);
    // Use timeout to avoid the same click that opened the menu from closing it
    const timer = setTimeout(() => document.addEventListener('mousedown', handleClick), 0);
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('mousedown', handleClick);
      clearTimeout(timer);
    };
  }, [onClose]);

  const menu = (
    <div
      ref={menuRef}
      style={{ ...menuStyle, left: pos.x, top: pos.y }}
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item, i) => {
        if (item.separator) return <div key={`sep-${i}`} style={separatorStyle} />;
        return (
          <button
            key={item.label}
            style={itemStyle}
            onClick={() => { item.action(); onClose(); }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(186,156,255,0.12)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'none'; }}
            disabled={item.disabled}
          >
            {item.icon && <i className={`fa-solid ${item.icon}`} style={{ fontSize: 11, width: 14, textAlign: 'center', opacity: 0.6 }} />}
            <span style={{ flex: 1 }}>{item.label}</span>
            {item.shortcut && (
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginLeft: 16 }}>{item.shortcut}</span>
            )}
          </button>
        );
      })}
    </div>
  );

  return createPortal(menu, document.body);
};

export default ContextMenu;
