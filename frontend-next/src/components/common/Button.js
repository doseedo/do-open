import React, { useCallback } from 'react';

/**
 * Reusable Button Component
 * Used throughout DAW for consistent styling and behavior
 */
const Button = React.memo(({
  id,
  className = '',
  onClick,
  children,
  icon,
  isActive = false,
  isDisabled = false,
  variant = 'default', // 'default', 'primary', 'danger', 'mute', 'solo'
  type = 'button',
  title,
  style = {}
}) => {
  const handleClick = useCallback((e) => {
    if (!isDisabled && onClick) {
      onClick(e);
    }
  }, [onClick, isDisabled]);

  const classes = [
    'daw-button',
    className,
    isActive ? 'active' : '',
    variant ? `variant-${variant}` : '',
    isDisabled ? 'disabled' : ''
  ].filter(Boolean).join(' ');

  return (
    <button
      id={id}
      type={type}
      className={classes}
      onClick={handleClick}
      disabled={isDisabled}
      title={title}
      style={style}
    >
      {icon && <i className={icon}></i>}
      {children}
    </button>
  );
});

Button.displayName = 'Button';

export default Button;
