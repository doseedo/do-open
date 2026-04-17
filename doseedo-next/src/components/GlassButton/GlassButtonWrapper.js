import './GlassButtonWrapper.css';

/**
 * GlassButtonWrapper - Simple button wrapper
 * Glass effect is applied via CSS when theme-glass is active
 * Removed react-glass-ui dependency for performance
 */
function GlassButtonWrapper({ children, className = '', style = {}, ...props }) {
  return (
    <button className={className} style={style} {...props}>
      {children}
    </button>
  );
}

export default GlassButtonWrapper;
