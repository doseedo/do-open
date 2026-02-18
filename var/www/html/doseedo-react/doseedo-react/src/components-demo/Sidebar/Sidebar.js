import React from 'react';
import { useApp } from '../../context/AppContext';
import './Sidebar.css';

function Sidebar() {
  const { state, dispatch } = useApp();

  const toggleSidebar = () => {
    dispatch({ type: 'TOGGLE_SIDEBAR' });
  };

  return (
    <div
      id="resizable-sidebar"
      className={state.sidebar.isExpanded ? 'expanded' : ''}
    >
      <h3 className="nav-link" id="expand-toggle" onClick={toggleSidebar}>
        <i className="fa-solid fa-bars"></i>
      </h3>

      {state.sidebar.isExpanded && (
        <div className="menulinks">
          <br />
          <a href="index.html" className="nav-link">
            <i className="fa-solid fa-house"></i> Home
          </a>
          <br />
          <a href="dashboard.html" className="nav-link">
            <i className="fa-solid fa-chart-line"></i> Dashboard
          </a>
          <br />
          <a href="plans.html" className="nav-link">
            <i className="fa-solid fa-arrow-up-right-dots"></i> Upgrade
          </a>
          <br />
          <br />
          <hr style={{
            flex: 1,
            border: 'none',
            height: '1px',
            backgroundColor: 'rgb(90, 90, 90)',
            margin: '0 -5px'
          }} />
          <br />
          <p className="nav-linkhead2">Generation</p>
          {/* Additional sidebar links from original HTML */}
        </div>
      )}
    </div>
  );
}

export default Sidebar;
