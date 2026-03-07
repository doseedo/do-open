import React, { useState, useEffect } from 'react';

const ACCESS_KEY = '***REDACTED***';

const AdminUsers = () => {
  const [unlocked, setUnlocked] = useState(() => sessionStorage.getItem('admin_auth') === '1');
  const [passInput, setPassInput] = useState('');
  const [passError, setPassError] = useState(false);
  const [users, setUsers] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!unlocked) return;
    setLoading(true);
    fetch(`/api/admin/users?key=${ACCESS_KEY}`)
      .then(r => {
        if (!r.ok) throw new Error('Failed to load users');
        return r.json();
      })
      .then(data => {
        setUsers(data.users || []);
        setTotal(data.total || 0);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [unlocked]);

  const handleUnlock = (e) => {
    e.preventDefault();
    if (passInput === ACCESS_KEY) {
      sessionStorage.setItem('admin_auth', '1');
      setUnlocked(true);
    } else {
      setPassError(true);
    }
  };

  if (!unlocked) {
    return (
      <div style={styles.page}>
        <form onSubmit={handleUnlock} style={styles.lockForm}>
          <i className="fa-solid fa-lock" style={{ fontSize: 28, color: 'rgba(186,156,255,0.5)' }} />
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, color: '#fff' }}>Admin</h2>
          <input
            type="password"
            value={passInput}
            onChange={(e) => { setPassInput(e.target.value); setPassError(false); }}
            placeholder="Password"
            autoFocus
            style={{
              ...styles.input,
              borderColor: passError ? 'rgba(255,100,100,0.5)' : 'rgba(186,156,255,0.3)',
            }}
          />
          {passError && <p style={{ margin: 0, fontSize: 13, color: 'rgba(255,100,100,0.8)' }}>Incorrect password</p>}
          <button type="submit" style={styles.submitBtn}>Enter</button>
        </form>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff' }}>Users</h1>
          <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.4)' }}>{total} total</span>
        </div>

        {loading && (
          <div style={styles.loading}>
            <i className="fa-solid fa-spinner fa-spin" /> Loading...
          </div>
        )}

        {error && (
          <p style={{ color: 'rgba(255,100,100,0.9)', fontSize: 14 }}>{error}</p>
        )}

        {!loading && !error && (
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>ID</th>
                  <th style={styles.th}>Username</th>
                  <th style={styles.th}>Email</th>
                  <th style={styles.th}>Plan</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} style={styles.tr}>
                    <td style={styles.td}>{u.id}</td>
                    <td style={styles.td}>{u.username}</td>
                    <td style={styles.td}>{u.email}</td>
                    <td style={styles.td}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 10, fontSize: 11,
                        background: u.subscription === 'pro' ? 'rgba(102,126,234,0.2)' : 'rgba(255,255,255,0.06)',
                        color: u.subscription === 'pro' ? '#667eea' : 'rgba(255,255,255,0.4)',
                      }}>
                        {u.subscription || 'free'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    boxSizing: 'border-box',
    background: '#0d0d1a',
  },
  lockForm: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 16,
    maxWidth: 300,
    width: '100%',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    fontSize: 14,
    borderRadius: 10,
    border: '1px solid rgba(186,156,255,0.3)',
    background: 'rgba(255,255,255,0.06)',
    color: '#fff',
    outline: 'none',
    boxSizing: 'border-box',
  },
  submitBtn: {
    width: '100%',
    padding: '10px 0',
    borderRadius: 10,
    border: 'none',
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    color: '#fff',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
  container: {
    width: '100%',
    maxWidth: 800,
  },
  header: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 12,
    marginBottom: 20,
  },
  loading: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  tableWrap: {
    borderRadius: 12,
    overflow: 'hidden',
    border: '1px solid rgba(255,255,255,0.08)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 13,
  },
  th: {
    textAlign: 'left',
    padding: '10px 14px',
    color: 'rgba(255,255,255,0.4)',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    background: 'rgba(255,255,255,0.03)',
    borderBottom: '1px solid rgba(255,255,255,0.06)',
  },
  tr: {
    borderBottom: '1px solid rgba(255,255,255,0.04)',
  },
  td: {
    padding: '10px 14px',
    color: 'rgba(255,255,255,0.7)',
  },
};

export default AdminUsers;
