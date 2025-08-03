// src/api.js
const BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

export const API = {
    register: (email) =>
        fetch(`${BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email }),
        }).then(r => r.json()),

    setPrefs: (user_id, prefs) =>
        fetch(`${BASE}/preferences/set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id, preferences: prefs }),
        }).then(r => r.json()),

    recommend: (req) =>
        fetch(`${BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
        }).then(r => r.json()),

    feedback: (fb) =>
        fetch(`${BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(fb),
        }).then(r => r.json()),

    getBandits: () =>
        fetch(`${BASE}/analytics/contextual-bandits`).then(r => r.json()),

    getAB: () =>
        fetch(`${BASE}/analytics/ab-testing`).then(r => r.json()),
};

