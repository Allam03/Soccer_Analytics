const ApiService = {
  baseUrl: '/api',
  teamId: null,

  async request(path) {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`API request failed: ${url}`);
    return res.json();
  },

  async loadTeams() {
    return this.request('/options/teams');
  },

  async loadDashboard() {
    return this.request(`/dashboard?team_id=${this.teamId}`);
  },

  async loadPlayer() {
    return this.request(`/player-efficiency?team_id=${this.teamId}`);
  },

  async loadCohesion() {
    return this.request(`/team-cohesion?team_id=${this.teamId}`);
  },

  async loadInjury() {
    return this.request(`/injury-risk?team_id=${this.teamId}`);
  },

  async loadEnvironment() {
    return this.request(`/environment-impact?team_id=${this.teamId}`);
  },

  async loadWinProb() {
    return this.request(`/win-probability?team_id=${this.teamId}`);
  },
};
