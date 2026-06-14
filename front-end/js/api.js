/* ============================================================
   api.js  — v2.2.0

   Changes vs v2.1.0
   -----------------
   - request() now logs the URL and status on every call so fetch
     failures are visible in the console without needing DevTools.
   - Non-2xx responses throw a descriptive error that includes the
     HTTP status and the response body (truncated), making it easy
     to distinguish 404 / 500 / 422 failures.
   - A validatePayload() guard checks that the returned object has
     the expected top-level shape; logs a warning if it doesn't but
     still returns the data (callers decide whether to use it).
   ============================================================ */

const ApiService = {
  baseUrl: "/api",
  teamId:  null,

  async request(path) {
    const url = `${this.baseUrl}${path}`;
    console.debug("[ApiService] GET", url);

    let res;
    try {
      res = await fetch(url);
    } catch (networkErr) {
      console.error("[ApiService] Network error for", url, networkErr);
      throw new Error(`Network error: ${networkErr.message} (is the server running?)`);
    }

    if (!res.ok) {
      let body = "";
      try { body = await res.text(); } catch (_) {}
      const truncated = body.length > 200 ? body.slice(0, 200) + "…" : body;
      console.error("[ApiService]", res.status, url, truncated);
      throw new Error(`HTTP ${res.status} from ${url}: ${truncated}`);
    }

    let data;
    try {
      data = await res.json();
    } catch (jsonErr) {
      console.error("[ApiService] JSON parse error for", url, jsonErr);
      throw new Error(`Invalid JSON from ${url}: ${jsonErr.message}`);
    }

    // Warn if source field is missing — indicates unexpected response shape
    if (data && typeof data === "object" && !("source" in data)) {
      console.warn("[ApiService] Response from", url, "has no 'source' field:", data);
    }

    console.debug("[ApiService] OK", url, "source=", data?.source);
    return data;
  },

  async loadTeams() {
    return this.request("/options/teams");
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

  async loadXG() {
    return this.request(`/xg-finishing?team_id=${this.teamId}`);
  },

  async loadWinProb() {
    return this.request(`/win-probability?team_id=${this.teamId}`);
  },
};