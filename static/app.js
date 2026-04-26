// app.js — Autonomy Calibration Environment UI
// Communicates with FastAPI backend at /api/reset and /api/step

const API = '/api';

let currentScenario = null;
let sessionStats = { episodes: 0, correct: 0, totalReward: 0 };

// ─── DOM REFERENCES ──────────────────────────────────────────
const elCategory    = document.getElementById('category-tag');
const elContext     = document.getElementById('context-text');
const elTask        = document.getElementById('task-text');
const elAction      = document.getElementById('action-text');
const elHistory     = document.getElementById('history-list');
const elScenarioId  = document.getElementById('scenario-id');
const elLoading     = document.getElementById('loading');
const elContent     = document.getElementById('content');
const elDoneBanner  = document.getElementById('done-banner');
const elButtons     = document.querySelectorAll('.decision-btn');

const elStatEpisodes = document.getElementById('stat-episodes');
const elStatCorrect  = document.getElementById('stat-correct');
const elStatReward   = document.getElementById('stat-reward');

const modal          = document.getElementById('modal-overlay');
const elVerdict      = document.getElementById('modal-verdict');
const elModalSub     = document.getElementById('modal-subtitle');
const elModalTotal   = document.getElementById('modal-total');
const elRewardRows   = document.getElementById('reward-rows');
const elBestDecision = document.getElementById('best-decision');

// ─── GPU TRAINING ──────────────────────────────────────────────
async function startTraining() {
    const btn = document.getElementById('train-btn');
    const status = document.getElementById('train-status');
    
    btn.disabled = true;
    btn.innerHTML = "⏳ Initializing...";
    status.innerHTML = "Requesting GPU compute resources...";
    status.style.display = 'block';
    status.style.color = '#9fa8da';

    try {
        const response = await fetch('/api/train', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'started') {
            btn.innerHTML = "⚙️ Training in Progress";
            status.innerHTML = `Success: Training started on ${data.device}.<br>Check logs for progress.`;
            status.style.color = '#81c784';
        } else {
            throw new Error(data.detail || "Failed to start training");
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = "🚀 Start GPU Training";
        status.innerHTML = `Error: ${err.message}`;
        status.style.color = '#e57373';
    }
}


// ─── STARTUP ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', startNewEpisode);


// ─── EPISODE MANAGEMENT ──────────────────────────────────────
async function startNewEpisode() {
  elLoading.style.display = 'block';
  elContent.style.display = 'none';
  elDoneBanner.classList.remove('visible');
  setButtonsEnabled(true);

  try {
    const res = await fetch(`${API}/reset`, { method: 'POST' });
    if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
    const data = await res.json();
    currentScenario = data;
    renderScenario(data);
    elLoading.style.display = 'none';
    elContent.style.display = 'block';
    sessionStats.episodes++;
    updateStats();
  } catch (err) {
    elLoading.textContent = `Error loading scenario: ${err.message}`;
    console.error(err);
  }
}


// ─── RENDER SCENARIO ─────────────────────────────────────────
function renderScenario(data) {
  // Support both wrapped {observation: ...} and flat Observation structures
  const obs = data.observation || data;
  const cat = data.category || obs.task_id || 'general';

  elCategory.textContent = cat.toUpperCase();
  elCategory.className = `category-tag category-${cat}`;
  elScenarioId.textContent = data.scenario_id || `seed:${data.seed || 'none'}`;

  // Map OpenEnv fields to UI
  elContext.textContent = obs.prompt || obs.context || 'No context provided.';
  elTask.textContent    = obs.task || `Step ${obs.step}`;
  elAction.textContent  = obs.action_to_evaluate || 'Select an action to proceed.';

  // History
  elHistory.innerHTML = '';
  const history = obs.history || [];
  if (history.length === 0) {
    elHistory.innerHTML = '<div class="no-history">No previous actions in this episode</div>';
  } else {
    history.forEach(h => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <div><strong>Action:</strong> ${escHtml(h.action)}</div>
        <div class="reward"><strong>Reward:</strong> ${h.reward !== undefined ? h.reward.toFixed(2) : 'N/A'}</div>
      `;
      elHistory.appendChild(div);
    });
  }

  // Dynamic Actions (REPLACE STATIC BUTTONS)
  const container = document.querySelector('.btn-grid');
  if (container && obs.available_actions) {
    container.innerHTML = '';
    obs.available_actions.forEach(action => {
      const btn = document.createElement('button');
      btn.className = 'decision-btn btn-act'; // Generic style
      btn.innerHTML = `
        <span class="btn-icon">▶</span>
        <span class="btn-label">${escHtml(action)}</span>
      `;
      btn.onclick = () => submitDecision(action);
      container.appendChild(btn);
    });
  }
}



// ─── DECISION SUBMISSION ─────────────────────────────────────
async function submitDecision(decision) {
  setButtonsEnabled(false);

  const actionText = `The agent carefully considered the context and constraints.\nDECISION: ${decision}`;

  try {
    const res = await fetch(`${API}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: decision }),
    });

    if (!res.ok) {
      const errData = await res.json();
      const msg = typeof errData.detail === 'string' 
        ? errData.detail 
        : JSON.stringify(errData.detail);
      alert(`Server error: ${msg}`);
      setButtonsEnabled(true);
      return;
    }


    const data = await res.json();
    showResultModal(data, decision);

    // Update stats
    const rewardVal = data.reward?.value || 0;
    sessionStats.totalReward += rewardVal;
    // Consider it 'correct' if reward is high (>= 0.8)
    if (rewardVal >= 0.8) sessionStats.correct++;
    updateStats();


    // Show done banner
    elDoneBanner.classList.add('visible');

  } catch (err) {
    alert(`Network error: ${err.message}`);
    setButtonsEnabled(true);
  }
}


// ─── RESULT MODAL ────────────────────────────────────────────
function showResultModal(data, chosenDecision) {
  const score     = data.info?.episode_score;
  const isDone    = data.done;
  const rewardVal = data.reward?.value ?? 0;
  const breakdown = data.reward?.breakdown || {};


  // Verdict
  if (isDone && score !== undefined) {
    elVerdict.textContent = score >= 0.8 ? '🏁 COMPLETED' : '⚠️ FINISHED';
    elVerdict.className = `modal-verdict ${score >= 0.8 ? 'correct' : 'wrong'}`;
    elModalSub.textContent = `Episode Score: ${(score * 100).toFixed(0)}%`;
  } else {
    elVerdict.textContent = '📥 STEP RECORDED';
    elVerdict.className = 'modal-verdict';
    elModalSub.textContent = `Action: ${chosenDecision}`;
  }

  // Total
  elModalTotal.textContent = `Step Reward: ${rewardVal.toFixed(2)}`;
  elModalTotal.style.color = rewardVal >= 0.5 ? '#66bb6a' : '#ef5350';


  // Reward breakdown rows
  const LABELS = {
    r1_action_correctness:  'Action Correctness',
    r2_risk_calibration:    'Risk Calibration',
    r3_constraint_adherence:'Constraint Adherence',
    r4_failure_awareness:   'Failure Awareness',
    r5_over_caution_penalty:'Over-Caution Penalty',
    r6_recovery_quality:    'Recovery Quality',
    r_action_correct:       'Action Alignment',
    r_safety:               'Safety Factor',
    r_efficiency:           'Efficiency',
    r_legitimate:           'Legitimacy Check'
  };

  elRewardRows.innerHTML = '';
  Object.entries(breakdown).forEach(([key, val]) => {
    const label = LABELS[key] || key;
    if (val === 0) return; 
    const sign = val > 0 ? '+' : '';
    const cls  = val > 0 ? 'pos' : val < 0 ? 'neg' : 'zero';
    const row  = document.createElement('div');
    row.className = 'reward-row';
    row.innerHTML = `
      <span class="r-name">${label}</span>
      <span class="r-val ${cls}">${sign}${val.toFixed(2)}</span>
    `;
    elRewardRows.appendChild(row);
  });

  // Best decision reveal (if provided in info)
  const best = data.info?.best_action || data.info?.best_decision;
  if (best) {
    elBestDecision.style.display = 'block';
    elBestDecision.innerHTML = `Suggested action was: <strong>${best}</strong>`;
  } else {
    elBestDecision.style.display = 'none';
  }


  // Show modal
  modal.classList.add('visible');
}


// ─── UTILITIES ───────────────────────────────────────────────
function closeModal() {
  modal.classList.remove('visible');
}

function setButtonsEnabled(enabled) {
  elButtons.forEach(btn => {
    btn.disabled = !enabled;
    btn.style.opacity = enabled ? '1' : '0.4';
    btn.style.cursor  = enabled ? 'pointer' : 'not-allowed';
  });
}

function updateStats() {
  elStatEpisodes.textContent = sessionStats.episodes;
  elStatCorrect.textContent  = sessionStats.correct;
  const avg = sessionStats.episodes > 0
    ? (sessionStats.totalReward / sessionStats.episodes).toFixed(1)
    : '0.0';
  const sign = parseFloat(avg) >= 0 ? '+' : '';
  elStatReward.textContent  = `${sign}${avg}`;
  elStatReward.className    = `stat-value ${parseFloat(avg) >= 0 ? 'positive' : 'negative'}`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
