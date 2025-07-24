#!/usr/bin/env python3
"""
Enhanced Email Inbox Graph Analyzer
Processes Google Takeout mbox files and creates intelligent email clusters
"""

import mailbox
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from email.utils import parsedate_to_datetime, parseaddr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from string import Template
import argparse

class EmailInboxAnalyzer:
    def __init__(self, mbox_path):
        self.mbox_path = Path(mbox_path)
        self.emails = []
        self.clusters = []
        self.user_email = None  # Will be detected from most common recipient
        
    def parse_emails(self):
        """Parse emails from mbox file"""
        print("Parsing emails from mbox file...")
        mbox = mailbox.mbox(str(self.mbox_path))
        
        for i, message in enumerate(mbox):
            if i % 1000 == 0:
                print(f"Processed {i} emails...")
                
            try:
                # Extract email metadata
                email_data = {
                    'id': i,
                    'subject': self._clean_subject(message.get('Subject', '')),
                    'from': self._extract_email(message.get('From', '')),
                    'to': self._extract_emails(message.get('To', '')),
                    'cc': self._extract_emails(message.get('Cc', '')),
                    'date': self._parse_date(message.get('Date')),
                    'body': self._extract_body(message),
                    'thread_id': message.get('Message-ID', f'thread_{i}')
                }
                
                # Skip emails without valid dates
                if email_data['date']:
                    self.emails.append(email_data)
                    
            except Exception as e:
                print(f"Error processing email {i}: {e}")
                continue
        
        print(f"Successfully parsed {len(self.emails)} emails")
        
        # Detect user's email (most common in 'to' field)
        self._detect_user_email()
        
        return self.emails
    
    def _detect_user_email(self):
        """Detect user's email address from the most common 'to' recipient"""
        to_emails = []
        for email in self.emails:
            to_emails.extend(email['to'])
        
        if to_emails:
            email_counts = Counter(to_emails)
            self.user_email = email_counts.most_common(1)[0][0]
            print(f"Detected user email: {self.user_email}")
        else:
            self.user_email = None
    
    def _clean_subject(self, subject):
        """Clean email subject line"""
        if not subject:
            return "No Subject"
        try:
            # Convert to string if it's not already
            subject = str(subject)
            # Remove Re:, Fwd:, etc.
            subject = re.sub(r'^(Re|Fwd|RE|FWD):\s*', '', subject, flags=re.IGNORECASE)
            return subject.strip()
        except:
            return "Invalid Subject"
    
    def _extract_email(self, email_str):
        """Extract email address from email string"""
        if not email_str:
            return ""
        name, addr = parseaddr(email_str)
        return addr.lower() if addr else ""
    
    def _extract_emails(self, email_str):
        """Extract list of email addresses"""
        if not email_str:
            return []
        emails = []
        for email in email_str.split(','):
            addr = self._extract_email(email.strip())
            if addr:
                emails.append(addr)
        return emails
    
    def _parse_date(self, date_str):
        """Parse email date"""
        if not date_str:
            return None
        try:
            dt = parsedate_to_datetime(date_str)
            # Convert to naive datetime to avoid comparison issues
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except:
            return None
    
    def _extract_body(self, message):
        """Extract email body text"""
        body = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        continue
        else:
            try:
                body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = ""
        return body[:1000]  # Limit body length
    
    def create_clusters(self, min_cluster_size=3, max_clusters=50):
        """Create email clusters using multiple signals"""
        print("Creating email clusters...")
        
        # Step 1: Group by similar subjects
        subject_groups = defaultdict(list)
        for email in self.emails:
            # Create a normalized subject key
            subject_key = re.sub(r'[^\w\s]', '', email['subject'].lower())
            subject_key = ' '.join(subject_key.split()[:5])  # First 5 words
            subject_groups[subject_key].append(email)
        
        # Step 2: Merge groups with similar participants
        clusters = []
        processed_emails = set()
        
        for subject_key, emails in subject_groups.items():
            if len(emails) < min_cluster_size:
                continue
                
            # Skip if emails already processed
            if any(email['id'] in processed_emails for email in emails):
                continue
            
            # Get all participants
            participants = set()
            date_range = []
            
            for email in emails:
                participants.add(email['from'])
                participants.update(email['to'])
                participants.update(email['cc'])
                if email['date']:
                    date_range.append(email['date'])
            
            # Remove empty emails and user's own email
            participants.discard('')
            if self.user_email:
                participants.discard(self.user_email)
            
            if len(participants) > 1 and date_range:
                cluster = {
                    'id': f"cluster_{len(clusters)}_{subject_key.replace(' ', '_')[:20]}",
                    'subject': subject_key.title(),
                    'emails': len(emails),
                    'participants': list(participants)[:10],  # Limit participants
                    'date_range': [min(date_range), max(date_range)],
                    'sample_subjects': list(set(email['subject'] for email in emails[:5]))
                }
                clusters.append(cluster)
                
                # Mark emails as processed
                for email in emails:
                    processed_emails.add(email['id'])
            
            if len(clusters) >= max_clusters:
                break
        
        # Step 3: Add important individual emails as single-email clusters
        remaining_emails = [e for e in self.emails if e['id'] not in processed_emails]
        important_senders = self._find_important_senders(remaining_emails)
        
        for email in remaining_emails[:20]:  # Limit individual emails
            if email['from'] in important_senders:
                # Filter out user's email from participants
                participants = [email['from']] + email['to'][:3]
                if self.user_email:
                    participants = [p for p in participants if p != self.user_email]
                
                cluster = {
                    'id': f"single_{len(clusters)}_{email['subject'].replace(' ', '_')[:20]}",
                    'subject': email['subject'],
                    'emails': 1,
                    'participants': participants,
                    'date_range': [email['date'], email['date']] if email['date'] else [datetime.now(), datetime.now()],
                    'sample_subjects': [email['subject']]
                }
                clusters.append(cluster)
        
        self.clusters = clusters
        print(f"Created {len(clusters)} clusters")
        return clusters
    
    def _find_important_senders(self, emails):
        """Find senders who sent many emails"""
        sender_counts = Counter(email['from'] for email in emails if email['from'])
        # Return senders with more than 5 emails
        return {sender for sender, count in sender_counts.items() if count > 5}
    
    def prepare_graph_data(self):
        """Prepare data for graph visualization"""
        if not self.clusters:
            raise ValueError("No clusters created. Run create_clusters() first.")
        
        # Convert clusters to the format expected by the visualization
        graph_clusters = []
        
        # Find the earliest and latest dates across all clusters
        all_dates = []
        for cluster in self.clusters:
            if cluster['date_range'] and cluster['date_range'][0] and cluster['date_range'][1]:
                all_dates.extend(cluster['date_range'])
        
        if all_dates:
            earliest_date = min(all_dates)
            latest_date = max(all_dates)
            total_days = (latest_date - earliest_date).days or 1
        else:
            earliest_date = datetime.now() - timedelta(days=365)
            latest_date = datetime.now()
            total_days = 365
        
        for cluster in self.clusters:
            if cluster['date_range'] and cluster['date_range'][0] and cluster['date_range'][1]:
                start_days = (cluster['date_range'][0] - earliest_date).days
                end_days = (cluster['date_range'][1] - earliest_date).days
            else:
                start_days = 0
                end_days = 0
            
            graph_cluster = {
                'id': cluster['id'],  # Use unique ID
                'label': cluster['subject'],  # Display name
                'emails': cluster['emails'],
                'participants': cluster['participants'],
                'spanDays': [start_days, end_days],
                'sample_subjects': cluster.get('sample_subjects', [])
            }
            graph_clusters.append(graph_cluster)
        
        return graph_clusters
    
    def generate_html(self, output_path="enhanced_inbox_graph.html"):
        """Generate the HTML visualization"""
        if not self.clusters:
            raise ValueError("No clusters created. Run create_clusters() first.")
        
        graph_data = self.prepare_graph_data()
        
        # Enhanced defaults with better parameters
        defaults = {
            "alpha_overview": 0.7,  # Slightly favor people connections
            "tau": 14,  # Two-week time decay
            "strong_w": 0.3,  # Lower threshold for strong connections
            "densityPerc": {"minimal": 0.1, "normal": 0.25, "dense": 0.8}
        }
        
        # Read the template
        template_path = Path(__file__).parent / "template.html"
        template_content = template_path.read_text(encoding='utf-8')
        
        # Calculate date range safely
        valid_dates = []
        for cluster in self.clusters:
            if cluster['date_range'] and cluster['date_range'][0] and cluster['date_range'][1]:
                valid_dates.extend(cluster['date_range'])
        
        if valid_dates:
            date_range_str = f"{min(valid_dates).strftime('%Y-%m-%d')} to {max(valid_dates).strftime('%Y-%m-%d')}"
        else:
            date_range_str = "Date range not available"
        
        # Simple string replacement instead of Template
        html = template_content
        html = html.replace('CLUSTERS_JSON_PLACEHOLDER', json.dumps(graph_data, separators=(",", ":"), default=str))
        html = html.replace('DEFAULTS_JSON_PLACEHOLDER', json.dumps(defaults, separators=(",", ":")))
        html = html.replace('TOTAL_EMAILS_PLACEHOLDER', str(len(self.emails)))
        html = html.replace('TOTAL_CLUSTERS_PLACEHOLDER', str(len(self.clusters)))
        html = html.replace('DATE_RANGE_PLACEHOLDER', date_range_str)
        
        output_file = Path(output_path)
        output_file.write_text(html, encoding='utf-8')
        print(f"Generated enhanced visualization: {output_file.resolve()}")
        return output_file
    
    def _get_enhanced_template(self):
        """Enhanced HTML template with better features"""
        return '''<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Enhanced Email Inbox Graph</title>
  <style>
    html, body { height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { display: flex; background: #f8fafc; }
    #sidebar {
      width: 350px; padding: 20px; background: white; border-right: 1px solid #e2e8f0;
      box-sizing: border-box; overflow-y: auto; box-shadow: 2px 0 4px rgba(0,0,0,0.05);
    }
    #network { flex: 1; position: relative; }
    h1 { margin: 0 0 8px; font-size: 20px; font-weight: 600; color: #1a202c; }
    .stats { background: #f7fafc; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; }
    .stats div { margin-bottom: 4px; }
    fieldset { border: 1px solid #e2e8f0; margin-bottom: 16px; padding: 12px 16px; border-radius: 8px; background: #fafafa; }
    legend { padding: 0 8px; font-weight: 500; color: #4a5568; }
    .preset-bar { margin-bottom: 20px; }
    .preset-bar button {
      margin: 0 6px 6px 0; padding: 8px 12px; border: 1px solid #cbd5e0; background: white;
      border-radius: 6px; cursor: pointer; font-size: 12px; transition: all 0.2s;
    }
    .preset-bar button:hover { background: #f7fafc; border-color: #4299e1; }
    .preset-bar button.active { background: #4299e1; color: white; border-color: #4299e1; }
    label { font-size: 14px; display: block; margin-bottom: 8px; cursor: pointer; }
    input[type="radio"], input[type="checkbox"] { margin-right: 8px; }
    #apply { 
      width: 100%; padding: 12px; background: #4299e1; color: white; border: none;
      border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;
    }
    #apply:hover { background: #3182ce; }
    .legend { font-size: 13px; margin-top: 20px; color: #4a5568; }
    .legend ul { padding-left: 18px; margin: 6px 0; }
    .info-panel {
      position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.95);
      padding: 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      font-size: 12px; max-width: 300px; display: none;
    }
  </style>
  <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
  <link href="https://unpkg.com/vis-network@9.1.2/styles/vis-network.min.css" rel="stylesheet" />
</head>
<body>

<div id="sidebar">
  <h1>📧 Email Inbox Graph</h1>
  
  <div class="stats">
    <div><strong>$TOTAL_EMAILS</strong> total emails</div>
    <div><strong>$TOTAL_CLUSTERS</strong> clusters found</div>
    <div><strong>Period:</strong> $DATE_RANGE</div>
  </div>

  <div class="preset-bar">
    <button data-preset="balanced">🎯 Balanced</button>
    <button data-preset="people">👥 People Focus</button>
    <button data-preset="time">⏰ Timeline</button>
    <button data-preset="communities">🏘️ Communities</button>
    <button data-preset="isolates">🔍 Isolates</button>
  </div>

  <fieldset>
    <legend>📊 Primary View</legend>
    <label><input type="radio" name="view" value="overview" checked> Overview (Hybrid)</label>
    <label><input type="radio" name="view" value="people"> People Connections</label>
    <label><input type="radio" name="view" value="time"> Time-based Clustering</label>
    <label><input type="radio" name="view" value="isolates"> Show Isolates Only</label>
    <label><input type="radio" name="view" value="communities"> Community Detection</label>
  </fieldset>

  <fieldset>
    <legend>🎚️ Display Density</legend>
    <label><input type="radio" name="density" value="minimal"> Minimal (Key connections)</label>
    <label><input type="radio" name="density" value="normal" checked> Normal</label>
    <label><input type="radio" name="density" value="dense"> Dense (Show all)</label>
  </fieldset>

  <fieldset>
    <legend>🎨 Node Coloring</legend>
    <label><input type="radio" name="colorBy" value="community" checked> By Community</label>
    <label><input type="radio" name="colorBy" value="participant"> By Top Participant</label>
    <label><input type="radio" name="colorBy" value="volume"> By Email Volume</label>
  </fieldset>

  <fieldset>
    <legend>🔧 Filters</legend>
    <label><input type="checkbox" id="hideWeak"> Hide weak connections</label>
    <label><input type="checkbox" id="hideIsolates"> Hide isolated nodes</label>
    <label><input type="checkbox" id="hideSingleEmail"> Hide single-email clusters</label>
  </fieldset>

  <button id="apply">🔄 Update Graph</button>

  <div class="legend">
    <p><strong>💡 How to read this graph:</strong></p>
    <ul>
      <li><strong>Nodes:</strong> Email conversation clusters</li>
      <li><strong>Size:</strong> Number of emails in cluster</li>
      <li><strong>Connections:</strong> Shared participants + timing</li>
      <li><strong>Solid lines:</strong> Strong relationships</li>
      <li><strong>Dashed lines:</strong> Weak relationships</li>
    </ul>
  </div>
</div>

<div id="network"></div>
<div id="info-panel" class="info-panel"></div>

<script>
const CLUSTERS = $CLUSTERS_JSON;
const DEFAULTS = $DEFAULTS_JSON;

// Enhanced utility functions
function jaccard(a, b) {
  const A = new Set(a), B = new Set(b);
  if (A.size === 0 && B.size === 0) return 0;
  let inter = 0;
  for (const x of A) if (B.has(x)) inter++;
  return inter / (A.size + B.size - inter);
}

function timeDecay(deltaDays, tau) {
  return Math.exp(-deltaDays / tau);
}

function midpoint(span) {
  return (span[0] + span[1]) / 2.0;
}

function topPercentEdges(edges, perc) {
  if (perc >= 1.0) return edges;
  const sorted = [...edges].sort((a, b) => b.w - a.w);
  const keep = Math.max(1, Math.round(sorted.length * perc));
  return sorted.slice(0, keep);
}

function components(nodes, edges) {
  const id2idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
  const adj = nodes.map(() => []);
  edges.forEach(e => {
    const u = id2idx[e.from], v = id2idx[e.to];
    if (u !== undefined && v !== undefined) {
      adj[u].push(v); adj[v].push(u);
    }
  });
  const comp = Array(nodes.length).fill(-1);
  let cid = 0;
  const nodeToComp = {};
  for (let i = 0; i < nodes.length; i++) {
    if (comp[i] !== -1) continue;
    const stack = [i];
    comp[i] = cid;
    while (stack.length) {
      const x = stack.pop();
      nodeToComp[nodes[x].id] = cid;
      for (const y of adj[x]) if (comp[y] === -1) {
        comp[y] = cid; stack.push(y);
      }
    }
    cid++;
  }
  return nodeToComp;
}

function build(view, densityKey, colorBy, hideWeak, hideIsolates, hideSingleEmail) {
  const tau = DEFAULTS.tau;
  const strongW = DEFAULTS.strong_w;
  const perc = DEFAULTS.densityPerc[densityKey || "normal"] || 0.25;

  // Filter clusters
  let filteredClusters = CLUSTERS;
  if (hideSingleEmail) {
    filteredClusters = filteredClusters.filter(c => c.emails > 1);
  }

  let edges = [];
  if (view !== "isolates") {
    for (let i = 0; i < filteredClusters.length; i++) {
      for (let j = i + 1; j < filteredClusters.length; j++) {
        const A = filteredClusters[i], B = filteredClusters[j];
        const people = jaccard(A.participants, B.participants);
        const ta = midpoint(A.spanDays), tb = midpoint(B.spanDays);
        const dt = Math.abs(tb - ta);
        const time = timeDecay(dt, tau);

        let w;
        if (view === "people") w = people;
        else if (view === "time") w = time;
        else w = DEFAULTS.alpha_overview * people + (1 - DEFAULTS.alpha_overview) * time;

        if (w > 0.01) {  // Minimum threshold
          edges.push({ from: A.id, to: B.id, w, people, time, dt });
        }
      }
    }
    edges = topPercentEdges(edges, perc);
  }

  if (hideWeak) edges = edges.filter(e => e.w >= strongW);

  // Calculate node degrees
  const degree = Object.fromEntries(filteredClusters.map(c => [c.id, 0]));
  edges.forEach(e => { 
    if (degree[e.from] !== undefined) degree[e.from] += e.w;
    if (degree[e.to] !== undefined) degree[e.to] += e.w;
  });
  const maxDeg = Math.max(1, ...Object.values(degree));

  // Enhanced color palette
  const palette = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
    "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#6366F1"
  ];
  
  const nodeToComp = components(filteredClusters, edges);

  // Create visual nodes
  const nodesVis = filteredClusters
    .filter(n => !(hideIsolates && degree[n.id] === 0))
    .map(n => {
      const d = degree[n.id];
      const baseSize = Math.max(20, Math.min(60, 20 + Math.sqrt(n.emails) * 8));
      const size = baseSize + (d / maxDeg) * 20;
      
      let color;
      if (colorBy === "participant") {
        const tp = n.participants[0] || "unknown";
        const hash = tp.split('').reduce((a, b) => { a = ((a << 5) - a) + b.charCodeAt(0); return a & a; }, 0);
        color = palette[Math.abs(hash) % palette.length];
      } else if (colorBy === "volume") {
        const intensity = Math.min(n.emails / 50, 1);
        const redValue = Math.round(255 * intensity);
        const blueValue = Math.round(255 * (1 - intensity));
        color = "rgb(" + redValue + ", 100, " + blueValue + ")";
      } else {
        const comp = nodeToComp[n.id] || 0;
        color = palette[comp % palette.length];
      }

      return {
        id: n.id,
        label: n.id.length > 30 ? n.id.substring(0, 30) + "..." : n.id,
        title: `<b>${n.id}</b><br/>${n.emails} emails<br/>Participants: ${n.participants.slice(0, 3).join(", ")}${n.participants.length > 3 ? "..." : ""}`,
        value: n.emails,
        size,
        color,
        font: { size: Math.max(12, Math.min(16, 8 + Math.sqrt(n.emails))), face: "Inter, Arial" }
      };
    });

  // Create visual edges
  const visibleIds = new Set(nodesVis.map(n => n.id));
  const edgesVis = edges
    .filter(e => visibleIds.has(e.from) && visibleIds.has(e.to))
    .map(e => {
      const strong = e.w >= strongW;
      return {
        from: e.from, to: e.to,
        value: e.w,
        width: strong ? 2 + 4 * e.w : 1 + 2 * e.w,
        color: strong ? "#3B82F6" : "rgba(59, 130, 246, 0.3)",
        dashes: !strong,
        smooth: { enabled: true, type: "dynamic", roundness: 0.3 },
        title: `Strength: ${e.w.toFixed(2)}<br/>People overlap: ${e.people.toFixed(2)}<br/>Time proximity: ${e.time.toFixed(2)} (${e.dt} days apart)`
      };
    });

  return { nodesVis, edgesVis };
}

// UI Management
function readControls() {
  return {
    view: document.querySelector('input[name="view"]:checked').value,
    density: document.querySelector('input[name="density"]:checked').value,
    colorBy: document.querySelector('input[name="colorBy"]:checked').value,
    hideWeak: document.getElementById('hideWeak').checked,
    hideIsolates: document.getElementById('hideIsolates').checked,
    hideSingleEmail: document.getElementById('hideSingleEmail').checked,
  };
}

function applyPreset(name) {
  // Remove active class from all buttons
  document.querySelectorAll('.preset-bar button').forEach(btn => btn.classList.remove('active'));
  document.querySelector(`[data-preset="${name}"]`).classList.add('active');

  const presets = {
    balanced: { view: "overview", density: "normal", hideWeak: false, hideIsolates: false, colorBy: "community" },
    people: { view: "people", density: "minimal", hideWeak: true, hideIsolates: false, colorBy: "participant" },
    time: { view: "time", density: "normal", hideWeak: false, hideIsolates: false, colorBy: "community" },
    communities: { view: "communities", density: "dense", hideWeak: false, hideIsolates: true, colorBy: "community" },
    isolates: { view: "isolates", density: "dense", hideWeak: false, hideIsolates: false, colorBy: "volume" }
  };

  const preset = presets[name];
  if (preset) {
    document.querySelector(`input[name="view"][value="${preset.view}"]`).checked = true;
    document.querySelector(`input[name="density"][value="${preset.density}"]`).checked = true;
    document.querySelector(`input[name="colorBy"][value="${preset.colorBy}"]`).checked = true;
    document.getElementById('hideWeak').checked = preset.hideWeak;
    document.getElementById('hideIsolates').checked = preset.hideIsolates;
  }
}

let network;
function render() {
  const controls = readControls();
  const { nodesVis, edgesVis } = build(
    controls.view, controls.density, controls.colorBy,
    controls.hideWeak, controls.hideIsolates, controls.hideSingleEmail
  );

  const container = document.getElementById('network');
  const data = { nodes: new vis.DataSet(nodesVis), edges: new vis.DataSet(edgesVis) };
  
  const options = {
    physics: {
      enabled: true,
      stabilization: { iterations: 300 },
      barnesHut: {
        gravitationalConstant: -2000,
        centralGravity: 0.3,
        springLength: 150,
        springConstant: 0.05,
        damping: 0.1,
        avoidOverlap: 0.2
      }
    },
    interaction: { hover: true, multiselect: true, tooltipDelay: 200 },
    edges: { smooth: { enabled: true, type: "dynamic", roundness: 0.3 } }
  };

  if (network) network.destroy();
  network = new vis.Network(container, data, options);
  
  // Add click handler for detailed info
  network.on("click", function(params) {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      const cluster = CLUSTERS.find(c => c.id === nodeId);
      if (cluster) {
        showNodeInfo(cluster, params.pointer.DOM);
      }
    }
  });
}

function showNodeInfo(cluster, position) {
  const panel = document.getElementById('info-panel');
  panel.innerHTML = `
    <h4>${cluster.id}</h4>
    <p><strong>Emails:</strong> ${cluster.emails}</p>
    <p><strong>Key Participants:</strong><br/>${cluster.participants.slice(0, 5).join('<br/>')}</p>
    ${cluster.sample_subjects ? `<p><strong>Sample Subjects:</strong><br/>${cluster.sample_subjects.slice(0, 3).join('<br/>')}</p>` : ''}
  `;
  panel.style.display = 'block';
  panel.style.left = Math.min(position.x + 10, window.innerWidth - 320) + 'px';
  panel.style.top = Math.min(position.y + 10, window.innerHeight - 200) + 'px';
  
  setTimeout(() => panel.style.display = 'none', 5000);
}

// Event Listeners
document.getElementById('apply').addEventListener('click', render);
document.querySelectorAll('.preset-bar button').forEach(btn => {
  btn.addEventListener('click', () => {
    applyPreset(btn.getAttribute('data-preset'));
    render();
  });
});

// Initialize
applyPreset('balanced');
render();
</script>

</body>
</html>'''

def main():
    parser = argparse.ArgumentParser(description='Analyze email inbox and create interactive graph')
    parser.add_argument('mbox_path', help='Path to mbox file')
    parser.add_argument('--output', '-o', default='enhanced_inbox_graph.html', help='Output HTML file')
    parser.add_argument('--max-clusters', type=int, default=50, help='Maximum number of clusters')
    parser.add_argument('--min-cluster-size', type=int, default=3, help='Minimum emails per cluster')
    
    args = parser.parse_args()
    
    analyzer = EmailInboxAnalyzer(args.mbox_path)
    analyzer.parse_emails()
    analyzer.create_clusters(min_cluster_size=args.min_cluster_size, max_clusters=args.max_clusters)
    analyzer.generate_html(args.output)

if __name__ == "__main__":
    main()
