// index.html → script.js
document.getElementById('citizen-btn').onclick = () =>
  document.getElementById('login-modal').style.display = 'block';

document.getElementById('close-modal').onclick = () =>
  document.getElementById('login-modal').style.display = 'none';

document.getElementById('login-form').onsubmit = e => {
  e.preventDefault();
  // demo only
  window.location.href = 'feedback.html';
};

document.getElementById('police-btn').onclick = () =>
  window.location.href = 'restricted.html';

// 2) Seasonal Chart (bar per month)
const ctx2 = document.getElementById('seasonalChart').getContext('2d');
fetch('data/seasonal_counts.json')          // you need this file: {"Jan":10,"Feb":12,...}
  .then(r => r.json())
  .then(season => {
    new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: Object.keys(season),
        datasets: [{
          label: 'Avg Monthly Burglaries',
          data: Object.values(season),
          backgroundColor: 'rgba(2,136,209,0.6)',
        }]
      },
      options: { responsive: true }
    });
  });

// 4) Last-6-Months Heatmap
const recentMap = L.map('recentMap').setView([51.5074, -0.1278], 10);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(recentMap);

fetch('data/ward_6months.json')   // you need to prepare this GeoJSON
  .then(r => r.json())
  .then(geojson => {
    L.geoJSON(geojson, {
      style: f => ({
        fillColor: getColor(f.properties.count),
        weight: 1, color: '#fff', fillOpacity: 0.7
      })
    }).addTo(recentMap);
  });
