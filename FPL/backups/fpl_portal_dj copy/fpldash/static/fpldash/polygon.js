let radarChart = null;
let allPlayers = [];
let selectedMetrics = [];
let thresholdValues = [];
let draggingIndex = null;

async function fetchFplData() {
  const response = await fetch("/api/data");
  allPlayers = await response.json();
  return allPlayers;
}

document.addEventListener("DOMContentLoaded", async () => {
  const drawBtn = document.getElementById("drawPolygonBtn");
  drawBtn.addEventListener("click", handleDrawPolygon);
  await fetchFplData();
});

async function handleDrawPolygon() {
  selectedMetrics = Array.from(
    document.querySelectorAll(".metric-options input:checked")
  ).map((i) => i.value);

  if (selectedMetrics.length < 3) {
    alert("En az 3 metrik seçmelisin!");
    return;
  }
  if (selectedMetrics.length > 6) {
    alert("En fazla 6 metrik seçebilirsin!");
    return;
  }

  const ctx = document.getElementById("radarChart").getContext("2d");

  // Ortalama değerleri al
  const metricMeans = selectedMetrics.map((m) => {
    const validValues = allPlayers.map((p) => parseFloat(p[m])).filter((v) => !isNaN(v));
    const avg = validValues.reduce((a, b) => a + b, 0) / validValues.length;
    return avg || 0;
  });

  thresholdValues = [...metricMeans]; // başlangıçta ortalamalarla başla

  if (radarChart) radarChart.destroy();

  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: selectedMetrics,
      datasets: [
        {
          label: "Thresholds",
          data: thresholdValues,
          fill: true,
          backgroundColor: "rgba(0,123,255,0.2)",
          borderColor: "#007bff",
          pointBackgroundColor: "#007bff",
          pointRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: "Interaktif Radar — Noktaları sürükle, oyuncuları filtrele",
        },
      },
      scales: {
        r: {
          min: 0,
          ticks: { display: true },
          grid: { color: "#ccc" },
          angleLines: { color: "#aaa" },
        },
      },
      events: ["mousemove", "mousedown", "mouseup", "mouseout"],
      onHover: (e, elements) => {
        const canvas = e.chart.canvas;
        canvas.style.cursor = elements.length ? "grab" : "default";
      },
      onClick: () => {},
    },
  });

  // === MOUSE EVENTS ===
  const canvas = radarChart.canvas;

  canvas.addEventListener("mousedown", (event) => {
    const points = radarChart.getElementsAtEventForMode(
      event,
      "nearest",
      { intersect: true },
      true
    );
    if (points.length) {
      draggingIndex = points[0].index;
      canvas.style.cursor = "grabbing";
    }
  });

  canvas.addEventListener("mousemove", (event) => {
    if (draggingIndex === null) return;

    const scale = radarChart.scales.r;
    const y = scale.getValueForPixel(event.offsetY);
    const clamped = Math.max(scale.min, Math.min(scale.max, y));
    thresholdValues[draggingIndex] = clamped;
    radarChart.data.datasets[0].data = thresholdValues;
    radarChart.update("none");

    filterPlayersByThreshold();
  });

  canvas.addEventListener("mouseup", () => {
    draggingIndex = null;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mouseout", () => {
    draggingIndex = null;
  });

  filterPlayersByThreshold(); // ilk yüklemede de filtrele
}

// Filtreleme fonksiyonu
function filterPlayersByThreshold() {
  if (!selectedMetrics.length) return;

  const suggestionsDiv = document.getElementById("suggestion-list");
  suggestionsDiv.innerHTML = "";

  // Oyuncu eşiğin altına düşmeyenleri bul
  const filtered = allPlayers.filter((p) => {
    return selectedMetrics.every((m, i) => {
      const val = parseFloat(p[m]);
      return !isNaN(val) && val >= thresholdValues[i];
    });
  });

  // Çok fazla olmasın diye ilk 10
  const top = filtered
    .sort((a, b) => b["Total Points"] - a["Total Points"])
    .slice(0, 10);

  if (top.length === 0) {
    suggestionsDiv.innerHTML = "<p>No players match your current polygon filter.</p>";
    return;
  }

  top.forEach((p) => {
    const div = document.createElement("div");
    div.classList.add("player-card");
    div.innerHTML = `
      <img src="https://resources.premierleague.com/premierleague/photos/players/250x250/p${p.PlayerPhoto || '99999'}.png"
           onerror="this.src='https://cdn-icons-png.flaticon.com/512/149/149071.png'">
      <div>
        <strong>${p.Player}</strong><br>
        <small>${p.Team} | ${p.Position}</small><br>
        <small style="color:#007bff;">Points: ${p["Total Points"]}</small>
      </div>`;
    suggestionsDiv.appendChild(div);
  });
}
