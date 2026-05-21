import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from scipy.sparse.linalg import eigs

from src.models.astgcn_mh import ASTGCN_MH


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)


def scaled_laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which="LR")[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


def inverse_transform_tensor(tensor, scalers):
    tensor = tensor.clone()
    for i, scaler in enumerate(scalers):
        tensor[:, :, i, :1] = scaler.inverse_transform(tensor[:, :, i, :1])
    return tensor


def load_graph_data_compat(pkl_filename):
    with open(pkl_filename, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = CompatibleUnpickler(f).load()
    adj_mx = adj_mx - np.eye(adj_mx.shape[0])
    return sensor_ids, sensor_id_to_ind, adj_mx


def get_num_nodes(dataset):
    mapping = {
        "Delivery_SH": 30,
        "Delivery_HZ": 31,
        "Delivery_CQ": 30,
        "Delivery_YT": 30,
        "Delivery_JL": 14,
    }
    if dataset not in mapping:
        raise KeyError(f"Unsupported dataset {dataset}")
    return mapping[dataset]


def load_data_bundle(data_path, input_dim, output_dim):
    arrays = {}
    for split in ["train", "val", "test"]:
        split_npz = np.load(os.path.join(data_path, f"{split}.npz"))
        arrays[f"x_{split}"] = split_npz["x"]
        arrays[f"y_{split}"] = split_npz["y"]

    scalers = []
    for i in range(arrays["x_train"].shape[2]):
        scalers.append(
            StandardScaler(
                mean=arrays["x_train"][:, :, i, 0].mean(),
                std=arrays["x_train"][:, :, i, 0].std(),
            )
        )

    processed = {}
    for split in ["train", "val", "test"]:
        x_array = arrays[f"x_{split}"].copy()
        y_array = arrays[f"y_{split}"].copy()
        for i in range(arrays["x_train"].shape[2]):
            x_array[:, :, i, :1] = scalers[i].transform(x_array[:, :, i, :1])
            y_array[:, :, i, :1] = scalers[i].transform(y_array[:, :, i, :1])
        processed[split] = (
            torch.tensor(x_array, dtype=torch.float32)[..., :input_dim],
            torch.tensor(y_array, dtype=torch.float32)[..., :output_dim],
        )

    return {"processed": processed, "scalers": scalers}


def load_region_centroids(raw_csv, city_name):
    df = pd.read_csv(raw_csv)
    df_city = df[df.city == city_name].copy()
    df_city["pickup_time"] = pd.to_datetime(df_city["pickup_time"], format="%m-%d %H:%M:%S")
    df_city = df_city.sort_values("pickup_time")

    region_ids = list(df_city.region_id.unique())
    centroids = df_city.groupby("region_id")[["lat", "lng"]].mean()

    regions = []
    for region_id in region_ids:
        regions.append(
            {
                "region_id": int(region_id),
                "lat": float(centroids.loc[region_id, "lat"]),
                "lng": float(centroids.loc[region_id, "lng"]),
            }
        )
    return regions


def build_model(args, device):
    _, _, adj_mat = load_graph_data_compat(args.graph_pkl)
    num_nodes = adj_mat.shape[0]
    new_adj = adj_mat + np.eye(num_nodes)
    L_tilde = scaled_laplacian(new_adj)
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(device)
        for i in cheb_polynomial(L_tilde, args.K)
    ]

    state_dict = torch.load(args.checkpoint, map_location=device)
    inferred_heads = None
    if "BlockList.0.SAt.head_weights" in state_dict:
        inferred_heads = int(state_dict["BlockList.0.SAt.head_weights"].shape[0])
        if inferred_heads != args.num_heads:
            print(
                f"Checkpoint was trained with num_heads={inferred_heads}. "
                f"Overriding requested num_heads={args.num_heads} for visualization."
            )
            args.num_heads = inferred_heads

    model = ASTGCN_MH(
        nb_block=args.n_blocks,
        K=args.K,
        nb_chev_filter=args.n_hidden,
        nb_time_filter=args.n_hidden,
        time_strides=1,
        cheb_polynomials=cheb_polynomials,
        name="astgcn_mh",
        dataset=args.dataset,
        device=device,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        fusion_type=args.fusion_type,
        num_heads=args.num_heads,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_sample_predictions(args, device):
    data = load_data_bundle(args.data_path, args.input_dim, args.output_dim)
    X_test, y_test = data["processed"]["test"]

    if args.sample_index < 0 or args.sample_index >= X_test.shape[0]:
        raise IndexError(
            f"sample_index={args.sample_index} is outside the test set range 0..{X_test.shape[0] - 1}"
        )

    model = build_model(args, device)

    sample_x = X_test[args.sample_index : args.sample_index + 1].to(device)
    sample_y = y_test[args.sample_index : args.sample_index + 1].clone()

    with torch.no_grad():
        pred = model(sample_x).cpu()

    pred = inverse_transform_tensor(pred, data["scalers"]).squeeze(0).squeeze(-1).numpy()
    actual = inverse_transform_tensor(sample_y, data["scalers"]).squeeze(0).squeeze(-1).numpy()
    history = inverse_transform_tensor(X_test[args.sample_index : args.sample_index + 1].clone(), data["scalers"])
    history = history.squeeze(0).squeeze(-1).numpy()

    test_npz = np.load(os.path.join(args.data_path, "test.npz"))
    if "y_hour" in test_npz:
        hour_labels = [f"+{i + 1}h (hour {int(h)})" for i, h in enumerate(test_npz["y_hour"][args.sample_index][: args.horizon])]
    else:
        hour_labels = [f"+{i + 1}h" for i in range(args.horizon)]

    return {
        "pred": pred[:, : args.num_nodes],
        "actual": actual[:, : args.num_nodes],
        "history": history[:, : args.num_nodes],
        "hour_labels": hour_labels,
        "sample_count": X_test.shape[0],
    }


def build_region_payload(regions, pred, actual, history):
    payload = []
    num_regions = min(len(regions), pred.shape[1], actual.shape[1], history.shape[1])
    for idx in range(num_regions):
        payload.append(
            {
                "region_id": regions[idx]["region_id"],
                "lat": regions[idx]["lat"],
                "lng": regions[idx]["lng"],
                "pred": np.round(pred[:, idx], 2).tolist(),
                "actual": np.round(actual[:, idx], 2).tolist(),
                "history_last": float(round(history[-1, idx], 2)),
            }
        )
    return payload


def render_html(output_path, payload, hour_labels, meta):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    max_value = max(
        max(max(region["pred"]) for region in payload),
        max(max(region["actual"]) for region in payload),
        1.0,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shanghai Demand Map - ASTGCN_MH</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f7fb;
      color: #172033;
    }}
    .page {{
      padding: 20px;
    }}
    .header {{
      margin-bottom: 16px;
    }}
    .title {{
      font-size: 28px;
      font-weight: 700;
      margin: 0 0 6px 0;
    }}
    .subtitle {{
      margin: 0;
      color: #4d5b76;
      line-height: 1.5;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr auto;
      align-items: center;
      background: white;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 2px 14px rgba(16, 24, 40, 0.08);
      margin-bottom: 16px;
    }}
    .control-label {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .pill {{
      background: #eef2ff;
      color: #334155;
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
    }}
    .maps {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .map-card {{
      background: white;
      border-radius: 10px;
      box-shadow: 0 2px 14px rgba(16, 24, 40, 0.08);
      overflow: hidden;
    }}
    .map-title {{
      padding: 14px 16px 0 16px;
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .map-caption {{
      padding: 4px 16px 12px 16px;
      color: #5a657a;
      font-size: 13px;
    }}
    .map {{
      height: 580px;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      padding: 12px 16px 16px 16px;
      color: #5a657a;
      font-size: 13px;
      flex-wrap: wrap;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    @media (max-width: 960px) {{
      .maps {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
      .meta {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1 class="title">Shanghai Demand Forecast Explorer</h1>
      <p class="subtitle">
        Interactive region-level demand view from <code>astgcn_mh</code>. Move the forecast slider to compare
        actual and predicted package demand across Shanghai regions.
      </p>
    </div>

    <div class="controls">
      <div>
        <div class="control-label">Forecast Hour</div>
        <input id="hourSlider" type="range" min="0" max="{len(hour_labels) - 1}" value="0" step="1" />
        <div id="hourLabel" class="subtitle" style="margin-top: 8px;"></div>
      </div>
      <div class="meta">
        <div class="pill">Sample {meta["sample_index"]} / {meta["sample_count"] - 1}</div>
        <div class="pill">{meta["dataset"]}</div>
        <div class="pill">Heads: {meta["num_heads"]}</div>
        <div class="pill">Fusion: {meta["fusion_type"]}</div>
      </div>
    </div>

    <div class="maps">
      <div class="map-card">
        <h2 class="map-title">Actual Demand</h2>
        <div class="map-caption">Blue circles show the observed package count by region.</div>
        <div id="actualMap" class="map"></div>
        <div class="legend">
          <span>Circle radius grows with demand.</span>
          <span>Popup shows actual, predicted, and error for that region.</span>
        </div>
      </div>
      <div class="map-card">
        <h2 class="map-title">Predicted Demand</h2>
        <div class="map-caption">Red circles show the ASTGCN_MH forecast for the same hour.</div>
        <div id="predMap" class="map"></div>
        <div class="legend">
          <span>Last input demand is included in each popup for quick context.</span>
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const regionData = {json.dumps(payload)};
    const hourLabels = {json.dumps(hour_labels)};
    const maxValue = {float(max_value)};

    const centerLat = regionData.reduce((acc, row) => acc + row.lat, 0) / regionData.length;
    const centerLng = regionData.reduce((acc, row) => acc + row.lng, 0) / regionData.length;

    const actualMap = L.map('actualMap').setView([centerLat, centerLng], 10);
    const predMap = L.map('predMap').setView([centerLat, centerLng], 10);

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(actualMap);

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(predMap);

    const actualLayer = L.layerGroup().addTo(actualMap);
    const predLayer = L.layerGroup().addTo(predMap);

    function radiusFor(value) {{
      const safe = Math.max(0, value);
      return 6 + 18 * Math.sqrt(safe / Math.max(maxValue, 1));
    }}

    function blueFor(value) {{
      const ratio = Math.min(Math.max(value / Math.max(maxValue, 1), 0), 1);
      const shade = Math.round(230 - ratio * 120);
      return `rgb(${{40}}, ${{110}}, ${{shade}})`;
    }}

    function redFor(value) {{
      const ratio = Math.min(Math.max(value / Math.max(maxValue, 1), 0), 1);
      const shade = Math.round(230 - ratio * 120);
      return `rgb(${{shade}}, ${{75}}, ${{75}})`;
    }}

    function popupHtml(region, hourIndex, viewType) {{
      const actual = region.actual[hourIndex];
      const pred = region.pred[hourIndex];
      const err = +(pred - actual).toFixed(2);
      if (viewType === 'actual') {{
        return `
          <div style="min-width: 190px;">
            <div style="font-weight: 700; margin-bottom: 6px;">Region ${{region.region_id}}</div>
            <div>Forecast: ${{hourLabels[hourIndex]}}</div>
            <div>Actual demand: <b>${{actual}}</b></div>
          </div>
        `;
      }}
      return `
        <div style="min-width: 190px;">
          <div style="font-weight: 700; margin-bottom: 6px;">Region ${{region.region_id}}</div>
          <div>Forecast: ${{hourLabels[hourIndex]}}</div>
          <div>Actual demand: <b>${{actual}}</b></div>
          <div>Predicted demand: <b>${{pred}}</b></div>
          <div>Error: <b>${{err}}</b></div>
          <div>Last input demand: <b>${{region.history_last}}</b></div>
        </div>
      `;
    }}

    function render(hourIndex) {{
      actualLayer.clearLayers();
      predLayer.clearLayers();

      regionData.forEach(region => {{
        const actualValue = region.actual[hourIndex];
        const predValue = region.pred[hourIndex];

        L.circleMarker([region.lat, region.lng], {{
          radius: radiusFor(actualValue),
          color: blueFor(actualValue),
          fillColor: blueFor(actualValue),
          fillOpacity: 0.45,
          weight: 1.5
        }})
          .bindPopup(popupHtml(region, hourIndex, 'actual'))
          .addTo(actualLayer);

        L.circleMarker([region.lat, region.lng], {{
          radius: radiusFor(predValue),
          color: redFor(predValue),
          fillColor: redFor(predValue),
          fillOpacity: 0.45,
          weight: 1.5
        }})
          .bindPopup(popupHtml(region, hourIndex, 'predicted'))
          .addTo(predLayer);
      }});

      document.getElementById('hourLabel').textContent = `Showing ${{hourLabels[hourIndex]}}`;
    }}

    const slider = document.getElementById('hourSlider');
    slider.addEventListener('input', (event) => {{
      render(Number(event.target.value));
    }});

    render(0);
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Build an interactive Shanghai demand map from ASTGCN_MH predictions.")
    parser.add_argument("--dataset", type=str, default="Delivery_SH")
    parser.add_argument("--city-name", type=str, default="Shanghai")
    parser.add_argument("--raw-csv", type=str, default="./data/pickup_sh.csv")
    parser.add_argument("--data-path", type=str, default="./data/Delivery_SH")
    parser.add_argument("--graph-pkl", type=str, default="./data/sensor_graph/adj_mx_delivery_sh.pkl")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./logs/Delivery_SH/astgcn_mh/2-32-3-1.0-16-0.0001/final_model_0.pt",
    )
    parser.add_argument("--output-html", type=str, default="./results/Delivery_SH/visualizations/astgcn_mh_shanghai_sample0.html")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=32)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--fusion-type", type=str, default="weighted_sum")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.num_nodes = get_num_nodes(args.dataset)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    regions = load_region_centroids(args.raw_csv, args.city_name)
    outputs = collect_sample_predictions(args, device)
    payload = build_region_payload(regions, outputs["pred"], outputs["actual"], outputs["history"])

    render_html(
        args.output_html,
        payload,
        outputs["hour_labels"],
        {
            "dataset": args.dataset,
            "num_heads": args.num_heads,
            "fusion_type": args.fusion_type,
            "sample_index": args.sample_index,
            "sample_count": outputs["sample_count"],
        },
    )
    print(f"Interactive map written to {args.output_html}")


if __name__ == "__main__":
    main()
