# Dynamic Analysis Report Design Summary

This document describes the design and configuration format for the dynamic LArPix deconvolution report viewer.

## 1. System Overview
The dynamic report system consists of two main components:
- **`report_config.json`**: A configuration file that defines datasets and regularization parameters.
- **`report_dynamic.html`**: A Vanilla JavaScript single-page application that loads the JSON and renders analysis plots dynamically.

## 2. Configuration Format (`report_config.json`)

The JSON file follows a structured format to allow easy addition of new runs or datasets.

### Root Object
| Key | Type | Description |
|:---|:---|:---|
| `datasets` | Array of Strings | List of dataset identifiers used in filenames (e.g., `thres5k_nburst256`). |
| `configurations` | Array of Objects | List of analysis runs to display. |

### Configuration Object
| Key | Type | Description |
|:---|:---|:---|
| `run` | Integer | The unique ID of the run (used for anchor links). |
| `tag` | String | A descriptive label for the configuration (e.g., "Standard", "Broad Pixel"). |
| `sigma-temporal` | Float | The temporal regularization width ($\sigma_t$). |
| `sigma-pixel` | Float | The spatial regularization width ($\sigma_{pxl}$). |

**Example Entry:**
```json
{
  "run": 1,
  "tag": "Standard",
  "sigma-temporal": 0.005,
  "sigma-pixel": 0.1
}
```

## 3. Webpage Design (`report_dynamic.html`)

### UI Components
- **Sidebar**: Automatically populated based on the `configurations` array. Provides quick navigation to specific runs.
- **Header Controls**: Contains a dropdown selector for the `datasets`. Changing the selection triggers an immediate re-render of all images.
- **Main Content**: A scrollable area displaying "Configuration Sections". Each section contains a grid of 4 plots.

### Dynamic Image Mapping
The application dynamically constructs image paths using the following convention:
`analysis_20260318_tpc0/v2_{dataset}_sp{sigma_t_str}_spp{sigma_p_str}_{plot_type}.png`

**Sigma String Logic**:
The numeric sigma values are converted to strings by taking the fractional part (e.g., `0.005` becomes `005`, `0.1` becomes `1`). This matches the file-naming convention produced by the analysis pipeline.

### Plot Types Displayed
1. **2D Charge Distribution (Hits)**: `_hist_2d_hits.png`
2. **Deconvoluted Charge Spectrum**: `_hist_deconv_q.png`
3. **Difference Histogram**: `_hist_diff.png`
4. **Signed vs True Charge Scatter**: `_signed_vs_true_scatter.png`

## 4. Usage Requirements
Due to browser security policies (CORS), the `fetch()` API used to load the JSON file requires the report to be served via a web server.
- **Command**: `python3 -m http.server 8000`
- **URL**: `http://localhost:8000/report_dynamic.html`
