# Real-World Multi-Aspect Review Benchmark Sample

### Review Subject: Gaming Laptop Multi-Aspect Evaluation

| Target Aspect | Human Ground Truth | Generic Foundation (MAMS) | Tech Specialist (Laptop14) | Home Specialist | Apparel Specialist | ProductLens MoE (Routed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaming** | **POSITIVE** | ⚪ **neutral** (95.0%) | ⚪ **neutral** (55.2%) | 🟢 **positive** (93.9%) | ⚪ **neutral** (54.5%) | ⚪ **neutral** (55.2%) |
| **Display** | **POSITIVE** | 🟢 **positive** (93.3%) | 🟢 **positive** (97.1%) | 🟢 **positive** (96.4%) | 🟢 **positive** (77.6%) | 🟢 **positive** (97.1%) |
| **screen bleeding** | **NEGATIVE** | 🔴 **negative** (88.5%) | 🔴 **negative** (95.4%) | 🔴 **negative** (98.5%) | 🔴 **negative** (91.7%) | 🔴 **negative** (95.4%) |
| **dead pixel** | **NEGATIVE** | 🔴 **negative** (95.3%) | 🔴 **negative** (96.0%) | 🔴 **negative** (98.6%) | 🔴 **negative** (92.8%) | 🔴 **negative** (96.0%) |
| **speaker** | **NEUTRAL** | ⚪ **neutral** (76.0%) | ⚪ **neutral** (74.7%) | ⚪ **neutral** (84.8%) | 🔴 **negative** (75.6%) | ⚪ **neutral** (74.7%) |
| **Heating** | **NEUTRAL** | 🔴 **negative** (53.1%) | ⚪ **neutral** (53.3%) | ⚪ **neutral** (78.9%) | 🔴 **negative** (80.6%) | ⚪ **neutral** (53.3%) |
| **keyboard** | **NEGATIVE** | 🔴 **negative** (61.3%) | 🔴 **negative** (90.8%) | 🔴 **negative** (98.5%) | 🔴 **negative** (89.7%) | 🔴 **negative** (90.8%) |
| **touch pad** | **NEGATIVE** | 🔴 **negative** (78.0%) | 🔴 **negative** (95.9%) | 🔴 **negative** (98.8%) | 🔴 **negative** (92.1%) | 🔴 **negative** (95.9%) |
| **battery** | **NEGATIVE** | ⚪ **neutral** (51.0%) | ⚪ **neutral** (51.8%) | ⚪ **neutral** (48.1%) | 🔴 **negative** (44.9%) | ⚪ **neutral** (51.8%) |
| **price** | **POSITIVE** | ⚪ **neutral** (91.4%) | 🟢 **positive** (80.8%) | 🟢 **positive** (79.1%) | ⚪ **neutral** (61.0%) | 🟢 **positive** (80.8%) |

### Auto-Extracted Aspects by ProductLensEngine:

| Extracted Aspect Span | Predicted Sentiment | Confidence | Routed Specialist |
| :--- | :---: | :---: | :--- |
| `graphics` | ⚪ **neutral** | 94.1% | `tech` (metadata_match:tech) |
| `Display` | 🟢 **positive** | 97.1% | `tech` (metadata_match:tech) |
| `screen` | 🔴 **negative** | 94.0% | `tech` (metadata_match:tech) |
| `Heat` | 🔴 **negative** | 63.1% | `tech` (metadata_match:tech) |
| `graphics` | ⚪ **neutral** | 90.6% | `tech` (metadata_match:tech) |
| `fan` | ⚪ **neutral** | 49.2% | `tech` (metadata_match:tech) |
| `speed` | 🔴 **negative** | 64.1% | `tech` (metadata_match:tech) |
| `keyboard` | 🔴 **negative** | 90.8% | `tech` (metadata_match:tech) |
| `keyboard` | 🔴 **negative** | 97.5% | `tech` (metadata_match:tech) |
| `battery` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |
| `battery life` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |
| `screen` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |
| `screen` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |
| `screen` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |
| `screen` | ⚪ **neutral** | 51.8% | `tech` (metadata_match:tech) |

**Aggregated Summary Sentiment**: **MIXED**