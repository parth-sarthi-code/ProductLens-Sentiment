# End-to-End Multi-Database Evaluation: Self-Extracted Evidence & MoE Analysis

This document demonstrates the full ProductLens ABSA pipeline evaluated on real-world samples from each benchmark database.
**Key Feature**: The `evidence_snippet` is now **self-extracted by the engine** alongside each aspect mention.

## Database: Tech Hardware (Laptop14)
> **Review Text**: *"Boot time is super fast , around anywhere from 35 seconds to 1 minute . The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."*
> **Input Category Tag**: `Electronics` | **Summary Sentiment**: **`POSITIVE`** | **Latency**: `202.92ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Active Backbone |
| :--- | :--- | :---: | :---: | :--- |
| **`Boot time`** | *"Boot time is super fast , around anywhere from 35 seconds to 1 minute ."* | 🟢 **POSITIVE** | 98.5% | `unified` (unified_calibrated_backbone) |
| **`keyboard`** | *"The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."* | 🟢 **POSITIVE** | 98.7% | `unified` (unified_calibrated_backbone) |
| **`touchpad`** | *"The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."* | 🔴 **NEGATIVE** | 98.4% | `unified` (unified_calibrated_backbone) |

---

## Database: Home Electronics (Television)
> **Review Text**: *"very good, cast screen working properly, best sound and picture quality. Remote sensor has slight lag though."*
> **Input Category Tag**: `Home_and_Kitchen` | **Summary Sentiment**: **`POSITIVE`** | **Latency**: `51.32ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Active Backbone |
| :--- | :--- | :---: | :---: | :--- |
| **`cast screen`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 98.8% | `unified` (unified_calibrated_backbone) |
| **`sound`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 98.7% | `unified` (unified_calibrated_backbone) |
| **`picture quality`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 98.9% | `unified` (unified_calibrated_backbone) |
| **`Remote sensor`** | *"Remote sensor has slight lag though."* | 🔴 **NEGATIVE** | 97.8% | `unified` (unified_calibrated_backbone) |

---

## Database: Apparel & Fashion (T-Shirt)
> **Review Text**: *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."*
> **Input Category Tag**: `Clothing_Shoes_and_Jewelry` | **Summary Sentiment**: **`NEGATIVE`** | **Latency**: `19.94ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Active Backbone |
| :--- | :--- | :---: | :---: | :--- |
| **`cloth`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🟢 **POSITIVE** | 98.3% | `unified` (unified_calibrated_backbone) |
| **`colored dye`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🔴 **NEGATIVE** | 97.8% | `unified` (unified_calibrated_backbone) |
| **`collar`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🔴 **NEGATIVE** | 98.5% | `unified` (unified_calibrated_backbone) |

---

## Database: MAMS Challenge (Universal Multi-Aspect)
> **Review Text**: *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."*
> **Input Category Tag**: `None (Universal Fallback)` | **Summary Sentiment**: **`MIXED`** | **Latency**: `24.75ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Active Backbone |
| :--- | :--- | :---: | :---: | :--- |
| **`food`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🟢 **POSITIVE** | 99.0% | `unified` (unified_calibrated_backbone) |
| **`appetizers`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🟢 **POSITIVE** | 99.1% | `unified` (unified_calibrated_backbone) |
| **`wait staff`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🔴 **NEGATIVE** | 98.8% | `unified` (unified_calibrated_backbone) |
| **`noise level`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🔴 **NEGATIVE** | 98.5% | `unified` (unified_calibrated_backbone) |

---
