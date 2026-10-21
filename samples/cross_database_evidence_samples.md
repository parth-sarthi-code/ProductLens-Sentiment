# End-to-End Multi-Database Evaluation: Self-Extracted Evidence & MoE Analysis

This document demonstrates the full ProductLens ABSA pipeline evaluated on real-world samples from each benchmark database.
**Key Feature**: The `evidence_snippet` is now **self-extracted by the engine** alongside each aspect mention.

## Database: Tech Hardware (Laptop14)
> **Review Text**: *"Boot time is super fast , around anywhere from 35 seconds to 1 minute . The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."*
> **Input Category Tag**: `Electronics` | **Summary Sentiment**: **`MIXED`** | **Latency**: `1084.34ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Routed Specialist Expert |
| :--- | :--- | :---: | :---: | :--- |
| **`Boot time`** | *"Boot time is super fast , around anywhere from 35 seconds to 1 minute ."* | 🟢 **POSITIVE** | 99.8% | `tech` (metadata_match:tech) |
| **`keyboard`** | *"The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."* | 🟢 **POSITIVE** | 99.5% | `tech` (metadata_match:tech) |
| **`touchpad`** | *"The keyboard is backlit and keys have good travel , but the touchpad feels sticky ."* | 🔴 **NEGATIVE** | 99.1% | `tech` (metadata_match:tech) |

---

## Database: Home Electronics (Television)
> **Review Text**: *"very good, cast screen working properly, best sound and picture quality. Remote sensor has slight lag though."*
> **Input Category Tag**: `Home_and_Kitchen` | **Summary Sentiment**: **`MIXED`** | **Latency**: `1311.58ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Routed Specialist Expert |
| :--- | :--- | :---: | :---: | :--- |
| **`cast screen`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 99.8% | `home` (metadata_match:home) |
| **`sound`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 99.7% | `home` (metadata_match:home) |
| **`picture quality`** | *"very good, cast screen working properly, best sound and picture quality."* | 🟢 **POSITIVE** | 99.8% | `home` (metadata_match:home) |
| **`Remote sensor`** | *"Remote sensor has slight lag though."* | 🔴 **NEGATIVE** | 99.7% | `home` (metadata_match:home) |

---

## Database: Apparel & Fashion (T-Shirt)
> **Review Text**: *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."*
> **Input Category Tag**: `Clothing_Shoes_and_Jewelry` | **Summary Sentiment**: **`MIXED`** | **Latency**: `885.87ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Routed Specialist Expert |
| :--- | :--- | :---: | :---: | :--- |
| **`cloth`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🟢 **POSITIVE** | 96.3% | `apparel` (metadata_match:apparel) |
| **`colored dye`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🔴 **NEGATIVE** | 98.1% | `apparel` (metadata_match:apparel) |
| **`collar`** | *"cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out."* | 🔴 **NEGATIVE** | 97.7% | `apparel` (metadata_match:apparel) |

---

## Database: MAMS Challenge (Universal Multi-Aspect)
> **Review Text**: *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."*
> **Input Category Tag**: `None (Universal Fallback)` | **Summary Sentiment**: **`MIXED`** | **Latency**: `682.76ms`

| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Routed Specialist Expert |
| :--- | :--- | :---: | :---: | :--- |
| **`food`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🟢 **POSITIVE** | 97.7% | `fallback` (mams_foundation_fallback) |
| **`appetizers`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🟢 **POSITIVE** | 99.0% | `fallback` (mams_foundation_fallback) |
| **`wait staff`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🔴 **NEGATIVE** | 99.7% | `fallback` (mams_foundation_fallback) |
| **`noise level`** | *"The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable ."* | 🔴 **NEGATIVE** | 99.7% | `fallback` (mams_foundation_fallback) |

---
