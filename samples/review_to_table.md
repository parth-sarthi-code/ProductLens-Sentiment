# Real-World Review Aspect-Based Sentiment Analysis

---

## 1. Original Customer Review

> **Review Title:** Good laptop at this price range
>
> 1. **Gaming (8/10)**: I played Genshin (max graphics 60fps), Elden ring (max + rtx off 60fps), Dying light 1 (max+rtx on 70+fps), Witcher 3 (max+rtx low 70-75 fps, max + rtx high 50-60 fps), Valorant (max 144-200fps), Overwatch 2 (max 70+fps)
> 2. **Display**: Display is very great and video quality at peak beacuse of 2k res , but mine has more screen bleeding and i got one dead pixel (my fault because I didn't notice before).
> 3. **Speaker (7/10)**: Average speaker
> 4. **Heating (7/10)**: Heat management is average. When I play games at high graphics it goes upto 90-95° sometimes at max fan speed. (Not a big deal, heating in normal in all gaming laptops)
> 5. **Keyboard and touch pad (4/10)**: Tbh worst keyboard and touch pad at this price and low quality. Now some keys are already started to not work.
> 6. **Battery (4/10)**: Worst battery life not even 90 minutes of playing or streaming videos. (Not a big problem because gaming laptops need active charging)
>
> **Conclusion:** good laptop at this price (8/10)
>
> **Buyer Advice:** IF YOU ARE GOING TO BUY Then don't forget to check if it has more screen bleeding. (All LCD displays have screen bleeding and it's common but buy which has the slightest screen bleeding). Mine has more screen bleeding in all four sides but only noticeable in dark room.

---

## 2. Review Aspect-to-Sentiment Table

| Aspect Term | Sub-Domain | Review Evidence / Context | Human Truth | ProductLens MoE Prediction | Generic Model (MAMS) | Winner & Behavioral Analysis |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`Display`** | Visuals | *"Display is very great and video quality at peak beacuse of 2k res"* | 🟢 **POSITIVE** | 🟢 **Positive** (97.1%) | 🟢 **Positive** (93.3%) | **Tie**: Both models captured strong visual praise. |
| **`screen bleeding`** | Defect | *"mine has more screen bleeding in all four sides... only noticeable in dark room"* | 🔴 **NEGATIVE** | 🔴 **Negative** (95.4%) | 🔴 **Negative** (88.5%) | **MoE (+6.9% conf)**: Tech specialist accurately penalizes panel defect terminology. |
| **`dead pixel`** | Defect | *"i got one dead pixel (my fault because I didn't notice before)"* | 🔴 **NEGATIVE** | 🔴 **Negative** (96.0%) | 🔴 **Negative** (95.3%) | **MoE (+0.7% conf)**: Both capture hardware defect polarity despite self-blaming clause. |
| **`price`** | Economics | *"Good laptop at this price range... Conclusion : good laptop at this price (8/10)"* | 🟢 **POSITIVE** | 🟢 **Positive** (80.8%) | ⚪ **Neutral** (91.4%) ⚠️ | **MoE Win (Correct Polarity)**: Generic model missed the value-for-money endorsement completely. |
| **`keyboard`** | Peripherals | *"Tbh worst keyboard and touch pad at this price and low quality. Now some keys are already started to not work."* | 🔴 **NEGATIVE** | 🔴 **Negative** (90.8%) | 🔴 **Negative** (61.3%) ⚠️ | **MoE (+29.5% conf)**: Tech specialist firmly recognizes hardware failure (*"keys started to not work"*). |
| **`touch pad`** | Peripherals | *"worst keyboard and touch pad at this price and low quality."* | 🔴 **NEGATIVE** | 🔴 **Negative** (95.9%) | 🔴 **Negative** (78.0%) ⚠️ | **MoE (+17.9% conf)**: Tech specialist decisively isolates input device degradation. |
| **`Heating`** | Thermals | *"Heat management is average... (Not a big deal, heating in normal in all gaming laptops)"* | ⚪ **NEUTRAL** | ⚪ **Neutral** (53.3%) | 🔴 **Negative** (53.1%) ⚠️ | **MoE Win (Nuance Handling)**: Generic model saw *"heating"* and panicked; MoE factored in the mitigating clause. |
| **`speaker`** | Audio | *"3.speaker (7/10) Average speaker"* | ⚪ **NEUTRAL** | ⚪ **Neutral** (74.7%) | ⚪ **Neutral** (76.0%) | **Tie**: Both correctly categorized the 7/10 rating as neutral/average. |
| **`battery`** | Battery | *"Worst battery life not even 90 minutes... (Not a big problem because gaming laptops need active charging)"* | ⚪ **NEUTRAL / MIXED** | ⚪ **Neutral** (51.8%) | ⚪ **Neutral** (51.0%) | **Tie**: Mitigated complaint correctly resolved as balanced/neutral. |
| **`Gaming`** | Performance | *"1.Gaming (8/10) I played Genshin (max 60fps), Elden ring (60fps), Dying light 1 (70+fps)..."* | ⚪ **NEUTRAL / POSITIVE** | ⚪ **Neutral** (55.2%) | ⚪ **Neutral** (95.0%) | **MoE (Higher positive leaning)**: MoE gave 44% positive probability vs generic's 4.8%. |

---

## 3. Review Verdict Synthesis

* **Total Aspects Evaluated**: 10 Target Dimensions
* **Aggregated Sentiment**: **`MIXED`**
  * **Positive Aspects (2)**: `Display` (2K resolution, video quality), `Price` (8/10 value proposition)
  * **Negative Aspects (4)**: `Screen Bleeding` (dark room defect), `Dead Pixel`, `Keyboard` (failing keys), `Touchpad` (low quality)
  * **Neutral / Average Aspects (4)**: `Heating` (normal for gaming), `Speakers` (average), `Battery` (short but expected), `Gaming` (solid baseline performance)
* **Execution Latency**: **1.24s** total across all 10 aspects on GPU.
