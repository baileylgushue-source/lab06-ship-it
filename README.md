---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---


# QuickBite Late Order Check
## What it does
This page helps a QuickBite dispatcher decide, when an order is placed, whether to offer the customer a $4 voucher. It checks the order details, scores them with our Lab 05 model, and shows the chance that the order arrives late or on time and the action to take. The voucher is offered when the chance of a late order is at or above the threshold 0.65.  
Space: https://huggingface.co/spaces/gurleen-s/lab06-ship-it


## How to run it locally
```bash
conda env create -f environment.yml
conda activate lab06-ship-it
gradio app.py
```

`environment.yml` installs `requirements-dev.txt`, which includes `requirements.txt`, so no other install step is needed. Open the local URL that Gradio prints.

## Decisions

| Field | Allowed Range | Why |
|---|---|---|
| distance_km | 0.1 to 30 | Blocks impossible or unrealistic distances. The model was trained on 0.3 to 15 km, so values from 15 to 30 are accepted but less reliable. |
| items | Whole number, 1 to 20 | An order has at least one item, and items are counted whole. The model was trained on 1 to 8, so larger orders are accepted but less reliable. |
| hour | Whole number, 10 to 22 | The model only saw orders placed from 10:00 to 22:00, so it cannot judge other hours. |
| prep_minutes | 1 to 60, or empty | The food prep time is sometimes not known yet. An empty value is accepted with a warning, because the pipeline fills it with the training median. |
| courier_months | 0 to 120 | A new courier is 0, and 120 months (10 years) covers a long career. The model was trained on 0 to 59, so longer experience is accepted but less reliable. |
| weather | clear, rain, snow | Only the three values the model was trained on. Anything else is refused (see below). |
| day_type | weekday, weekend | Only the two values the model was trained on. |
| vehicle | bike, scooter, car | Only the three values the model was trained on. |

Every other missing or wrong value is refused with a message that names the field and the rule.    

The page does not ask for `customer_rating` or `order_id`. The rating is given after delivery and is partly caused by lateness, so it would leak the answer (in Lab 05, adding it back raised CV ROC-AUC from 0.806 to 0.916). The order ID is only a label.  

**snow versus Snow.** We scored the same order with `predict_order` and changed only the weather. With `snow` the probability of a late order was 0.752. With `Snow` our check raises `weather: choose one of clear, rain, snow`. When we skipped the check and called the pipeline directly, `Snow` scored 0.662. The one-hot encoder ignores the unknown value, so the model treats the order as if it had no weather at all, which is wrong without any error. The page only offers the allowed choices, and `predict.py` checks the value again before scoring, so this cannot happen on the page.

## Result
Our Lab 05 model is a LogisticRegression with `C=0.1` and `class_weight="balanced"`. The final pipeline was fitted on all 2,250 training orders and scored once on the 750 test orders at threshold 0.65.

| | Predicted on time | Predicted late |
|---|---|---|
| **Actually on time** | 570 | 63 |
| **Actually late** | 62 | 55 |

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.466 | 0.470 | 0.468 | 0.766 |

| Policy | Total cost on the test set |
|---|---|
| Flag nobody (current policy) | $1,755 |
| Flag everybody | $3,000 |
| **Our model** | **$1,402** |

The model saves $353 (about 20%) compared with flagging nobody. This comes from one test set of 750 invented orders, so the exact saving would change with new data.

## Tests
- `tests/test_predict.py`: probabilities are between 0 and 1, a harder order scores higher than an easier one, the voucher decision at, just below and just above the threshold, the threshold comes from `model_info.json`, an empty `prep_minutes` is accepted, impossible numbers and unseen categories are refused with the field named, and the model file is loaded only once.
- `tests/test_deployment.py`: the installed scikit-learn version matches `model_info.json` and the pin in `requirements.txt`, gradio is not in `requirements.txt`, the README `sdk_version` matches the installed Gradio, and the `app_file` in the README exists.

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations
- It only sees the eight fields on the page. It cannot see road closures, traffic, food delays after the order, or courier order assignment delays.
- Use it to decide whether to offer a voucher, not to judge a courier or a restaurant.
- **Values outside the training range.** The page accepts some values the model never saw (for example 15 to 30 km, or more than 8 items). There the model only extends the trend it learned, so the probability can look certain (a 30 km order scores 1.00) without any data behind it.
- **Invented data, small test set.** The data is generated by `data.py`, and the test set holds only 117 late orders, so the size of the saving is uncertain.
- **`prep_minutes` at prediction time.** The model learned from how long the restaurant actually took, but that is only known after the food is ready; when an order is placed only an estimate exists. In real use we would also need to predict the restaurant's prep time from past orders, and the model would likely do a little worse with that estimate.
- **Probabilities are not real chances.** `class_weight="balanced"` pushes predicted probabilities up, which is why the threshold is 0.65. With more time we would calibrate them so that a user could be shown a real chance of lateness. The chances shown on the page are these raw probabilities, so a dispatcher should read them as a ranking against the threshold, not as an exact chance of lateness.
- **`hour` is treated as a number.** Lateness rises at the lunch and dinner rush and falls again (as is common for restaurants), which a single linear slope on `hour` cannot represent. Treating it as a category, or adding a rush-hour flag, might help.
- **Fixed cost rule.** The rule assumes a voucher always works and every missed late order costs exactly $15. If either changes, the threshold must be chosen again.