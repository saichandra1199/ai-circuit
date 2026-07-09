# Kaggle Setup Guide — AI Circuit

---

## Step 1 — Create Kaggle Account
1. Go to [kaggle.com](https://kaggle.com) → Sign Up
2. Verify email
3. Go to **Account Settings** → **Phone Verification** (required to use GPU)

---

## Step 2 — Upload the Notebook
1. Click **Create** → **New Notebook** (top-left on Kaggle)
2. In the new notebook, click **File** → **Import Notebook**
3. Upload `kaggle_setup.ipynb` from your local machine
4. Notebook opens in editor

---

## Step 3 — Add H&M Dataset
1. In the notebook editor, look for **right sidebar** → click **+ Add Data**
2. Search: `H&M Personalized Fashion Recommendations`
3. Click the dataset → **Add** button
4. Confirm it appears under **Input** in the sidebar as `/kaggle/input/h-and-m-personalized-fashion-recommendations/`

---

## Step 4 — Add OpenAI API Key as Secret
1. Left sidebar → **Secrets** (lock icon) OR go to [kaggle.com/settings](https://kaggle.com/settings) → **API** section → **Secrets**
2. Click **+ Add a new secret**
3. Name: `OPENAI_API_KEY` (exact spelling — notebook uses this name)
4. Value: your OpenAI key (`sk-...`)
5. Toggle **Notebook access** ON for this secret

---

## Step 5 — Enable GPU
1. In notebook editor, right sidebar → **Notebook options** (pencil/settings icon) OR bottom bar
2. Under **Accelerator** → select **GPU T4 x2** (or **P100** — both free)
3. Kaggle gives ~30 GPU hours/week free
4. Save settings

---

## Step 6 — Run the Notebook

**Choose your workflow before running:**

### Workflow A — Human + Agent
You control which classes to use; agent handles training loop.

1. Run **Cell 1** — clones repo and installs dependencies (~2 min)
2. Run **Cell 2** — loads OpenAI API key from Kaggle Secrets
3. Run **Cell 3** — verifies GPU is active
4. Run **Cell A1** — prepares H&M dataset (~5 min)
   - Edit `max_per_class=500` for a fast test run
   - Remove the limit (`max_per_class=None`) for full dataset
5. Run **Cell A2** — agent trains, evaluates, and iterates
   - Edit `max_iterations` to control how many train cycles to run
   - Edit `experiment_name` to label your run

### Workflow B — Full Agent (Autonomous)
LLM picks classes, preps data, trains — no manual input needed.

1. Run **Cell 1** — clones repo and installs dependencies
2. Run **Cell 2** — loads OpenAI API key
3. Run **Cell 3** — verifies GPU
4. Run **Cell B1** — agent does everything end-to-end
   - Edit `max_train_per_class` to control dataset size (500 = fast, None = full)
   - Edit `max_iterations` to control training cycles
   - Edit `experiment_name` to label your run

**To run a cell:** Click it → press **Shift + Enter** OR click the ▷ button on the left of the cell

---

## Step 7 — Monitor Training
- Output appears below each cell in real time
- Training logs show epoch progress and F1 scores after each run
- Expect each agent iteration to take **5–15 min** with T4 GPU
- Agent stops early if `target_f1` is reached
- Final cell reads and prints the experiment log

---

## Step 8 — Save and Download Results
- All results saved under `/kaggle/working/ai_circuit/experiments/`
- Each session creates a timestamped folder with:
  - `experiment_log.json` — summary of all runs (F1, accuracy, config changes)
  - `run_N/metrics.json` — detailed metrics per run
  - `run_N/best_model.pth` — saved model weights
  - `run_N/notes.md` — LLM analysis of that run
- To download files: right sidebar → **Output** tab → select files → download
- To save a snapshot: click **Save Version** (top-right) → saves notebook + all outputs

---

## Common Issues

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY not found` | Check secret name is exactly `OPENAI_API_KEY`; toggle notebook access ON |
| `No module named kaggle_secrets` | Cell 2 only works on Kaggle — not for local runs |
| `CUDA not available` | Accelerator not set — go to Notebook Options → set GPU T4 |
| `Dataset not found` | H&M dataset not added to notebook — redo Step 3 |
| Slow training | GPU not active, or `max_per_class` too high — reduce to 200–500 |
| Git clone fails | Repo URL may need VPN/access — see tip below |

---

## Tip — Push Repo to GitHub First

The notebook clones from EPAM GitLab (`git.garage.epam.com`). If that URL is
private or requires VPN, Kaggle cannot access it. Push to a public GitHub repo
instead:

```bash
# on your local machine
git remote add github https://github.com/<your-username>/<your-repo>.git
git push github main
```

Then edit **Cell 1** in the notebook — change `REPO_URL` to your GitHub URL:

```python
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"
```
