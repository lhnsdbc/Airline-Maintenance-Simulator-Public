# Publication Checklist

Run this checklist before pushing the repository to a public remote.

## Data Safety

- Confirm generated `Data/`, `artifacts/`, and `reports/` are ignored by Git.
- Confirm no private source data, derived real-input bundles, model weights, PDFs, or server handoff files are staged.
- Run the sensitive-term scan:

```powershell
rg -n -S "<private-source-term-pattern>" -g "!docs/DATA_NOTICE.md" -g "!docs/PUBLICATION_CHECKLIST.md" -g "!Data/**" -g "!artifacts/**" -g "!reports/**"
```

Expected result: no matches.

## Functional Smoke Test

```powershell
py generate_dummy_data.py
py generate_mock_nr_artifact.py
py -m experiments.synthetic_experiment --scenario default_run --seed 20260706
py -m analyst.experiment_report default_run_comparison_seed20260706
py -m analyst.live_llm default_run_comparison_seed20260706 --provider deterministic
py -m retrieval.vector build --backend local
py -m retrieval.vector search "predicted uncovered"
py scripts/smoke_public_workflow.py
py -m unittest discover -s tests -v
```

Optional service checks:

```powershell
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`
- `http://127.0.0.1:8000/search?q=predicted%20uncovered`
- `http://127.0.0.1:8000/rag/search?q=predicted%20uncovered`

For public deployment, follow [docs/DEPLOYMENT.md](DEPLOYMENT.md) and save the deployed API/dashboard URLs before adding them to a CV.

## Git Identity

Check whether the commit author email is appropriate for a public GitHub profile:

```powershell
git log -1 --format="%an <%ae>"
```

If needed, set a public email before future commits:

```powershell
git config user.email "your-public-email@example.com"
```

## Remote Publishing

Use a fresh public remote. Do not push the private repository or its Git history.

```powershell
git remote add origin <new-public-repo-url>
git push -u origin main
```
