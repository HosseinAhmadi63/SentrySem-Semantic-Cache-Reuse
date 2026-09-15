# PyCharm setup

## Open the project

1. Start PyCharm and choose **Open**.
2. Select the `SentrySem-Semantic-Cache-Reuse` directory, which contains `main.py` and `pyproject.toml`.
3. Allow PyCharm to finish indexing before running an experiment.

## Create the interpreter

The reference interpreter is Python 3.12.

1. Open **Settings | Project | Python Interpreter**.
2. Choose **Add Interpreter | Add Local Interpreter | Virtualenv**.
3. Select Python 3.12 as the base interpreter.
4. Set the environment location to `.venv` inside the repository.
5. Open PyCharm's terminal at the repository root and run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The source layout is configured by `pyproject.toml`; no manual source-root changes are required.

## Shared run configurations

The `.run/` directory provides six project-level launchers:

| PyCharm configuration | Equivalent command | Output |
|---|---|---|
| `SentrySem Verify` | `python main.py verify` | Read-only validation |
| `SentrySem Smoke` | `python main.py smoke` | `results/smoke/` |
| `SentrySem Reproduce Paper` | `python main.py reproduce --config configs/paper.json --overwrite` | `results/generated/` and `figures/generated/` |
| `SentrySem Figures` | `python main.py figures` | `figures/generated/` |
| `SentrySem Extract Embeddings` | `python main.py extract-embeddings --config configs/paper.json --output artifacts/generated_inputs` | `artifacts/generated_inputs/` |
| `SentrySem Reproduce Extracted Embeddings` | `python main.py reproduce --config configs/paper.json --inputs artifacts/generated_inputs --overwrite` | `results/generated/` and `figures/generated/` |

Select a configuration in the top toolbar and choose **Run**. Every launcher uses the repository root as its working directory and the selected project interpreter.

## Full raw-image reproduction

Run `SentrySem Extract Embeddings` first. This route obtains CIFAR-10 and the pretrained TorchVision checkpoint when they are absent from the machine, then writes a new input artifact set under `artifacts/generated_inputs/`.

Install the raw-image dependencies once before using this configuration:

```bash
python -m pip install -e ".[raw]"
```

After extraction completes, run `SentrySem Reproduce Extracted Embeddings` to execute the downstream experiments from the regenerated inputs.

The resulting numerical tables are written under `results/generated/`; the checked-in paper artifacts remain unchanged.

## Tests and debugging

Run the complete test suite in PyCharm by right-clicking the `tests` directory and selecting **Run 'pytest in tests'**. Individual protocol functions can be debugged directly because the repository is installed in editable mode.

For deterministic debugging, retain the seed values in `configs/paper.json`. The package derives component-specific random streams from those values, so stepping through one component does not alter unrelated experiment streams.

## Memory and storage

The frozen-input route uses the compact feature archive in `artifacts/frozen_inputs/` and is suitable for CPU execution. The raw-image route additionally stores CIFAR-10, the ResNet-18 checkpoint, and generated features. PyCharm's working directory should remain the repository root so all relative artifact paths resolve consistently.
