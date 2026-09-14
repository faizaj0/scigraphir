# TOMATO-Star training notebook (template)

`colab_train_v16sc_fusion_greasoner.ipynb` is the original Colab notebook that trained the
fusion model on TOMATO-Star. The SIR-4 notebook generators in `sir4-retrieval/prep/`
(`build_notebook.py`, `build_baselines_notebook.py`) copy its engine-install, fusion-model
and Qwen3 cells verbatim so that the TOMATO-Star and SIR-4 runs cannot drift. Keep it here
if you regenerate notebooks; run the generated notebooks in `sir4-retrieval/notebooks/`
for the thesis experiments.
