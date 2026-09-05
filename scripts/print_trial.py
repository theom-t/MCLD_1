import optuna
study = optuna.load_study(study_name="MCLD-1-Joint-Architecture", storage="sqlite:///mcld1_optuna.db")
for t in study.trials:
    if t.number == 275:
        print(t.params)
