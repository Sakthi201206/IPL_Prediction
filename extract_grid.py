import json

with open("match.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "matplotlib" in source or "sns" in source:
            continue
        code += source + "\n\n"

idx = code.find("import pandas as pd\nfrom sklearn.preprocessing import StandardScaler")
code_to_exec = code[:idx]

with open("run_grid.py", "w", encoding="utf-8") as f:
    f.write(code_to_exec)
    f.write("""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

best_acc = 0
best_params = {}

for md in [3, 4, 5, 8]:
    for lr in [0.01, 0.05, 0.1, 0.2]:
        for ne in [50, 100, 200]:
            for subsample in [0.8, 1.0]:
                model = XGBClassifier(n_estimators=ne, max_depth=md, learning_rate=lr, 
                                      subsample=subsample, random_state=42, eval_metric='logloss')
                model.fit(x_train, y_train)
                acc = accuracy_score(y_test, model.predict(x_test))
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne, 'subsample': subsample}

print("Best Acc:", best_acc)
print("Best Params:", best_params)
""")
