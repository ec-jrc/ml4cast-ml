#https://mapie.readthedocs.io/en/stable/examples_regression/2-advanced-analysis/plot_nested-cv.html#sphx-glr-examples-regression-2-advanced-analysis-plot-nested-cv-py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_sparse_uncorrelated

from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor


random_state = 42

# Load the toy data
X, y = make_sparse_uncorrelated(500, random_state=random_state)

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Define the Random Forest model as base regressor with parameter ranges.
rf_model = RandomForestRegressor(random_state=random_state, verbose=0)
rf_params = {"max_depth": randint(2, 10), "n_estimators": randint(10, 100)}

# Cross-validation and prediction-interval parameters.
cv = 10
n_iter = 5
alpha = 0.05

# Non-nested approach with the CV+ strategy using the Random Forest model.
cv_obj = RandomizedSearchCV(
    rf_model,
    param_distributions=rf_params,
    n_iter=n_iter,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,
    verbose=0,
    random_state=random_state,
    n_jobs=-1,
)
cv_obj.fit(X_train, y_train)
best_est = cv_obj.best_estimator_
mapie_non_nested = MapieRegressor(
    best_est, method="plus", cv=cv, agg_function="median", n_jobs=-1,
    random_state=random_state
)
mapie_non_nested.fit(X_train, y_train)
y_pred_non_nested, y_pis_non_nested = mapie_non_nested.predict(
    X_test, alpha=alpha
)
widths_non_nested = y_pis_non_nested[:, 1, 0] - y_pis_non_nested[:, 0, 0]
coverage_non_nested = regression_coverage_score(
    y_test, y_pis_non_nested[:, 0, 0], y_pis_non_nested[:, 1, 0]
)
#score_non_nested = mean_squared_error(y_test, y_pred_non_nested, squared=False)
score_non_nested = root_mean_squared_error(y_test, y_pred_non_nested)



# Print scores and effective coverages.
print(
    "Scores and effective coverages for the CV+ strategy using the "
    "Random Forest model."
)
print(
    "Score on the test set for the non-nested : ",
    f"{score_non_nested: .3f}",
)
print(
    "Effective coverage on the test set for the non-nested :"

    f"{coverage_non_nested: .3f}",
)

# # Compare prediction interval widths.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
# min_x = np.min([np.min(widths_nested), np.min(widths_non_nested)])
# max_x = np.max([np.max(widths_nested), np.max(widths_non_nested)])
# ax1.set_xlabel("Prediction interval width using the nested CV approach")
# ax1.set_ylabel("Prediction interval width using the non-nested CV approach")
# ax1.scatter(widths_nested, widths_non_nested)
# ax1.plot([min_x, max_x], [min_x, max_x], ls="--", color="k")
# ax2.axvline(x=0, color="r", lw=2)
# ax2.set_xlabel(
#     "[width(non-nested CV) - width(nested CV)] / width(non-nested CV)"
# )
# ax2.set_ylabel("Counts")
# ax2.hist(
#     (widths_non_nested - widths_nested) / widths_non_nested,
#     bins=15,
#     edgecolor="black",
# )
# plt.show()