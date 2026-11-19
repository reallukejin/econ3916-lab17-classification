# econ3916-lab17-classification
# Classification: When 90% Accurate Is Worthless

## Objective
I wanted to show why accuracy is a poor guide for a rare-event classifier, and to choose a loan-approval threshold based on what each kind of error actually costs.

## Methodology
- Simulated 6,000 loan applications from a known logistic formula using log income, debt-to-income ratio and number of prior defaults, then split them 70/30 into training and test sets, stratified on default.
- Scored a baseline rule that approves everyone, then fitted a logistic regression (scaled features, scikit-learn pipeline) and compared the two at the default 0.5 threshold using accuracy and a confusion matrix.
- Refitted the model in statsmodels to read the coefficients as log-odds and as odds ratios, and converted one odds ratio into probability changes at starting probabilities of 2%, 10% and 40%.
- Traced precision, recall and accuracy across six thresholds (0.50 down to 0.04).
- Priced the two errors: a false negative (approving a loan that defaults) costs $12,000, and a false positive (declining a good applicant) costs $1,600. Computed the closed-form threshold, FP / (FP + FN).
- Searched thresholds from 0.010 to 0.600 on out-of-fold training probabilities from 5-fold cross-validation, picked the cheapest, and priced it on the test set once.
- Wrote a function, `precision_recall_at()`, that returns a classifier's precision and recall at any threshold without refitting the model.
- Extension: refitted the model with balanced class weights and compared it with simply moving the threshold.

## Key Findings
- **Accuracy hides the problem.** The default rate was 9.6%, so approving everyone is already 90.4% accurate. The fitted model matched that exactly (90.4%) at 0.5 because it predicted 0 defaults: it approved all 172 defaulters in the test set.
- **Odds ratios are not probability changes.** One more prior default multiplies the odds of default by 1.63, but that moves the probability from 2% to 3.2%, from 10% to 15.3%, or from 40% to 52.0% depending on where the applicant starts.
- **The threshold is a business decision.** The cost-minimizing threshold chosen on the training rows was 0.121, close to the closed-form 0.118. On the test set it cost 944 per applicant, against 1,147 at 0.5, so the default threshold costs about 21% more.
- **At the closed-form threshold**, precision was 0.181 and recall was 0.477: the model declines some good applicants but catches almost half of the defaults, where it caught none at 0.5.
- **Calibration (extension).** Balancing the classes pushed the average predicted probability of default to 46.7%, against a true rate of 9.6% (the original model averaged 9.6%). Its ranking ability barely changed (AUC 0.674 vs 0.675), and its cheapest threshold cost 953 per applicant against 944 for the original model. Moving the threshold did the same job and kept the probabilities meaningful.
