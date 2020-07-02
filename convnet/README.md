# Conv net for M5
- 1D weekly, b-weekly, monthly and last week/bi-week convs over historic sales
- Self attention didn't work
- Concat conv net output with dense net output for current sales
- iid helps, radomly sample data instead of following the timeseries 
- Scores very well, around ~0.54 on public LB (without including validation data, for now it does include validation data as this is for the final submission) but scored ~0.7xxx on private LB, todo: Fix overfitting.