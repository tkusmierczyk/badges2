METHODS' NAMES:
BT  - bootstrap test
WT  - theoretic (Wilks') test
2SB - 2-phase (=2-step) with bootstrap
2SW - 2-phase (=2-step) with theotetic (Wilk's)
2P  - 2-poisson 
LR0 - logistic regression applied to BT results
LR1 - logistic regression applied to 2SB results

PREDEFINED WORKING MODES:
mode=="trends": #trend strength
mode=="influence_frac": #fraction of impacted users       
mode=="proportion": #proportion between training and testing users       
mode==101 / 102: #studying impact of differences in intensities and covariates between user groups
mode=="test_influence_frac": #keep fixed proportion of influenced in training but change in testing between 0.1 - 0.9 

THE MEANING OF THE PARAMETERS FOR SYNTETHIC DATA GENERATION:
(CHECK OUT config.py AND IN sample_data.py)
F = fraction of influenced (chaning parameters) users
TF = fraction of test users
ishift = intensity shift (of mean)
covclust = covariates clustering mode = how covariates of influenced/not-influenced are generated
    default 0:  one cluster per each class: not-influenced users in the center, influenced shifted 
    2: not-influenced users in the center, influenced users in two clusters on both sides 

THE MEANING OF THE OTHER PARAMETERS:
twopoisson_priors - selects priors model for two-poisson model: use twopoisson_priors=nn for Neural Networks, default logistic    
    
#NHST EM2 bootstrap
sigma = weighting priors 


