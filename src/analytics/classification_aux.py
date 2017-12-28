
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
import logging
import sys
sys.path.append("../")


from aux.events_io import load_events
from classification import nhst_with_em2
from classification import nhst_testing

EPS = 10**(-18)


def extract_train_test(samples, user2features, max_train=10000, max_test=10000):
    
    id2events = load_events(samples)
    ids = set(user2features["id"])
    id2events = dict((i,e) for i, e in id2events.items() if i in ids)
    
    busers = [i for i, e in id2events.items() if "switch_time" in e]
    bfeatures = user2features[user2features.id.isin(busers)]#["age"]
    bfeatures = bfeatures[~bfeatures.isnull().any(axis=1)]
    busers = list(set(bfeatures.id))
    
    nusers = [i for i, e in id2events.items() if "research_assistant" not in e]
    nfeatures = user2features[user2features.id.isin(nusers)]
    nfeatures = nfeatures[~nfeatures.isnull().any(axis=1)]
    nusers = list(set(nfeatures.id))
    
    #pyplot.hist(bfeatures.age, bins=np.array(range(50))*2, label="users with badge", fill=None, normed=True, histtype="step", lw=3)
    #pyplot.hist(nfeatures.age, bins=np.array(range(50))*2, label="users without badge", normed=True, histtype="step", lw=3)
    #pyplot.legend()
    #pyplot.xlabel("age")
    #pyplot.xlim((10,60))
        
    np.random.shuffle(nusers)
    train_ids, test_ids = busers[ :max_train], nusers[ :max_test]
    features = nfeatures.append(bfeatures, ignore_index=True)
    
    #print("features=",list(enumerate(features.columns[1:])))    
    return features, train_ids, test_ids



def em2_classification_cached(features, samples, train_ids, test_ids, 
                       sigma=1, kappa=1,
                       nhst_preds_file="???",
                       test=nhst_testing.bootstrap_pvalue):
    #nhst classification
    recalculate = False
    try:
        id2nhst = pd.read_csv(nhst_preds_file, sep="\t")
        logging.info("NHST predictions loaded from %s" % nhst_preds_file)
        missing_ids = set(train_ids).difference(set(id2nhst["id"]))
        if len(missing_ids)>0:
            logging.warn("Missing ids: %s" % missing_ids)
            recalculate = True
    except:
        recalculate = True

    if recalculate:
        nhst_ids = list(set(samples[samples["type"]=="switch_time"]["id"]))
        logging.info("Recalculating NHST predictions for %i users" % len(nhst_ids))        
        train_preds, _ = nhst_with_em2.nhst_classification(
                                            pd.DataFrame(), samples, 
                                            nhst_ids, [], 
                                            test=test, 
                                            badge_name="switch_time", 
                                            pvalue_threshold=0.05, cpus=1)
        id2nhst = pd.DataFrame({"id": nhst_ids, "nhst_pred": train_preds})
        id2nhst.to_csv(nhst_preds_file, sep="\t", index=False, header=True)
    id2nhst = dict(zip(id2nhst["id"], id2nhst["nhst_pred"]))
    nhst_preds = np.array([id2nhst[i] for i in train_ids])
    
    logging.info("Clustering")
    train_preds, test_preds = nhst_with_em2.two_step_classification(features, samples, 
                                                                    train_ids, test_ids,
                                                                    sigma=sigma, kappa=kappa,
                                                                    train_preds=nhst_preds)
    em2_results = nhst_with_em2.two_step_classification.intermediate_results
    
    return train_preds, test_preds, em2_results


def print_results(train_preds, test_preds=[], margin = 0.0):    
    train_preds, test_preds = np.asarray(train_preds), np.asarray(test_preds)
    print("train_preds0 = %i = %.2f" % (sum(train_preds<0.5-margin), sum(train_preds<0.5-margin)/len(train_preds)))
    print("train_preds1 = %i = %.2f" % (sum(train_preds>0.5+margin), sum(train_preds>0.5+margin)/len(train_preds)))
    print("test_preds0 = %i = %.2f" % (sum(test_preds<0.5-margin), sum(test_preds<0.5-margin)/(len(test_preds)+EPS)))
    print("test_preds1 = %i = %.2f" % (sum(test_preds>0.5+margin), sum(test_preds>0.5+margin)/(len(test_preds)+EPS)))
    

def agreement_pvalue(p1, p2, N, agreement, B=10000):
    ts = []
    for _ in range(B):
        r1 = bernoulli.rvs(p1, size=N)
        r2 = bernoulli.rvs(p2, size=N)
        ts.append( sum(r1==r2)/len(r1) )
    ts = np.array(ts)
    return float(sum(ts>agreement))/B


def compare_results(nhst_train_preds, train_preds, B=3000):
    if len(nhst_train_preds)==0 or len(train_preds)==0: return
    from scipy.stats.stats import pearsonr
    print("\npearsonr=", pearsonr(nhst_train_preds, train_preds))
    
    threshold = 0.5
    true  = np.asarray((np.asarray(nhst_train_preds)>threshold), dtype=int)
    preds = np.asarray((np.asarray(train_preds)>threshold), dtype=int)
    TP = sum((true == 1) & (preds == 1))
    FP = sum((true == 0) & (preds == 1))
    TN = sum((true == 0) & (preds == 0))
    FN = sum((true == 1) & (preds == 0))
    
    print("G:", TP, TN, FP, FN)
    p1, p2 = float(sum(true))/(len(true)+EPS), float(sum(preds))/(len(preds)+EPS)
    N = (TP+TN+FP+FN) 
    agreement = float(TP+TN)/((TP+TN+FP+FN)+EPS)
    print("p1=",p1,"\np2=",p2)
    print("N=", N, "\nagreement=", agreement)
    print("p-value=", agreement_pvalue(p1, p2, N, agreement, B))
    