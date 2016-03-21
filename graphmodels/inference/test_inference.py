import graphmodels as gm
import numpy as np
import sys

def generate_query(dgm, n_query, n_evidence):
    selected = np.random.choice(dgm.nodes(), size=n_query + n_evidence, replace=False)
    query = selected[:n_query]
    evidence = dict((e, np.random.choice(dgm.node[e]['cpd'].values(e))) for e in selected[n_query:])
    return query, evidence


def test_sum_product_inference():
    n_tests = 5
    dgm = gm.DGM.read('../../networks/earthquake.bif')
    inference = gm.SumProductInference(dgm)
    naive_inference = gm.NaiveInference(dgm)
    for n_query in range(1, 1 + len(dgm.nodes())):
        for n_evidence in range(0, 1 + len(dgm.nodes()) - n_query):
            for test_i in range(n_tests):
                query = generate_query(dgm, n_query=n_query, n_evidence=n_evidence)
                #print(query)
                sys.stdout.flush()
                query = (np.array(['MaryCalls'],
      dtype='<U10'), {'Burglary': 'True', 'JohnCalls': 'True'})
                assert inference(*query) == naive_inference(*query)

test_sum_product_inference()
