import pickle
import os

Name = 'HLA_TCR'

with open(os.path.join(Name, 'models', 'model_type.pkl'), 'rb') as f:
    model_type, get, use_alpha, use_beta, \
    use_v_beta, use_d_beta, use_j_beta, \
    use_v_alpha, use_j_alpha, use_hla, use_hla_sup, keep_non_supertype_alleles, \
    lb_v_beta, lb_d_beta, lb_j_beta, \
    lb_v_alpha, lb_j_alpha, lb_hla, lb, \
    ind, regression = pickle.load(f)

max_length = 40

with open(os.path.join(Name, 'models', 'model_type.pkl'), 'wb') as f:
    pickle.dump([model_type, get, use_alpha, use_beta,
    use_v_beta, use_d_beta, use_j_beta,
    use_v_alpha, use_j_alpha, use_hla, use_hla_sup, keep_non_supertype_alleles,
    lb_v_beta, lb_d_beta, lb_j_beta,
    lb_v_alpha, lb_j_alpha, lb_hla, lb,
    ind, regression, max_length],f,protocol=4)

with open(os.path.join(Name, 'models', 'model_type.pkl'), 'rb') as f:
    model_type, get, use_alpha, use_beta, \
    use_v_beta, use_d_beta, use_j_beta, \
    use_v_alpha, use_j_alpha, use_hla, use_hla_sup, keep_non_supertype_alleles, \
    lb_v_beta, lb_d_beta, lb_j_beta, \
    lb_v_alpha, lb_j_alpha, lb_hla, lb, \
    ind, regression,max_length = pickle.load(f)
